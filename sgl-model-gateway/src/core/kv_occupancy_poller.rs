//! Background poller that keeps each worker's `kv_occupancy()` fresh by
//! hitting the sglang engine's `/get_load` endpoint.
//!
//! Why this exists: `Worker::load()` (in-flight request count) is what
//! prefix_hash's bounded-load routing checks today, but it is blind to KV
//! pool pressure -- a decode worker holding one long-context conversation
//! can sit near 100% KV occupancy while `load()` reads 1, so a request whose
//! prefix hashes to that worker gets routed there anyway (2026-07-24, live:
//! one decode engine parked at 85-95% token usage with 1-3 running requests
//! while its two siblings ran at 500-2000 tok/s). This poller supplies the
//! missing signal so routing can see it.
//!
//! Polls `/v1/loads`, NOT the legacy `/get_load`: the latter's `num_tokens`
//! is `used + queued`, not pool capacity, so an occupancy computed from it
//! reads 1.0 whenever nothing is queued. `/v1/loads` reports the real
//! `token_usage` fraction (plus `max_total_num_tokens` as a fallback).

use std::{
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use serde::Deserialize;
use tracing::{debug, warn};

use super::{worker::ConnectionMode, WorkerRegistry};

/// One /v1/loads entry. NOT /get_load: that legacy projection reports
/// `num_tokens = num_total_tokens`, which is `num_used_tokens + queued
/// tokens` -- NOT the pool capacity. Computing used/num_tokens from it
/// yields 1.0 whenever nothing is queued (verified live 2026-07-24:
/// `num_tokens:113152, num_pending_tokens:0` on an engine the scheduler
/// reported at 10% usage). /v1/loads exposes the real fraction directly.
#[derive(Debug, Deserialize)]
struct LoadEntry {
    #[serde(default)]
    token_usage: Option<f64>,
    #[serde(default)]
    num_used_tokens: Option<f64>,
    #[serde(default)]
    max_total_num_tokens: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct LoadsResponse {
    loads: Vec<LoadEntry>,
}

/// Occupancy fraction from one /v1/loads entry, clamped to [0.0, 1.0].
/// Prefers the engine's own `token_usage`; falls back to
/// num_used_tokens/max_total_num_tokens. Returns None when neither is
/// usable so the caller leaves the last-known value alone rather than
/// writing a fabricated 0.0 ("free") over a real reading.
fn occupancy_from_entry(entry: &LoadEntry) -> Option<f64> {
    if let Some(usage) = entry.token_usage {
        if usage.is_finite() {
            return Some(usage.clamp(0.0, 1.0));
        }
    }
    match (entry.num_used_tokens, entry.max_total_num_tokens) {
        (Some(used), Some(cap)) if cap > 0.0 && used.is_finite() => {
            Some((used / cap).clamp(0.0, 1.0))
        }
        _ => None,
    }
}

/// Highest occupancy across dp_rank entries -- routing must avoid the
/// worker's MOST pressured shard, not the average.
fn occupancy_from_response(entries: &[LoadEntry]) -> Option<f64> {
    entries
        .iter()
        .filter_map(occupancy_from_entry)
        .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
}

pub(crate) struct KvOccupancyPoller {
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
}

impl KvOccupancyPoller {
    pub fn start(registry: Arc<WorkerRegistry>, interval: Duration) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .expect("failed to build KV occupancy poller HTTP client");

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if shutdown_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                for worker in registry.get_all() {
                    // Only HTTP workers expose /v1/loads over this path;
                    // gRPC occupancy would need its own RPC, not built here.
                    if !matches!(worker.connection_mode(), ConnectionMode::Http) {
                        continue;
                    }
                    let url = format!("{}/v1/loads", worker.url());
                    match client.get(&url).send().await {
                        Ok(resp) if resp.status().is_success() => {
                            match resp.json::<LoadsResponse>().await {
                                Ok(parsed) => {
                                    if let Some(occ) = occupancy_from_response(&parsed.loads) {
                                        worker.set_kv_occupancy(occ);
                                    }
                                }
                                Err(e) => {
                                    debug!(
                                        "KV occupancy poll: failed to parse /v1/loads from {}: {}",
                                        worker.url(),
                                        e
                                    );
                                }
                            }
                        }
                        Ok(resp) => {
                            debug!(
                                "KV occupancy poll: {} returned {} for /v1/loads",
                                worker.url(),
                                resp.status()
                            );
                        }
                        Err(e) => {
                            // Transient (worker restarting, network blip) --
                            // leave the last-known occupancy in place rather
                            // than resetting it, since a stale-high value is
                            // safer (keeps steering away) than a stale-zero
                            // one (would look free again).
                            warn!(
                                "KV occupancy poll: request to {} failed: {}",
                                worker.url(),
                                e
                            );
                        }
                    }
                }
            }
        });

        Self { handle, shutdown }
    }

    #[allow(dead_code)]
    pub async fn shutdown(self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        let _ = self.handle.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        token_usage: Option<f64>,
        num_used_tokens: Option<f64>,
        max_total_num_tokens: Option<f64>,
    ) -> LoadEntry {
        LoadEntry {
            token_usage,
            num_used_tokens,
            max_total_num_tokens,
        }
    }

    #[test]
    fn test_prefers_engine_reported_token_usage() {
        assert_eq!(occupancy_from_entry(&entry(Some(0.1046), None, None)), Some(0.1046));
    }

    #[test]
    fn test_falls_back_to_used_over_capacity() {
        // The real shape of a /v1/loads entry from a live engine.
        let e = entry(None, Some(113152.0), Some(1081536.0));
        let occ = occupancy_from_entry(&e).unwrap();
        assert!((occ - 0.1046).abs() < 1e-3, "got {occ}");
    }

    #[test]
    fn test_regression_never_reports_full_when_nothing_queued() {
        // The bug this endpoint switch fixes: /get_load's num_tokens is
        // used+queued, so used/num_tokens was 1.0 with an empty queue --
        // on an engine actually at ~10%. token_usage must win.
        assert_eq!(
            occupancy_from_entry(&entry(Some(0.10), Some(113152.0), Some(1081536.0))),
            Some(0.10)
        );
    }

    #[test]
    fn test_missing_fields_is_none_not_zero() {
        // None => caller keeps the last-known value; a fabricated 0.0 would
        // make a saturated worker look free.
        assert_eq!(occupancy_from_entry(&entry(None, None, None)), None);
        assert_eq!(occupancy_from_entry(&entry(None, Some(5.0), Some(0.0))), None);
        assert_eq!(occupancy_from_entry(&entry(Some(f64::NAN), None, None)), None);
    }

    #[test]
    fn test_clamped_to_unit_range() {
        assert_eq!(occupancy_from_entry(&entry(Some(1.5), None, None)), Some(1.0));
        assert_eq!(occupancy_from_entry(&entry(Some(-0.2), None, None)), Some(0.0));
    }

    #[test]
    fn test_response_takes_max_across_dp_ranks() {
        let entries = vec![
            entry(Some(0.2), None, None),
            entry(Some(0.9), None, None),
            entry(Some(0.1), None, None),
        ];
        assert_eq!(occupancy_from_response(&entries), Some(0.9));
    }

    #[test]
    fn test_response_empty_or_all_unusable_is_none() {
        assert_eq!(occupancy_from_response(&[]), None);
        assert_eq!(occupancy_from_response(&[entry(None, None, None)]), None);
    }
}
