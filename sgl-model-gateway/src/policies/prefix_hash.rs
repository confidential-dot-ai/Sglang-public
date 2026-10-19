//! Prefix Hash routing policy for KV cache-aware load balancing
//!
//! A lightweight alternative to the full radix tree cache_aware policy.
//! Routes requests based on a hash of their prefix tokens to maximize
//! KV cache hits across workers.
//!
//! ## Algorithm
//!
//! 1. Extract first N tokens from the request (configurable prefix length)
//! 2. Hash the token sequence using xxhash for fast, stable hashing
//! 3. Use consistent hash ring to find the target worker
//! 4. If worker is overloaded (load > avg * load_factor), find least loaded
//! 5. Return least loaded worker that passes load check, or initial if all overloaded
//!
//! ## Complexity
//!
//! - Hash computation: O(prefix_length)
//! - Ring lookup: O(log n) binary search
//! - Load balance fallback: O(n) scan for least loaded
//!
//! ## Comparison with cache_aware
//!
//! | Aspect          | prefix_hash       | cache_aware (radix) |
//! |-----------------|-------------------|---------------------|
//! | Lookup          | O(log n)          | O(prefix_len)       |
//! | Memory          | O(workers × vn)   | O(total_tokens)     |
//! | Update          | O(1)              | O(prefix_len)       |
//! | Precision       | Prefix grouping   | Exact matching      |
//!
//! prefix_hash trades optimal cache utilization for predictable O(log n) performance.

use std::sync::Arc;

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, observability::metrics::Metrics};

/// Configuration for the PrefixHash load balancing policy
#[derive(Debug, Clone)]
pub struct PrefixHashConfig {
    /// Number of prefix tokens to use for hashing.
    /// Longer prefixes = more precise routing but less grouping.
    /// Shorter prefixes = more requests grouped together.
    /// Default: 256 tokens (~1 paragraph of text)
    pub prefix_token_count: usize,

    /// Load factor threshold for walking the ring.
    /// If a worker's load > (total_load / num_workers) * load_factor,
    /// walk clockwise to the next worker.
    /// Default: 1.25 (125% of average load)
    pub load_factor: f64,

    /// KV/token-pool occupancy fraction (from Worker::kv_occupancy(), kept
    /// fresh by the KvOccupancyPoller) at or above which a worker is
    /// treated as overloaded regardless of request count. Default 0.90:
    /// leaves headroom for the in-flight requests a worker already holds
    /// to finish without OOM, while catching the "1 huge request, pool
    /// nearly full" case load_factor's request-count check can't see.
    pub kv_occupancy_threshold: f64,
}

impl Default for PrefixHashConfig {
    fn default() -> Self {
        Self {
            prefix_token_count: 256,
            load_factor: 1.25,
            kv_occupancy_threshold: 0.90,
        }
    }
}

/// Execution branch for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    NoTokens,
    RingHit,
    LoadBalanceWalk,
    FallbackLeastLoad,
}

impl Branch {
    #[inline]
    const fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::NoTokens => "no_tokens",
            Self::RingHit => "ring_hit",
            Self::LoadBalanceWalk => "load_balance_walk",
            Self::FallbackLeastLoad => "fallback_least_load",
        }
    }
}

/// Prefix Hash load balancing policy
///
/// Routes requests based on prefix token hash for KV cache locality.
/// Uses consistent hashing with bounded load balancing.
#[derive(Debug)]
pub struct PrefixHashPolicy {
    config: PrefixHashConfig,
}

impl PrefixHashPolicy {
    /// Create a new PrefixHashPolicy with the given configuration
    pub fn new(config: PrefixHashConfig) -> Self {
        Self { config }
    }

    /// Create a new PrefixHashPolicy with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PrefixHashConfig::default())
    }

    /// Compute hash of prefix tokens using xxhash
    #[inline]
    fn compute_prefix_hash(&self, tokens: &[u32]) -> u64 {
        let prefix_len = tokens.len().min(self.config.prefix_token_count);
        let prefix = &tokens[..prefix_len];

        let bytes: &[u8] = bytemuck::cast_slice(prefix);
        xxhash_rust::xxh3::xxh3_64(bytes)
    }

    /// Check if a worker's load is acceptable
    #[inline]
    fn load_ok(&self, worker_load: usize, total_load: usize, num_workers: usize) -> bool {
        if total_load == 0 || num_workers == 0 {
            return true;
        }

        // Average load per worker (with +1 to simulate incoming request)
        let avg_load = (total_load + 1) as f64 / num_workers as f64;
        let threshold = avg_load * self.config.load_factor;

        (worker_load as f64) <= threshold
    }

    /// Combined overload check: request-count load_ok, AND KV/token-pool
    /// occupancy. A worker can look idle by request count (load()) while
    /// sitting near-full on KV -- one long-context conversation is enough
    /// (observed live, 2026-07-24: 85-95% occupancy at 1-3 running
    /// requests). load_ok alone never catches that; this does, without
    /// touching the ring-affinity logic itself -- an occupancy-overloaded
    /// worker is just treated as "overloaded" the same way a
    /// request-count-overloaded one already is, falling through to the
    /// existing least-loaded/rendezvous spill.
    #[inline]
    fn worker_ok(&self, worker: &Arc<dyn Worker>, total_load: usize, num_workers: usize) -> bool {
        worker.kv_occupancy() < self.config.kv_occupancy_threshold
            && self.load_ok(worker.load(), total_load, num_workers)
    }

    /// Find worker using consistent hash ring with load balancing
    fn find_worker_with_load_balance(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        prefix_hash: u64,
    ) -> (Option<usize>, Branch) {
        // Build healthy worker URL to index map
        let healthy_workers: Vec<(usize, &Arc<dyn Worker>)> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .collect();

        if healthy_workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Calculate total load for load balancing
        let total_load: usize = healthy_workers.iter().map(|(_, w)| w.load()).sum();
        let num_workers = healthy_workers.len();

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            // Convert prefix hash to a ring key string for lookup
            let key = format!("{:016x}", prefix_hash);

            // Build URL to (index, worker) map for healthy workers
            let healthy_url_map: std::collections::HashMap<&str, (usize, &Arc<dyn Worker>)> =
                healthy_workers
                    .iter()
                    .map(|(idx, w)| (w.url(), (*idx, *w)))
                    .collect();

            // Find initial worker from ring
            if let Some(initial_url) =
                ring.find_healthy_url(&key, |url| healthy_url_map.contains_key(url))
            {
                if let Some(&(idx, worker)) = healthy_url_map.get(initial_url) {
                    // Check if initial worker has acceptable load AND KV occupancy
                    if self.worker_ok(worker, total_load, num_workers) {
                        return (Some(idx), Branch::RingHit);
                    }

                    // Initial worker overloaded, find least loaded healthy worker
                    // This is a simpler approach than walking the ring
                    let least_loaded = healthy_workers
                        .iter()
                        .filter(|(_, w)| self.worker_ok(w, total_load, num_workers))
                        .min_by_key(|(_, w)| w.load());

                    if let Some(&(idx, _)) = least_loaded {
                        return (Some(idx), Branch::LoadBalanceWalk);
                    }

                    // All workers overloaded, use initial worker anyway
                    return (Some(idx), Branch::LoadBalanceWalk);
                }
            }
        }

        // Fallback: no ring or ring lookup failed. Rendezvous-hash the
        // prefix over the healthy workers instead of picking least-loaded:
        // load values are frequently tied (or stale), and min_by_key on a
        // tie returns the FIRST worker every time -- observed live
        // (2026-07-24) as a total collapse of all decode traffic onto one
        // engine whenever the ring lookup missed. Rendezvous keeps the
        // same-prefix -> same-worker affinity the ring would have given
        // (stable across worker-set changes for surviving workers) while
        // spreading distinct prefixes evenly. The bounded-load check still
        // applies: an overloaded rendezvous winner falls through to the
        // least-loaded acceptable worker, same as the ring path.
        let mut ranked: Vec<(usize, &Arc<dyn Worker>, u64)> = healthy_workers
            .iter()
            .map(|(idx, w)| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&prefix_hash, &mut hasher);
                std::hash::Hash::hash(&w.url(), &mut hasher);
                (*idx, *w, std::hash::Hasher::finish(&hasher))
            })
            .collect();
        ranked.sort_by(|a, b| b.2.cmp(&a.2));

        for (idx, worker, _) in &ranked {
            if self.worker_ok(worker, total_load, num_workers) {
                return (Some(*idx), Branch::FallbackLeastLoad);
            }
        }
        // All overloaded: keep the rendezvous winner (same policy as the
        // ring path's "all workers overloaded, use initial worker anyway").
        (ranked.first().map(|(idx, _, _)| *idx), Branch::FallbackLeastLoad)
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Get tokens from SelectWorkerInfo
        let tokens = match info.tokens {
            Some(t) if !t.is_empty() => t,
            _ => return (None, Branch::NoTokens),
        };

        // Compute prefix hash
        let prefix_hash = self.compute_prefix_hash(tokens);

        // Find worker using ring with load balancing
        self.find_worker_with_load_balance(workers, info, prefix_hash)
    }
}

#[async_trait::async_trait]
impl LoadBalancingPolicy for PrefixHashPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_prefix_hash_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "prefix_hash"
    }

    fn prefix_tokens_needed(&self) -> Option<usize> {
        Some(self.config.prefix_token_count)
    }

    fn wants_system_prompt_excluded(&self) -> bool {
        true
    }

    fn needs_request_text(&self) -> bool {
        // Required on the HTTP path: the router only extracts request text
        // when some active policy asks for it, and that text is what
        // tokenize_bounded_prefix turns into the token ids this policy
        // routes on. Without this override, a topology where BOTH policies
        // are prefix_hash gets no text at all (previously masked because
        // the unset decode policy fell back to CacheAware, whose
        // needs_request_text() is true).
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, HashRing, WorkerType};

    fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_prefix_hash_consistent_routing() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Same tokens should always route to same worker
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (first_result, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Verify consistency
        for _ in 0..10 {
            let (result, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(first_idx));
        }
    }

    #[test]
    fn test_kv_occupancy_overload_spills_to_another_worker() {
        // The exact live failure mode this closes: a worker with load()==0
        // (no in-flight requests by count) but pinned near-full on KV from
        // one huge conversation must NOT keep winning the ring lookup.
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        let tokens: Vec<u32> = vec![11, 22, 33, 44, 55];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        let winner_idx = result.unwrap();
        assert_eq!(branch, Branch::RingHit);

        // Saturate the ring winner's KV occupancy, request count untouched.
        workers[winner_idx].set_kv_occupancy(0.97);

        let (result2, branch2) = policy.select_worker_impl(&workers, &info);
        assert_ne!(
            result2.unwrap(),
            winner_idx,
            "occupancy-saturated worker must not keep winning"
        );
        assert_eq!(branch2, Branch::LoadBalanceWalk);
    }

    #[test]
    fn test_kv_occupancy_below_threshold_does_not_spill() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        let tokens: Vec<u32> = vec![11, 22, 33, 44, 55];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let winner_idx = result.unwrap();

        // Below the 0.90 default threshold: must NOT spill.
        workers[winner_idx].set_kv_occupancy(0.5);
        let (result2, branch2) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result2.unwrap(), winner_idx);
        assert_eq!(branch2, Branch::RingHit);
    }

    #[test]
    fn test_all_workers_kv_saturated_still_returns_a_worker() {
        // Degrade gracefully (same policy as "all workers overloaded" on
        // request count): never return None just because every worker is
        // KV-saturated -- a request must still be routed somewhere.
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        for w in &workers {
            w.set_kv_occupancy(0.99);
        }
        let ring = Arc::new(HashRing::new(&workers));
        let tokens: Vec<u32> = vec![1, 2, 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let (result, _) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
    }

    #[test]
    fn test_ringless_fallback_respects_kv_occupancy() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let tokens: Vec<u32> = vec![42, 43, 44, 45];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: None,
            ..Default::default()
        };
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let winner_idx = result.unwrap();

        workers[winner_idx].set_kv_occupancy(0.97);
        let (result2, _) = policy.select_worker_impl(&workers, &info);
        assert_ne!(result2.unwrap(), winner_idx);
    }

    #[test]
    fn test_different_prefixes_distribute() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        let mut distribution = std::collections::HashMap::new();

        // Different token sequences should distribute across workers
        for i in 0..100 {
            let tokens: Vec<u32> = vec![i, i + 1, i + 2, i + 3];
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };

            let (result, _) = policy.select_worker_impl(&workers, &info);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across workers, got {:?}",
            distribution
        );
    }

    #[test]
    fn test_ringless_fallback_distributes_and_keeps_affinity() {
        // hash_ring=None is the live shape whenever the registry ring lookup
        // misses (2026-07-24: every non-igw PD deploy). The old fallback
        // (min_by_key on load) collapsed ALL traffic onto one worker on tied
        // loads; the rendezvous fallback must (a) spread distinct prefixes,
        // (b) keep same-prefix -> same-worker affinity.
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let mut distribution = std::collections::HashMap::new();
        for i in 0..100u32 {
            let tokens: Vec<u32> = vec![i * 7 + 1, i * 13 + 2, i * 29 + 3, i];
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: None,
                ..Default::default()
            };
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch.as_str(), Branch::FallbackLeastLoad.as_str());
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }
        assert!(
            distribution.len() == 3 && distribution.values().all(|&c| c >= 10),
            "ringless fallback must spread across all 3 workers, got {:?}",
            distribution
        );

        // Affinity: the same tokens always land on the same worker.
        let tokens: Vec<u32> = vec![42, 43, 44, 45];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: None,
            ..Default::default()
        };
        let (first, _) = policy.select_worker_impl(&workers, &info);
        for _ in 0..10 {
            let (again, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(again, first);
        }
    }

    #[test]
    fn test_shared_prefix_routes_same() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 5, // Only look at first 5 tokens
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Two sequences with same first 5 tokens should route to same worker
        let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
        let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 999, 888, 777];

        let info1 = SelectWorkerInfo {
            tokens: Some(&tokens1),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            tokens: Some(&tokens2),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result1, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _) = policy.select_worker_impl(&workers, &info2);

        assert_eq!(result1, result2, "Same prefix should route to same worker");
    }

    #[test]
    fn test_no_tokens_returns_none() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Empty tokens
        let tokens: Vec<u32> = vec![];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoTokens);

        // No tokens field
        let info_no_tokens = SelectWorkerInfo {
            tokens: None,
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result2, branch2) = policy.select_worker_impl(&workers, &info_no_tokens);
        assert_eq!(result2, None);
        assert_eq!(branch2, Branch::NoTokens);
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        workers[0].set_healthy(false);

        let ring = Arc::new(HashRing::new(&workers));
        let tokens: Vec<u32> = vec![1, 2, 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_load_ok_calculation() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.25,
            ..Default::default()
        });

        // Total load 100, 4 workers -> avg 25, threshold 31.25
        assert!(policy.load_ok(30, 100, 4)); // 30 <= 31.25
        assert!(!policy.load_ok(35, 100, 4)); // 35 > 31.25

        // Edge cases
        assert!(policy.load_ok(0, 0, 4)); // No load = OK
        assert!(policy.load_ok(100, 0, 0)); // No workers = OK (shouldn't happen)
    }

    #[test]
    fn test_policy_name() {
        let policy = PrefixHashPolicy::with_defaults();
        assert_eq!(policy.name(), "prefix_hash");
    }

    /// Regression test for the 2026-07-24 decode-load-collapse bug: real
    /// tokenized first-turn prefixes from 50 conversations in
    /// Wejh/ninja-agent-traces (a SWE-agent trace dataset -- oracle-replay's
    /// actual benchmark workload), captured via the target model's own
    /// tokenizer. 49 of these 50 conversations share an identical first 263
    /// tokens (a boilerplate SWE-agent system prompt) -- at the OLD
    /// prefix_token_count=256, this collapsed nearly all of them onto a
    /// single consistent-hash-ring bucket, defeating 3-way load balancing
    /// regardless of client-side concurrency (confirmed via
    /// bindings/python/src/lib.rs's convert_policy, which hardcodes this
    /// value with no CLI override path). At the fixed 1024, real per-task
    /// content (which begins after the shared boilerplate) enters the hash
    /// window and restores distribution. This test would have caught the
    /// original bug and guards against a future regression back to a
    /// too-short window for this workload shape.
    #[test]
    fn test_prefix_token_count_1024_fixes_agent_trace_collapse() {
        let raw = include_str!("testdata/oracle_conv_prefixes.json");
        let conversations: Vec<Vec<u32>> = serde_json::from_str(raw).unwrap();
        assert_eq!(
            conversations.len(),
            50,
            "fixture should have 50 real conversations"
        );

        let workers = create_workers(&[
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
        ]);
        let ring = Arc::new(HashRing::new(&workers));

        let route_all = |prefix_token_count: usize| -> std::collections::HashMap<usize, usize> {
            let policy = PrefixHashPolicy::new(PrefixHashConfig {
                prefix_token_count,
                ..Default::default()
            });
            let mut counts = std::collections::HashMap::new();
            for tokens in &conversations {
                let info = SelectWorkerInfo {
                    tokens: Some(tokens),
                    hash_ring: Some(ring.clone()),
                    ..Default::default()
                };
                let (result, _) = policy.select_worker_impl(&workers, &info);
                *counts.entry(result.unwrap()).or_insert(0) += 1;
            }
            counts
        };

        let old_counts = route_all(256);
        let old_max = *old_counts.values().max().unwrap();
        assert!(
            old_max >= 45,
            "sanity check that the OLD 256-token window really does collapse \
             this dataset onto ~1 worker (got max cluster {old_max}/50, \
             counts={old_counts:?}) -- if this fails, the fixture no longer \
             reproduces the original bug and this test needs a new dataset"
        );

        let new_counts = route_all(1024);
        let new_max = *new_counts.values().max().unwrap();
        assert!(
            new_max <= 40,
            "prefix_token_count=1024 should meaningfully break up the \
             shared-boilerplate collision (got max cluster {new_max}/50, \
             counts={new_counts:?})"
        );
        assert!(
            new_counts.len() >= 2,
            "prefix_token_count=1024 should spread conversations across \
             more than one worker (got counts={new_counts:?})"
        );
    }
}
