//! Inference-quality metrics for provider bypass-traffic validation.
//!
//! The quality inspection manual requires concurrent service monitoring across
//! these consistency dimensions:
//!
//!   success rate | TTFT, TPS | input length | output length distribution |
//!   reasoning length distribution and reasoning triggering | cache hit rate |
//!   tool invocation, parallel calls, and invalid parameters
//!
//! plus a non-stop ratio (share of successful requests that exhaust
//! max_tokens), which is a pass/fail threshold rather than an observation.
//!
//! `observability::metrics` covers success rate and request duration. It also
//! defines `record_router_ttft`, `record_router_tpot` and `record_router_tokens`
//! -- but nothing anywhere calls them, so TTFT, TPOT and token counts are
//! described in the Prometheus registry and never recorded. Everything in this
//! module is therefore new observation, not a re-export.
//!
//! # Cost
//!
//! The router is on the hot path for every request, so this is built to be
//! close to free:
//!
//! * Non-streaming responses are already fully in memory as `Bytes` before
//!   this runs. Extraction is one `serde_json` pass into a small struct that
//!   borrows nothing and ignores every field it does not need -- tens of
//!   microseconds against a multi-second generation, i.e. well under 0.01%.
//! * Streaming observation assembles one bounded SSE event at a time across
//!   arbitrary transport chunks. SIMD searches skip JSON parsing for events
//!   without usage or finish metadata. Oversized events are ignored by the
//!   observer; their response bytes still pass through unchanged.
//! * Recording itself is the `metrics` crate's atomic counters/histograms with
//!   interned label strings, the same mechanism the existing router metrics
//!   use. SSE buffers reuse their allocations across events; no extra channel.
//! * The Prometheus exporter already listens on its own socket, so scraping
//!   never contends with inference traffic.
//!
//! Nothing here can fail a request: every extraction is best-effort and a parse
//! failure records nothing rather than propagating.

use std::time::{Duration, Instant};

use bytes::Bytes;
use metrics::{counter, describe_counter, describe_histogram, histogram};
use serde::Deserialize;

use super::metrics::intern_string;

// =============================================================================
// WIRE SHAPES
// =============================================================================
//
// Deliberately minimal. serde walks the whole document either way, but
// ignoring a field costs a skip rather than an allocation, so this stays far
// cheaper than deserializing the full ChatCompletionResponse.

#[derive(Deserialize, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Deserialize, Default)]
struct ToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Default)]
struct ToolCall {
    #[serde(default)]
    function: Option<ToolCallFunction>,
}

#[derive(Deserialize, Default)]
struct Message {
    #[serde(default)]
    content: Option<String>,
    /// M3 splits chain-of-thought out of `content` when a reasoning parser is
    /// configured. "Reasoning length distribution and reasoning triggering" is
    /// a named bypass dimension and lives only here.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Default)]
struct Choice {
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    message: Option<Message>,
}

#[derive(Deserialize, Default)]
struct ChatResponse {
    #[serde(default)]
    choices: Option<Vec<Choice>>,
    #[serde(default)]
    usage: Option<Usage>,
}

// =============================================================================
// METRIC NAMES
// =============================================================================

const M_INPUT_TOKENS: &str = "smg_inference_input_tokens";
const M_OUTPUT_TOKENS: &str = "smg_inference_output_tokens";
const M_REASONING_TOKENS: &str = "smg_inference_reasoning_tokens";
const M_REASONING_CHARS: &str = "smg_inference_reasoning_chars";
const M_CACHED_TOKENS: &str = "smg_inference_cached_tokens";
const M_CACHE_HIT_RATIO: &str = "smg_inference_cache_hit_ratio";
const M_FINISH_REASON: &str = "smg_inference_finish_reason_total";
const M_REASONING_TRIGGERED: &str = "smg_inference_reasoning_triggered_total";
const M_TOOL_CALLS_TOTAL: &str = "smg_inference_tool_calls_total";
const M_TOOL_CALLS_PER_RESPONSE: &str = "smg_inference_tool_calls_per_response";
const M_TOOL_CALLS_INVALID: &str = "smg_inference_tool_calls_invalid_total";
const M_TTFT: &str = "smg_inference_ttft_seconds";
const M_TPOT: &str = "smg_inference_tpot_seconds";
const M_STREAM_CHUNK_BYTES: &str = "smg_inference_stream_chunk_bytes";
const M_STREAM_CHUNKS: &str = "smg_inference_stream_chunks_per_response";
const M_OBSERVED: &str = "smg_inference_observed_total";

/// Register descriptions. Call once at startup, alongside the existing
/// `observability::metrics` describe pass.
pub fn describe_inference_metrics() {
    describe_histogram!(M_INPUT_TOKENS, "Prompt tokens per request, by model");
    describe_histogram!(M_OUTPUT_TOKENS, "Completion tokens per response, by model");
    describe_histogram!(
        M_REASONING_TOKENS,
        "Reasoning tokens per response when the backend reports them"
    );
    describe_histogram!(
        M_REASONING_CHARS,
        "reasoning_content length in characters; fallback signal when the backend reports no reasoning_tokens"
    );
    describe_histogram!(M_CACHED_TOKENS, "Cached prompt tokens per request");
    describe_histogram!(
        M_CACHE_HIT_RATIO,
        "cached_tokens / prompt_tokens per request, 0.0-1.0"
    );
    describe_counter!(
        M_FINISH_REASON,
        "Responses by finish_reason; non-stop ratio is reason=length over the total"
    );
    describe_counter!(
        M_REASONING_TRIGGERED,
        "Responses carrying non-empty reasoning_content, by triggered=true|false"
    );
    describe_counter!(M_TOOL_CALLS_TOTAL, "Total tool calls emitted");
    describe_histogram!(
        M_TOOL_CALLS_PER_RESPONSE,
        "Tool calls per response; >1 indicates parallel calls"
    );
    describe_counter!(
        M_TOOL_CALLS_INVALID,
        "Tool calls with a missing name or unparseable JSON arguments"
    );
    describe_histogram!(
        M_TTFT,
        "Time to first token, seconds (streaming): upstream dispatch to first chunk, so prefill is included"
    );
    describe_histogram!(
        M_TPOT,
        "Time per output token, seconds (streaming, requires usage in the final chunk)"
    );
    describe_histogram!(M_STREAM_CHUNK_BYTES, "SSE chunk size in bytes");
    describe_histogram!(M_STREAM_CHUNKS, "SSE chunks per streamed response");
    describe_counter!(
        M_OBSERVED,
        "Responses this module observed, by stream and by whether usage was found"
    );
}

// =============================================================================
// SHARED RECORDING
// =============================================================================

fn record_usage(model: &std::sync::Arc<str>, usage: &Usage) -> Option<u64> {
    if let Some(pt) = usage.prompt_tokens {
        histogram!(M_INPUT_TOKENS, "model" => model.clone()).record(pt as f64);

        if let Some(cached) = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
        {
            histogram!(M_CACHED_TOKENS, "model" => model.clone()).record(cached as f64);
            if pt > 0 {
                // Per-request ratio, not a global one. A global hit rate can be
                // recovered by summing the two token histograms; this keeps the
                // per-request distribution, which is what a bypass comparison
                // against MiniMax's own traffic needs.
                histogram!(M_CACHE_HIT_RATIO, "model" => model.clone())
                    .record((cached as f64 / pt as f64).clamp(0.0, 1.0));
            }
        }
    }

    let completion = usage.completion_tokens;
    if let Some(ct) = completion {
        histogram!(M_OUTPUT_TOKENS, "model" => model.clone()).record(ct as f64);
    }
    if let Some(rt) = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
    {
        histogram!(M_REASONING_TOKENS, "model" => model.clone()).record(rt as f64);
    }
    completion
}

fn record_choice(model: &std::sync::Arc<str>, choice: &Choice) {
    let reason = choice.finish_reason.as_deref().unwrap_or("none");
    // finish_reason is a small closed set, so interning keeps cardinality bounded.
    counter!(M_FINISH_REASON, "model" => model.clone(), "reason" => intern_string(reason))
        .increment(1);

    let Some(msg) = choice.message.as_ref() else {
        return;
    };

    let reasoning_len = msg.reasoning_content.as_ref().map_or(0, |s| s.len());
    counter!(
        M_REASONING_TRIGGERED,
        "model" => model.clone(),
        "triggered" => if reasoning_len > 0 { "true" } else { "false" }
    )
    .increment(1);
    if reasoning_len > 0 {
        histogram!(M_REASONING_CHARS, "model" => model.clone()).record(reasoning_len as f64);
    }

    if let Some(calls) = msg.tool_calls.as_ref() {
        histogram!(M_TOOL_CALLS_PER_RESPONSE, "model" => model.clone()).record(calls.len() as f64);
        counter!(M_TOOL_CALLS_TOTAL, "model" => model.clone()).increment(calls.len() as u64);

        for call in calls {
            let f = call.function.as_ref();
            let bad_name = f.and_then(|f| f.name.as_deref()).unwrap_or("").is_empty();
            // The manual asks for "invalid parameters" to be observed. Arguments
            // arrive as a JSON string; unparseable arguments are the signal.
            let bad_args = f
                .and_then(|f| f.arguments.as_deref())
                .map(|a| {
                    !a.trim().is_empty() && serde_json::from_str::<serde_json::Value>(a).is_err()
                })
                .unwrap_or(false);
            if bad_name || bad_args {
                counter!(
                    M_TOOL_CALLS_INVALID,
                    "model" => model.clone(),
                    "kind" => if bad_name { "missing_name" } else { "bad_arguments" }
                )
                .increment(1);
            }
        }
    } else if msg.content.is_some() {
        histogram!(M_TOOL_CALLS_PER_RESPONSE, "model" => model.clone()).record(0.0);
    }
}

// =============================================================================
// NON-STREAMING
// =============================================================================

/// Observe a complete non-streaming response body.
///
/// `body` is the buffer the router is about to forward; it is not copied. A
/// malformed or non-JSON body records nothing.
pub fn observe_non_streaming(model_id: &str, status_ok: bool, body: &Bytes) {
    if !status_ok || body.is_empty() {
        return;
    }
    let Ok(parsed) = serde_json::from_slice::<ChatResponse>(body) else {
        counter!(M_OBSERVED, "stream" => "false", "usage" => "parse_failed").increment(1);
        return;
    };
    let model = intern_string(model_id);

    if let Some(usage) = parsed.usage.as_ref() {
        record_usage(&model, usage);
        counter!(M_OBSERVED, "stream" => "false", "usage" => "present").increment(1);
    } else {
        counter!(M_OBSERVED, "stream" => "false", "usage" => "absent").increment(1);
    }
    for choice in parsed.choices.iter().flatten() {
        record_choice(&model, choice);
    }
}

// =============================================================================
// STREAMING
// =============================================================================

/// Incremental observer for an SSE response.
///
/// Keeps at most one bounded event; response chunks pass through unchanged.
pub struct StreamObserver {
    model: std::sync::Arc<str>,
    started: Instant,
    first_byte: Option<Instant>,
    chunks: u64,
    completion_tokens: Option<u64>,
    saw_usage: bool,
    finished: bool,
    line: Vec<u8>,
    data: Vec<u8>,
    line_overflow: bool,
    discard_event: bool,
    after_cr: bool,
}

// Metadata observation must not retain an unbounded upstream event.
const MAX_OBSERVED_SSE_BYTES: usize = 1024 * 1024;

/// SIMD substring searchers, built once.
static USAGE_FINDER: once_cell::sync::Lazy<memchr::memmem::Finder<'static>> =
    once_cell::sync::Lazy::new(|| memchr::memmem::Finder::new(b"\"usage\""));
static FINISH_FINDER: once_cell::sync::Lazy<memchr::memmem::Finder<'static>> =
    once_cell::sync::Lazy::new(|| memchr::memmem::Finder::new(b"\"finish_reason\""));

impl StreamObserver {
    /// `started` must be the pre-dispatch instant, not the instant this is
    /// built: the observer is constructed after upstream response headers
    /// arrive, which the engine does not flush until it has a first chunk, so
    /// anything measured from here excludes prefill -- the whole of TTFT.
    pub fn new(model_id: &str, started: Instant) -> Self {
        Self {
            model: intern_string(model_id),
            started,
            first_byte: None,
            chunks: 0,
            completion_tokens: None,
            saw_usage: false,
            finished: false,
            line: Vec::new(),
            data: Vec::new(),
            line_overflow: false,
            discard_event: false,
            after_cr: false,
        }
    }

    /// Feed bytes without assuming transport chunks end at SSE boundaries.
    pub fn on_chunk(&mut self, chunk: &Bytes) {
        if chunk.is_empty() {
            return;
        }
        if self.first_byte.is_none() {
            let now = Instant::now();
            self.first_byte = Some(now);
            histogram!(M_TTFT, "model" => self.model.clone())
                .record(now.duration_since(self.started).as_secs_f64());
        }
        self.chunks += 1;
        histogram!(M_STREAM_CHUNK_BYTES, "model" => self.model.clone()).record(chunk.len() as f64);

        let mut bytes = chunk.as_ref();
        while !bytes.is_empty() {
            if self.after_cr {
                self.after_cr = false;
                if bytes[0] == b'\n' {
                    bytes = &bytes[1..];
                    continue;
                }
            }
            let end = memchr::memchr2(b'\r', b'\n', bytes).unwrap_or(bytes.len());
            if !self.line_overflow {
                if self.line.len() + end <= MAX_OBSERVED_SSE_BYTES {
                    self.line.extend_from_slice(&bytes[..end]);
                } else {
                    self.line.clear();
                    self.line_overflow = true;
                    self.discard_event = true;
                }
            }
            if end == bytes.len() {
                break;
            }
            if !self.line_overflow {
                let mut line = std::mem::take(&mut self.line);
                self.on_line(&line);
                line.clear();
                self.line = line;
            }
            self.line_overflow = false;
            self.after_cr = bytes[end] == b'\r';
            bytes = &bytes[end + 1..];
        }
    }

    fn on_line(&mut self, line: &[u8]) {
        if line.is_empty() {
            if !self.discard_event {
                self.observe_event();
            }
            self.data.clear();
            self.discard_event = false;
            return;
        }
        if self.discard_event {
            return;
        }
        let Some(value) = line.strip_prefix(b"data:") else {
            return;
        };
        let value = value.strip_prefix(b" ").unwrap_or(value);
        if self.data.len() + value.len() + 1 > MAX_OBSERVED_SSE_BYTES {
            self.data.clear();
            self.discard_event = true;
            return;
        }
        self.data.extend_from_slice(value);
        self.data.push(b'\n');
    }

    fn observe_event(&mut self) {
        if USAGE_FINDER.find(&self.data).is_none() && FINISH_FINDER.find(&self.data).is_none() {
            return;
        }
        let Ok(parsed) = serde_json::from_slice::<ChatResponse>(&self.data) else {
            return;
        };
        if let Some(usage) = parsed.usage.as_ref() {
            if !self.saw_usage {
                self.saw_usage = true;
                self.completion_tokens = record_usage(&self.model, usage);
            }
        }
        for choice in parsed.choices.iter().flatten() {
            if choice.finish_reason.is_some() {
                record_choice(&self.model, choice);
            }
        }
    }

    /// Called once when the stream ends, errors, or is dropped by a
    /// disconnecting client.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        histogram!(M_STREAM_CHUNKS, "model" => self.model.clone()).record(self.chunks as f64);
        counter!(
            M_OBSERVED,
            "stream" => "true",
            "usage" => if self.saw_usage { "present" } else { "absent" }
        )
        .increment(1);

        if let (Some(first), Some(ct)) = (self.first_byte, self.completion_tokens) {
            if ct > 1 {
                // Decode time only: total minus the prefill wait, divided by the
                // tokens after the first. This is the quantity the manual's
                // "P50 TPS > 60" is about, so it must not include TTFT.
                let decode: Duration = Instant::now().duration_since(first);
                histogram!(M_TPOT, "model" => self.model.clone())
                    .record(decode.as_secs_f64() / (ct - 1) as f64);
            }
        }
    }
}

impl Drop for StreamObserver {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(s: &str) -> Bytes {
        Bytes::from(s.to_string())
    }

    #[test]
    fn non_streaming_extracts_usage_and_reasoning() {
        // Should not panic and should tolerate the full M3 response shape.
        observe_non_streaming(
            "m3",
            true,
            &body(
                r#"{"choices":[{"finish_reason":"stop","message":{"content":"hi","reasoning_content":"because"}}],
                    "usage":{"prompt_tokens":100,"completion_tokens":20,
                             "prompt_tokens_details":{"cached_tokens":80},
                             "completion_tokens_details":{"reasoning_tokens":7}}}"#,
            ),
        );
    }

    #[test]
    fn malformed_body_is_ignored_not_propagated() {
        observe_non_streaming("m3", true, &body("not json at all"));
        observe_non_streaming("m3", true, &body(""));
        // A non-2xx body must never be mistaken for a generation.
        observe_non_streaming("m3", false, &body(r#"{"error":"boom"}"#));
    }

    #[test]
    fn stream_observer_handles_plain_token_chunks() {
        let mut obs = StreamObserver::new("m3", Instant::now());
        // Ordinary delta chunks carry neither marker and must take the fast path.
        for _ in 0..50 {
            obs.on_chunk(&body(
                "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
            ));
        }
        assert_eq!(obs.chunks, 50);
        assert!(!obs.saw_usage);
        obs.finish();
    }

    #[test]
    fn stream_observer_picks_up_final_usage_chunk() {
        let mut obs = StreamObserver::new("m3", Instant::now());
        obs.on_chunk(&body(
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
        ));
        obs.on_chunk(&body(
            "data: {\"choices\":[{\"finish_reason\":\"length\",\"message\":{\"content\":\"\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
        ));
        assert!(obs.saw_usage);
        assert_eq!(obs.completion_tokens, Some(5));
        obs.finish();
    }

    #[test]
    fn stream_observer_reassembles_events_at_every_byte_boundary() {
        let event = b"data: {\"choices\":[{\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\r\n\r\n";
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, || {
            for split in 1..event.len() {
                let mut observer = StreamObserver::new("fragmented", Instant::now());
                observer.on_chunk(&Bytes::copy_from_slice(&event[..split]));
                observer.on_chunk(&Bytes::copy_from_slice(&event[split..]));
                assert_eq!(observer.completion_tokens, Some(5), "split at {split}");
                observer.finish();
            }
        });
        let rendered = handle.render();
        let finish = rendered
            .lines()
            .find(|line| line.starts_with("smg_inference_finish_reason_total{"))
            .expect("fragmented finish reason was lost");
        assert!(
            finish.ends_with(&format!(" {}", event.len() - 1)),
            "{finish}"
        );
    }

    #[test]
    fn stream_observer_accepts_multiline_data_and_cr_line_endings() {
        let mut observer = StreamObserver::new("multiline", Instant::now());
        let event = b": heartbeat\rdata:{\"usage\":\rdata: {\"completion_tokens\":5}}\r\r";
        for byte in event {
            observer.on_chunk(&Bytes::copy_from_slice(&[*byte]));
        }
        assert_eq!(observer.completion_tokens, Some(5));
    }

    #[test]
    fn stream_observer_bounds_incomplete_event_and_recovers() {
        let mut observer = StreamObserver::new("bounded", Instant::now());
        observer.on_chunk(&Bytes::from(vec![b'x'; MAX_OBSERVED_SSE_BYTES + 1]));
        assert!(observer.line.len() <= MAX_OBSERVED_SSE_BYTES);
        assert!(observer.data.len() <= MAX_OBSERVED_SSE_BYTES);
        assert!(observer.discard_event);
        observer.on_chunk(&body("\n\ndata: {\"usage\":{\"completion_tokens\":7}}\n\n"));
        assert_eq!(observer.completion_tokens, Some(7));
    }

    #[test]
    fn stream_observer_does_not_count_incomplete_events() {
        let mut observer = StreamObserver::new("incomplete", Instant::now());
        observer.on_chunk(&body("data: {\"usage\":{\"completion_tokens\":9}}\n"));
        observer.finish();
        assert!(!observer.saw_usage);
    }

    #[test]
    fn enabled_recorder_counts_usage_and_stream_completion_once() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, || {
            describe_inference_metrics();
            let response = body(
                r#"{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}],
                    "usage":{"prompt_tokens":10,"completion_tokens":3,
                    "prompt_tokens_details":{"cached_tokens":8}}}"#,
            );
            observe_non_streaming("test-model", true, &response);
            // HTTP errors must not inflate the successful response totals.
            observe_non_streaming("test-model", false, &response);
            let mut observer = StreamObserver::new("test-model", Instant::now());
            let usage =
                body("data: {\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":5}}\n\n");
            observer.on_chunk(&usage);
            observer.on_chunk(&usage);
            observer.finish();
            observer.finish();
            // Drop also calls finish; it must not add a third observation.
        });
        let rendered = handle.render();
        for (metric, value) in [
            ("smg_inference_input_tokens_sum", "30"),
            ("smg_inference_output_tokens_sum", "8"),
            ("smg_inference_cached_tokens_sum", "8"),
            ("smg_inference_stream_chunks_per_response_count", "1"),
        ] {
            let line = rendered
                .lines()
                .find(|line| line.starts_with(&format!("{metric}{{")))
                .unwrap_or_else(|| panic!("missing metric {metric}: {rendered}"));
            assert!(line.ends_with(&format!(" {value}")), "{line}");
        }
        let observed: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_inference_observed_total{"))
            .collect();
        assert_eq!(observed.len(), 2);
        assert!(observed.iter().all(|line| line.ends_with(" 1")));
    }

    #[test]
    fn finish_is_idempotent() {
        let mut obs = StreamObserver::new("m3", Instant::now());
        obs.on_chunk(&body("data: {}\n\n"));
        obs.finish();
        obs.finish(); // Drop will call it a third time; must not double-count.
    }
}
