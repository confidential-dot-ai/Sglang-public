//! Request deadline policy.
//!
//! The gateway has five clocks on an outbound worker request. Collapsing them
//! into one is how a healthy long generation gets killed.
//!
//! | clock | bounds | enforced by |
//! |---|---|---|
//! | connect | TCP/TLS handshake | `ClientBuilder::connect_timeout` |
//! | read | socket silence, including the wait for response headers | `ClientBuilder::read_timeout` |
//! | total, non-streaming | whole request, headers through body | `ClientBuilder::timeout` |
//! | total, streaming | whole request, headers through last SSE frame | [`DeadlinePolicy::apply_stream`] |
//! | stream idle | gap between SSE frames, after the first | [`crate::routers::streaming_utils::StallGuard`] |
//!
//! A reqwest total deadline runs "from when the request starts connecting until
//! the response body has finished", so on a streaming response it counts
//! generation time. Cancelling there truncates the stream, which at the client
//! is indistinguishable from a worker crash. Streaming is therefore bounded by
//! silence, not by elapsed time.
//!
//! The first-byte and inter-frame clocks are separate because they differ by
//! orders of magnitude on this workload: TTFT is a prefill, measured p50
//! 8.55-50.30s depending on topology and concurrency; ITL is a decode step,
//! measured p50 14.5-39.6ms. A single knob has to be sized for the prefill,
//! which leaves a mid-stream wedge holding the connection for that whole
//! budget. `read_timeout` applies to every read and cannot tell them apart, so
//! it takes the first-byte case; `StallGuard` arms only after the first frame
//! and takes the tighter one.

use std::{sync::OnceLock, time::Duration};

/// Stand-in for "no total deadline on this request".
///
/// `RequestBuilder::timeout` can only set a deadline, never clear the one
/// inherited from `ClientBuilder::timeout`; clearing it needs `Client::execute`
/// and `Request::timeout_mut`, which every streaming call site would have to be
/// rebuilt around. A year is unreachable in practice and stays inside tokio's
/// timer range (~2.2 years), which `Duration::MAX` does not.
const EFFECTIVELY_UNBOUNDED: Duration = Duration::from_secs(365 * 24 * 60 * 60);

/// Resolved deadline policy for outbound worker requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeadlinePolicy {
    total: Duration,
    stream_total: Option<Duration>,
    read: Option<Duration>,
    stream_idle: Option<Duration>,
}

impl DeadlinePolicy {
    /// `0` disables an optional clock. `total_secs` has no disabled state;
    /// config validation rejects 0 for it.
    pub fn from_secs(
        total_secs: u64,
        stream_total_secs: u64,
        read_secs: u64,
        stream_idle_secs: u64,
    ) -> Self {
        Self {
            total: Duration::from_secs(total_secs),
            stream_total: (stream_total_secs > 0).then(|| Duration::from_secs(stream_total_secs)),
            read: (read_secs > 0).then(|| Duration::from_secs(read_secs)),
            stream_idle: (stream_idle_secs > 0).then(|| Duration::from_secs(stream_idle_secs)),
        }
    }

    /// Total deadline for ordinary (non-streaming) requests.
    pub fn total(&self) -> Duration {
        self.total
    }

    /// Socket-silence timeout, covering the wait for response headers.
    pub fn read(&self) -> Option<Duration> {
        self.read
    }

    /// Longest gap tolerated between SSE frames once streaming has started.
    pub fn stream_idle(&self) -> Option<Duration> {
        self.stream_idle
    }

    /// Total deadline for streaming requests, or `None` when unbounded.
    pub fn stream_total(&self) -> Option<Duration> {
        self.stream_total
    }

    /// Replace the client-level total deadline on a request that is about to
    /// stream. Required on every streaming request builder: without it the
    /// request inherits the non-streaming total and is cancelled mid-generation.
    pub fn apply_stream(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.timeout(self.stream_total.unwrap_or(EFFECTIVELY_UNBOUNDED))
    }
}

impl Default for DeadlinePolicy {
    /// Must track the config defaults; this is what a call site gets when it
    /// runs before [`install`].
    fn default() -> Self {
        Self::from_secs(
            crate::config::types::DEFAULT_REQUEST_TIMEOUT_SECS,
            crate::config::types::DEFAULT_STREAM_TOTAL_TIMEOUT_SECS,
            crate::config::types::DEFAULT_REQUEST_READ_TIMEOUT_SECS,
            crate::config::types::DEFAULT_STREAM_IDLE_TIMEOUT_SECS,
        )
    }
}

static POLICY: OnceLock<DeadlinePolicy> = OnceLock::new();

/// Publish the resolved policy for the process. Returns false if one was
/// already installed.
///
/// Installed from HTTP client construction, the only place the client-level
/// clocks are set, so the two cannot drift apart. Streaming call sites read
/// this global rather than their own client, so a second client with a
/// different policy could not be honoured anyway.
pub fn install(policy: DeadlinePolicy) -> bool {
    POLICY.set(policy).is_ok()
}

/// The installed policy, or the shipped defaults if none was installed.
pub fn get() -> DeadlinePolicy {
    *POLICY.get().unwrap_or(&DeadlinePolicy::default())
}

/// Stamp the installed policy's streaming deadline onto a request builder.
pub fn apply_stream_deadline(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    get().apply_stream(builder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_disables_the_optional_clocks() {
        let p = DeadlinePolicy::from_secs(1800, 0, 0, 0);
        assert_eq!(p.total(), Duration::from_secs(1800));
        assert_eq!(p.stream_total(), None);
        assert_eq!(p.read(), None);
        assert_eq!(p.stream_idle(), None);
    }

    #[test]
    fn nonzero_values_are_carried_through_as_seconds() {
        let p = DeadlinePolicy::from_secs(1800, 7200, 600, 120);
        assert_eq!(p.total(), Duration::from_secs(1800));
        assert_eq!(p.stream_total(), Some(Duration::from_secs(7200)));
        assert_eq!(p.read(), Some(Duration::from_secs(600)));
        assert_eq!(p.stream_idle(), Some(Duration::from_secs(120)));
    }

    /// A streaming request must not inherit the non-streaming total deadline.
    #[test]
    fn an_unbounded_stream_does_not_inherit_the_total_deadline() {
        let p = DeadlinePolicy::from_secs(1800, 0, 600, 120);
        let client = reqwest::Client::builder()
            .timeout(p.total())
            .build()
            .expect("client");

        // A per-request timeout is what overrides the client default at
        // execute time; carrying none is how a request inherits it.
        let plain = client.post("http://example.invalid").build().expect("req");
        assert_eq!(
            plain.timeout(),
            None,
            "non-streaming requests must carry no override, so the client total applies"
        );

        let streamed = p
            .apply_stream(client.post("http://example.invalid"))
            .build()
            .expect("req");
        assert_eq!(
            streamed.timeout(),
            Some(&EFFECTIVELY_UNBOUNDED),
            "streaming requests must not carry the 1800s total"
        );
    }

    #[test]
    fn a_bounded_stream_total_overrides_the_client_default() {
        let p = DeadlinePolicy::from_secs(1800, 7200, 600, 120);
        let client = reqwest::Client::builder()
            .timeout(p.total())
            .build()
            .expect("client");
        let streamed = p
            .apply_stream(client.post("http://example.invalid"))
            .build()
            .expect("req");
        assert_eq!(streamed.timeout(), Some(&Duration::from_secs(7200)));
    }

    #[test]
    fn the_unbounded_sentinel_stays_inside_tokios_timer_range() {
        // tokio's timer wheel saturates around 2.2 years.
        assert!(EFFECTIVELY_UNBOUNDED < Duration::from_secs(2 * 365 * 24 * 60 * 60));
        assert!(EFFECTIVELY_UNBOUNDED > Duration::from_secs(30 * 24 * 60 * 60));
    }

    #[test]
    fn get_without_install_returns_the_shipped_defaults() {
        // Deliberately does not install(): another test in this binary may
        // own the OnceLock, and the fallback is what is under test.
        assert_eq!(get().total(), DeadlinePolicy::default().total());
    }
}
