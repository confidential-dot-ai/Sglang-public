//! Shared streaming helpers used by the HTTP / OpenAI / PD routers.
//!
//! The main type here is [`BreakerTrackedStream`], a `Stream` adapter that
//! records circuit-breaker outcomes based on how the upstream stream
//! terminates:
//!
//! - **Clean end** (`Poll::Ready(None)`) → `record_success`.
//! - **Upstream transport error** (`Poll::Ready(Some(Err(_)))`) → `record_failure`.
//! - **Caller drops the stream while still active** (client disconnect) → no
//!   breaker call. The HTTP response already shipped a 200 status; whether
//!   the worker is healthy is unknown from this signal alone.
//!
//! Callers can override the terminal state in two cases:
//!
//! - [`BreakerTrackedStream::mark_completed`] — for routers that detect
//!   end-of-stream via an in-band sentinel (e.g. PD's `data: [DONE]`)
//!   before the underlying byte stream returns `None`. Note that
//!   `Completed` is *not* absorbing — a subsequent `poll_next` returning
//!   `Err` will still escalate the terminal to `Errored`. Callers should
//!   stop polling after `mark_completed`.
//! - [`BreakerTrackedStream::mark_errored`] — for routers wrapping content
//!   that already represents a worker failure (e.g. a non-2xx response
//!   body or a synthetic SSE error envelope built from a 5xx). `Errored`
//!   is absorbing: once set, it persists regardless of later polls.
//!
//! [`StallGuard`] sits between the upstream byte stream and
//! `BreakerTrackedStream` and converts a mid-stream silence into an `Err`, so
//! a wedged worker terminates the response instead of holding it open and the
//! breaker records the failure through the existing path.

use std::{
    error::Error as StdError,
    fmt::{self, Display},
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use bytes::Bytes;
use futures_util::Stream;
use tokio::time::{sleep, Sleep};
use tracing::{error, warn};

use crate::core::Worker;

/// Error surfaced by [`StallGuard`].
#[derive(Debug)]
pub enum StreamRelayError<E = reqwest::Error> {
    /// The upstream stream itself failed.
    Upstream(E),
    /// No frame arrived within the inter-frame deadline.
    Stalled { idle: Duration },
}

impl<E: Display> Display for StreamRelayError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upstream(e) => write!(f, "{}", e),
            Self::Stalled { idle } => write!(
                f,
                "upstream stream stalled: no data for {:?} (stream_idle_timeout_secs)",
                idle
            ),
        }
    }
}

impl<E: StdError + 'static> StdError for StreamRelayError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Upstream(e) => Some(e),
            Self::Stalled { .. } => None,
        }
    }
}

/// Fails a stream that goes silent mid-flight.
///
/// The deadline arms only after the first frame, so a long prefill is not
/// charged against it; the wait for the first byte is the socket read
/// timeout's job (see [`crate::core::deadlines`]). Every frame resets it.
///
/// After firing, the guard yields one `Err` and then ends. Dropping it drops
/// the upstream stream, which cancels the in-flight request -- so a stalled
/// worker does not keep a connection or a response body alive.
#[must_use = "StallGuard does nothing unless polled"]
pub struct StallGuard<E = reqwest::Error> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send + 'static>>,
    idle: Option<Duration>,
    timer: Option<Pin<Box<Sleep>>>,
    log_url: String,
    seen_first_frame: bool,
    stalled: bool,
}

impl<E> StallGuard<E> {
    /// `idle: None` disables detection and the guard is a pass-through.
    pub fn new<S>(inner: S, idle: Option<Duration>, log_url: String) -> Self
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            idle,
            timer: None,
            log_url,
            seen_first_frame: false,
            stalled: false,
        }
    }
}

impl<E> Stream for StallGuard<E> {
    type Item = Result<Bytes, StreamRelayError<E>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.stalled {
            return Poll::Ready(None);
        }

        let this = &mut *self;
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.seen_first_frame = true;
                this.timer = None;
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.timer = None;
                Poll::Ready(Some(Err(StreamRelayError::Upstream(e))))
            }
            Poll::Ready(None) => {
                this.timer = None;
                Poll::Ready(None)
            }
            Poll::Pending => {
                let Some(idle) = this.idle else {
                    return Poll::Pending;
                };
                if !this.seen_first_frame {
                    return Poll::Pending;
                }
                let timer = this.timer.get_or_insert_with(|| Box::pin(sleep(idle)));
                match timer.as_mut().poll(cx) {
                    Poll::Ready(()) => {
                        this.stalled = true;
                        warn!(
                            "Upstream stream from worker {} stalled: no data for {:?}",
                            this.log_url, idle
                        );
                        Poll::Ready(Some(Err(StreamRelayError::Stalled { idle })))
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

/// Wrap an upstream byte stream in the stall guard configured by the installed
/// deadline policy, then in breaker tracking. The order matters: a stall must
/// reach `BreakerTrackedStream` as an `Err` for the worker to be marked failed.
pub fn guarded_tracked_stream<S, E>(
    inner: S,
    worker: Arc<dyn Worker>,
    log_url: String,
) -> BreakerTrackedStream<StreamRelayError<E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Display + Send + 'static,
{
    let idle = crate::core::deadlines::get().stream_idle();
    BreakerTrackedStream::new(
        StallGuard::new(inner, idle, log_url.clone()),
        worker,
        log_url,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Terminal {
    /// Stream is still in flight, or was dropped before terminating.
    Active,
    /// Stream returned `None` or the caller marked it complete.
    Completed,
    /// Stream yielded an `Err` item.
    Errored,
}

/// Wraps a `Stream<Item = Result<Bytes, E>>` so that the circuit breaker on
/// `worker` is updated exactly once on drop:
/// - completed → `record_success`
/// - errored → `record_failure`
/// - dropped while still active → neither (client disconnected; the worker
///   is innocent from our point of view).
///
/// `E` defaults to `reqwest::Error` to match the common producer
/// (`response.bytes_stream()`), but can be any `Display + Send + 'static`
/// error type — useful for tests that construct streams with simpler
/// error types.
#[must_use = "BreakerTrackedStream must be polled to completion (or pre-marked) \
              and then dropped for the circuit breaker to record an outcome; \
              discarding it immediately records nothing"]
pub struct BreakerTrackedStream<E = reqwest::Error> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send + 'static>>,
    worker: Arc<dyn Worker>,
    log_url: String,
    terminal: Terminal,
}

impl<E> BreakerTrackedStream<E> {
    pub fn new<S>(inner: S, worker: Arc<dyn Worker>, log_url: String) -> Self
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            worker,
            log_url,
            terminal: Terminal::Active,
        }
    }

    /// Mark the stream as cleanly completed. Use this from callers that
    /// detect end-of-stream via an in-band sentinel (e.g. `data: [DONE]`)
    /// before the underlying byte stream returns `None`.
    ///
    /// Has no effect once the wrapper is in any non-Active state. `Completed`
    /// is *not* absorbing — a later `poll_next` returning `Err` will still
    /// escalate the terminal to `Errored`. Callers should stop polling
    /// after calling this.
    pub fn mark_completed(&mut self) {
        if self.terminal == Terminal::Active {
            self.terminal = Terminal::Completed;
        }
    }

    /// Pre-tag the stream as terminally errored. Use this from callers
    /// constructing a wrapper around content that already represents a
    /// failed worker outcome (e.g. a non-2xx response body or a
    /// synthetic error envelope) so Drop records `record_failure` even
    /// if the underlying stream terminates cleanly. `Errored` is
    /// absorbing — once set it stays set regardless of later events.
    pub fn mark_errored(&mut self) {
        self.terminal = Terminal::Errored;
    }
}

impl<E: Display> Stream for BreakerTrackedStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => {
                error!("Upstream stream error from worker {}: {}", self.log_url, e);
                self.terminal = Terminal::Errored;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if self.terminal == Terminal::Active {
                    self.terminal = Terminal::Completed;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<E> Drop for BreakerTrackedStream<E> {
    fn drop(&mut self) {
        match self.terminal {
            Terminal::Completed => self.worker.circuit_breaker().record_success(),
            Terminal::Errored => self.worker.circuit_breaker().record_failure(),
            // Client disconnected before we knew the worker's verdict.
            // Leaving the breaker untouched is the correct default — we
            // got a 200 header and some bytes; nothing said the worker
            // is unhealthy.
            Terminal::Active => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt, sync::Arc, time::Duration};

    use bytes::Bytes;
    use futures_util::{stream, Stream, StreamExt};
    use tokio::time::{sleep, timeout};

    use super::{BreakerTrackedStream, StallGuard, StreamRelayError};
    use crate::core::{BasicWorkerBuilder, Worker};

    /// Lightweight error type for tests — keeps the wrapper generic so we
    /// don't need to fabricate `reqwest::Error` instances.
    #[derive(Debug)]
    struct TestErr(&'static str);

    impl fmt::Display for TestErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new("http://test-worker").build())
    }

    fn breaker_counters(w: &Arc<dyn Worker>) -> (u64, u64) {
        let cb = w.circuit_breaker();
        (cb.total_successes(), cb.total_failures())
    }

    #[tokio::test]
    async fn drop_while_active_records_nothing() {
        let w = worker();
        let inner = stream::pending::<Result<Bytes, TestErr>>();
        let tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 0));
    }

    #[tokio::test]
    async fn clean_stream_records_one_success() {
        let w = worker();
        let inner = stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn stream_error_records_one_failure() {
        let w = worker();
        let inner = stream::iter(vec![Ok(Bytes::from_static(b"a")), Err(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn errored_is_absorbing_across_subsequent_polls() {
        let w = worker();
        let inner = stream::iter(vec![
            Err::<Bytes, _>(TestErr("boom")),
            Ok(Bytes::from_static(b"after")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_then_drop_records_success() {
        let w = worker();
        let inner = stream::pending::<Result<Bytes, TestErr>>();
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn mark_errored_then_clean_end_still_records_failure() {
        let w = worker();
        let inner = stream::iter(vec![Ok::<_, TestErr>(Bytes::from_static(b"a"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_errored();
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_does_not_overwrite_errored() {
        let w = worker();
        let inner = stream::iter(vec![Err::<Bytes, _>(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    // PD's [DONE] handler calls mark_completed before the underlying byte
    // stream finishes; a trailing transport error must still flip the
    // terminal to Errored so the breaker records failure.
    #[tokio::test]
    async fn mark_completed_then_later_err_escalates_to_failure() {
        let w = worker();
        let inner = stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(TestErr("trailing transport error")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        // Caller observes the [DONE] chunk and pre-marks completed...
        assert!(tracked.next().await.is_some());
        tracked.mark_completed();
        // ...but a trailing poll surfaces a transport error.
        assert!(matches!(tracked.next().await, Some(Err(_))));
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    // ================= StallGuard =================

    const IDLE: Duration = Duration::from_secs(5);

    /// Frames separated by `gaps`, then clean end.
    fn framed(gaps: Vec<Duration>) -> impl Stream<Item = Result<Bytes, TestErr>> {
        stream::unfold(gaps.into_iter(), |mut gaps| async move {
            let gap = gaps.next()?;
            sleep(gap).await;
            Some((Ok(Bytes::from_static(b"frame")), gaps))
        })
    }

    /// A long but healthy generation: 200 frames, 1s apart, against a 5s idle
    /// deadline. Total elapsed (200s) far exceeds the deadline, which is the
    /// whole point -- elapsed time must not terminate a stream that is talking.
    #[tokio::test(start_paused = true)]
    async fn a_long_but_active_stream_is_never_cut_off() {
        let guard = StallGuard::new(
            framed(vec![Duration::from_secs(1); 200]),
            Some(IDLE),
            "u".into(),
        );
        futures_util::pin_mut!(guard);
        let mut frames = 0;
        while let Some(item) = guard.next().await {
            assert!(item.is_ok(), "healthy stream must not error");
            frames += 1;
        }
        assert_eq!(frames, 200);
    }

    /// Gaps just under the deadline must not accumulate: the timer resets on
    /// every frame rather than measuring total duration.
    #[tokio::test(start_paused = true)]
    async fn each_frame_resets_the_idle_deadline() {
        let gaps = vec![Duration::from_millis(4_900); 50];
        let guard = StallGuard::new(framed(gaps), Some(IDLE), "u".into());
        futures_util::pin_mut!(guard);
        assert_eq!(
            guard
                .filter(|i| futures_util::future::ready(i.is_ok()))
                .count()
                .await,
            50
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stream_that_goes_silent_mid_flight_is_failed_then_ended() {
        let inner = stream::once(async { Ok::<_, TestErr>(Bytes::from_static(b"frame")) })
            .chain(stream::pending());
        let guard = StallGuard::new(inner, Some(IDLE), "u".into());
        futures_util::pin_mut!(guard);

        assert!(matches!(guard.next().await, Some(Ok(_))));
        assert!(matches!(
            guard.next().await,
            Some(Err(StreamRelayError::Stalled { idle })) if idle == IDLE
        ));
        assert!(guard.next().await.is_none(), "guard ends after a stall");
    }

    /// Silence before the first frame is a slow prefill, not a stall. It is
    /// the socket read timeout's job, so the guard must stay quiet.
    #[tokio::test(start_paused = true)]
    async fn silence_before_the_first_frame_does_not_trip_the_guard() {
        let guard = StallGuard::new(
            stream::pending::<Result<Bytes, TestErr>>(),
            Some(IDLE),
            "u".into(),
        );
        futures_util::pin_mut!(guard);
        assert!(
            timeout(Duration::from_secs(3600), guard.next())
                .await
                .is_err(),
            "guard fired during the wait for the first frame"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_disabled_guard_is_a_pass_through() {
        let guard = StallGuard::new(framed(vec![Duration::from_secs(600); 3]), None, "u".into());
        futures_util::pin_mut!(guard);
        assert_eq!(guard.count().await, 3);
    }

    #[tokio::test(start_paused = true)]
    async fn an_upstream_error_keeps_its_identity() {
        let inner = stream::iter(vec![Err::<Bytes, _>(TestErr("boom"))]);
        let guard = StallGuard::new(inner, Some(IDLE), "u".into());
        futures_util::pin_mut!(guard);
        match guard.next().await {
            Some(Err(StreamRelayError::Upstream(e))) => assert_eq!(e.0, "boom"),
            other => panic!(
                "expected an upstream error, got {:?}",
                other.map(|r| r.is_ok())
            ),
        }
    }

    /// A stall must reach the breaker as a worker failure, which is why the
    /// guard sits inside the tracker rather than outside it.
    #[tokio::test(start_paused = true)]
    async fn a_stall_records_a_breaker_failure() {
        let w = worker();
        let inner = stream::once(async { Ok::<_, TestErr>(Bytes::from_static(b"frame")) })
            .chain(stream::pending());
        let mut tracked = BreakerTrackedStream::new(
            StallGuard::new(inner, Some(IDLE), "u".into()),
            Arc::clone(&w),
            "u".into(),
        );
        assert!(matches!(tracked.next().await, Some(Ok(_))));
        assert!(matches!(tracked.next().await, Some(Err(_))));
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    /// A healthy stream must still tick the breaker as a success through the
    /// extra layer.
    #[tokio::test(start_paused = true)]
    async fn a_clean_guarded_stream_records_a_breaker_success() {
        let w = worker();
        let mut tracked = BreakerTrackedStream::new(
            StallGuard::new(
                framed(vec![Duration::from_secs(1); 3]),
                Some(IDLE),
                "u".into(),
            ),
            Arc::clone(&w),
            "u".into(),
        );
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }
}
