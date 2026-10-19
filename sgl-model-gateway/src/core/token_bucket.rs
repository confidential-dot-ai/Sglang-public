use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::sync::Notify;
use tracing::{debug, trace};

/// Token bucket for rate limiting.
///
/// This implementation provides:
/// - Smooth rate limiting with configurable refill rate
/// - Burst capacity handling
/// - Fair queuing for waiting requests via Notify
/// - Sync token return for Drop handlers (via `return_tokens_sync`)
///
/// Uses `parking_lot::Mutex` for sync-compatible locking (no async required).
#[derive(Clone)]
pub struct TokenBucket {
    inner: Arc<Mutex<TokenBucketInner>>,
    notify: Arc<Notify>,
    refill_rate: f64, // tokens per second
}

struct TokenBucketInner {
    tokens: f64,
    last_refill: Instant,
    /// Live ceiling. Lives inside the mutex (rather than on TokenBucket) so the
    /// admission limit can be retuned at runtime via `set_capacity` without a
    /// restart -- see `POST /admission`.
    capacity: f64,
    /// Permits owed back to a capacity reduction, held by requests still in
    /// flight. While positive, returned permits retire debt instead of becoming
    /// available.
    debt: f64,
}

impl TokenBucket {
    /// Create a new token bucket
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens (burst capacity)
    /// * `refill_rate` - Tokens added per second (0 for pure concurrency limiting)
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        let capacity = capacity as f64;
        // Allow refill_rate=0 for pure concurrency limiting (semaphore behavior)
        // When refill_rate=0, tokens are only returned via return_tokens()
        let refill_rate = refill_rate as f64;

        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                tokens: capacity,
                last_refill: Instant::now(),
                capacity,
                debt: 0.0,
            })),
            notify: Arc::new(Notify::new()),
            refill_rate,
        }
    }

    /// Current ceiling.
    pub fn capacity(&self) -> f64 {
        self.inner.lock().capacity
    }

    /// Retune the ceiling at runtime.
    ///
    /// Raising it credits the difference immediately and wakes waiters, so a
    /// raise takes effect on queued requests without waiting for a refill tick.
    ///
    /// Lowering it never aborts in-flight requests: the reduction comes out of
    /// available permits first, and the remainder becomes debt, so concurrency
    /// drains to the new ceiling as those requests complete.
    pub fn set_capacity(&self, new_capacity: f64) {
        let new_capacity = new_capacity.max(0.0);
        let raised = {
            let mut inner = self.inner.lock();
            let delta = new_capacity - inner.capacity;
            inner.capacity = new_capacity;
            if delta > 0.0 {
                // Cancel outstanding debt before crediting anything back.
                let paid = inner.debt.min(delta);
                inner.debt -= paid;
                inner.tokens += delta - paid;
                true
            } else {
                let reduction = -delta;
                let from_available = inner.tokens.min(reduction);
                inner.tokens -= from_available;
                inner.debt += reduction - from_available;
                false
            }
        };
        if raised {
            self.notify.notify_waiters();
        }
        debug!("Token bucket: capacity set to {}", new_capacity);
    }

    /// Requests currently holding a permit, for metrics. Includes debt: those
    /// permits are still held even though the ceiling no longer accounts for
    /// them. Not refill-adjusted; meaningful for the `refill_rate=0` case.
    pub fn in_flight(&self) -> f64 {
        let inner = self.inner.lock();
        (inner.capacity - inner.tokens).max(0.0) + inner.debt
    }

    /// Try to acquire tokens immediately.
    ///
    /// Returns `Ok(())` if tokens were acquired, `Err(())` if insufficient tokens.
    pub async fn try_acquire(&self, tokens: f64) -> Result<(), ()> {
        self.try_acquire_sync(tokens)
    }

    /// Sync version of try_acquire (for internal use).
    fn try_acquire_sync(&self, tokens: f64) -> Result<(), ()> {
        let mut inner = self.inner.lock();

        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(inner.capacity);
        inner.last_refill = now;

        trace!(
            "Token bucket: {} tokens available, requesting {}",
            inner.tokens,
            tokens
        );

        if inner.tokens >= tokens {
            inner.tokens -= tokens;
            debug!(
                "Token bucket: acquired {} tokens, {} remaining",
                tokens, inner.tokens
            );
            Ok(())
        } else {
            Err(())
        }
    }

    /// Acquire tokens, waiting until they are available.
    ///
    /// Waits indefinitely and imposes no deadline of its own: a request larger
    /// than the ceiling, or one made against a bucket nothing is returning to,
    /// never completes. Bound it with `acquire_timeout()`.
    pub async fn acquire(&self, tokens: f64) -> Result<(), tokio::time::error::Elapsed> {
        debug!("Token bucket: waiting for {} tokens", tokens);

        loop {
            // Register before re-checking: notify_waiters() only wakes waiters
            // already queued, so a return in between is lost.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if self.try_acquire(tokens).await.is_ok() {
                return Ok(());
            }

            if self.refill_rate == 0.0 {
                // Pure concurrency limiting: nothing accrues on its own, so a
                // return is the only thing that can help.
                notified.await;
                continue;
            }

            // Recomputed every pass. A single estimate taken on entry goes
            // stale as soon as another waiter takes the tokens it counted on.
            let until_refilled = {
                let inner = self.inner.lock();
                let deficit = tokens - inner.tokens;
                Duration::from_secs_f64((deficit / self.refill_rate).max(0.0))
            };

            tokio::select! {
                _ = notified => {},
                _ = tokio::time::sleep(until_refilled) => {},
            }
        }
    }

    /// Acquire tokens with custom timeout.
    pub async fn acquire_timeout(
        &self,
        tokens: f64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// Return tokens to the bucket (sync version).
    ///
    /// This is safe to call from sync contexts (e.g., Drop handlers).
    /// Uses `parking_lot::Mutex` which never blocks indefinitely.
    pub fn return_tokens_sync(&self, tokens: f64) {
        {
            let mut inner = self.inner.lock();
            // Retire debt from a lowered ceiling first: those permits are being
            // reclaimed, not handed to the next request.
            let paid = inner.debt.min(tokens);
            inner.debt -= paid;
            inner.tokens = (inner.tokens + (tokens - paid)).min(inner.capacity);
            debug!(
                "Token bucket: returned {} tokens ({} retired debt), {} available",
                tokens, paid, inner.tokens
            );
        } // Release lock before notify
        self.notify.notify_waiters();
    }

    /// Return tokens to the bucket (async version for API compatibility).
    pub async fn return_tokens(&self, tokens: f64) {
        self.return_tokens_sync(tokens);
    }

    /// Get current available tokens (for monitoring).
    pub async fn available_tokens(&self) -> f64 {
        let mut inner = self.inner.lock();

        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(inner.capacity);
        inner.last_refill = now;

        inner.tokens
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::sync::Arc;

    /// A burst of returns must wake every waiter, not just the one that wins
    /// the first signal. The losers re-check and re-wait, and a return landing
    /// in that gap used to be dropped.
    #[tokio::test]
    async fn a_burst_of_returns_is_not_lost_by_requeueing_waiters() {
        let bucket = Arc::new(TokenBucket::new(4, 0));
        for _ in 0..4 {
            assert!(bucket.try_acquire(1.0).await.is_ok());
        }

        let waiters: Vec<_> = (0..4)
            .map(|_| {
                let bucket = bucket.clone();
                tokio::spawn(async move {
                    bucket.acquire_timeout(1.0, Duration::from_secs(2)).await
                })
            })
            .collect();

        // Let every waiter reach the wait before any token comes back.
        tokio::time::sleep(Duration::from_millis(50)).await;
        for _ in 0..4 {
            bucket.return_tokens_sync(1.0);
        }

        for waiter in waiters {
            assert!(
                waiter.await.expect("waiter task ran").is_ok(),
                "a waiter missed its wakeup and timed out with a token available"
            );
        }
    }

    /// A waiter must get the caller's whole budget. The deadline used to be
    /// estimated once on entry, so a waiter that lost the race for a refill
    /// gave up at its own estimate with time left on the clock.
    #[tokio::test]
    async fn a_contended_waiter_uses_the_callers_full_budget() {
        // One token per second, so the second waiter cannot be served until
        // roughly 2s, well inside the 3s each of them is given.
        let bucket = Arc::new(TokenBucket::new(2, 1));
        assert!(bucket.try_acquire(2.0).await.is_ok());

        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let bucket = bucket.clone();
                tokio::spawn(async move {
                    bucket.acquire_timeout(1.0, Duration::from_secs(3)).await
                })
            })
            .collect();

        for waiter in waiters {
            assert!(
                waiter.await.expect("waiter task ran").is_ok(),
                "a waiter gave up on its own estimate instead of the 3s budget"
            );
        }
    }

    /// Lowering the ceiling below the in-flight count must not hand the
    /// reduction back to new requests when a permit is returned.
    #[tokio::test]
    async fn returns_retire_debt_before_they_refill() {
        let bucket = TokenBucket::new(4, 0);
        for _ in 0..4 {
            assert!(bucket.try_acquire(1.0).await.is_ok());
        }

        // Nothing is available to give up, so the whole cut becomes debt.
        bucket.set_capacity(2.0);
        assert_eq!(bucket.in_flight(), 4.0);

        // The first two returns pay it down and admit nobody.
        bucket.return_tokens_sync(1.0);
        bucket.return_tokens_sync(1.0);
        assert_eq!(bucket.available_tokens().await, 0.0);
        assert_eq!(bucket.in_flight(), 2.0);

        // Only now does a return free a slot, and the ceiling holds at 2.
        bucket.return_tokens_sync(1.0);
        assert_eq!(bucket.available_tokens().await, 1.0);
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    /// Raising the ceiling again must not mint permits that are still held.
    #[tokio::test]
    async fn raising_the_ceiling_clears_debt_before_adding_tokens() {
        let bucket = TokenBucket::new(4, 0);
        for _ in 0..4 {
            assert!(bucket.try_acquire(1.0).await.is_ok());
        }
        bucket.set_capacity(2.0);

        // Undoing the cut restores the ceiling, not two free slots: all four
        // permits are still out.
        bucket.set_capacity(4.0);
        assert_eq!(bucket.capacity(), 4.0);
        assert_eq!(bucket.in_flight(), 4.0);
        assert_eq!(bucket.available_tokens().await, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5);

        assert!(bucket.try_acquire(5.0).await.is_ok());
        assert!(bucket.try_acquire(5.0).await.is_ok());

        assert!(bucket.try_acquire(1.0).await.is_err());

        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(bucket.try_acquire(1.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10);

        assert!(bucket.try_acquire(10.0).await.is_ok());

        tokio::time::sleep(Duration::from_millis(500)).await;

        let available = bucket.available_tokens().await;
        assert!((4.0..=6.0).contains(&available));
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_rate() {
        // With refill_rate=0, tokens should only come back via return_tokens()
        let bucket = TokenBucket::new(2, 0);

        // Acquire both tokens
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens available
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Wait - should NOT refill automatically
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Return a token - now we should be able to acquire
        bucket.return_tokens(1.0).await;
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens again
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_with_notify() {
        // Test that acquire wakes up when tokens are returned
        let bucket = Arc::new(TokenBucket::new(1, 0));

        // Acquire the only token
        assert!(bucket.try_acquire(1.0).await.is_ok());

        let bucket_clone = bucket.clone();

        // Spawn a task that will return the token after a delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            bucket_clone.return_tokens(1.0).await;
        });

        // This should wait and then succeed when token is returned
        let result = bucket.acquire_timeout(1.0, Duration::from_secs(1)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_return_tokens_sync() {
        // Test that sync return works correctly
        let bucket = TokenBucket::new(2, 0);

        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Use sync return
        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).await.is_ok());
    }
}
