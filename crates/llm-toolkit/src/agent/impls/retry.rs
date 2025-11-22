//! Retry wrapper for agents.
//!
//! This module provides `RetryAgent`, a decorator that adds retry functionality
//! to any agent implementation.

use crate::agent::{Agent, AgentError, Payload, retry::retry_execution};
use async_trait::async_trait;

/// A wrapper agent that adds retry logic to any underlying agent.
///
/// This follows the decorator pattern, allowing retry behavior to be
/// added to any agent without modifying its implementation.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::impls::{ClaudeCodeAgent, RetryAgent};
///
/// let base_agent = ClaudeCodeAgent::new();
/// let retry_agent = RetryAgent::new(base_agent, 3); // Max 3 retries
///
/// let result = retry_agent.execute("Generate a summary".into()).await?;
/// ```
pub struct RetryAgent<T: Agent> {
    inner: T,
    max_retries: u32,
}

impl<T: Agent> RetryAgent<T> {
    /// Creates a new retry agent wrapping the given agent.
    ///
    /// # Arguments
    ///
    /// * `inner` - The agent to wrap with retry logic
    /// * `max_retries` - Maximum number of retry attempts (not including the first attempt)
    pub fn new(inner: T, max_retries: u32) -> Self {
        Self { inner, max_retries }
    }

    /// Returns a reference to the inner agent.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns the maximum number of retries configured.
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }
}

#[async_trait]
impl<T: Agent> Agent for RetryAgent<T>
where
    T::Output: Send,
{
    type Output = T::Output;
    type Expertise = T::Expertise;

    fn expertise(&self) -> &Self::Expertise {
        self.inner.expertise()
    }

    fn name(&self) -> String {
        // Return inner agent's name for transparency
        // This allows RetryAgent to be a true decorator that doesn't affect
        // orchestrator's agent lookup by name
        self.inner.name()
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        // Use the unified retry_execution function
        let inner = &self.inner;
        retry_execution(self.max_retries, &payload, move |p| {
            let p = p.clone();
            async move { inner.execute(p).await }
        })
        .await
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        self.inner.is_available().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::error::ParseErrorReason;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Mock agent for testing that fails a configurable number of times.
    struct FailingAgent {
        fail_count: Arc<AtomicU32>,
        total_calls: Arc<AtomicU32>,
    }

    impl FailingAgent {
        fn new(fail_count: u32) -> Self {
            Self {
                fail_count: Arc::new(AtomicU32::new(fail_count)),
                total_calls: Arc::new(AtomicU32::new(0)),
            }
        }

        fn total_calls(&self) -> u32 {
            self.total_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Agent for FailingAgent {
        type Output = String;

        fn expertise(&self) -> &str {
            "Test agent that fails a configurable number of times"
        }

        async fn execute(&self, _payload: Payload) -> Result<String, AgentError> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let remaining = self.fail_count.load(Ordering::SeqCst);

            if remaining > 0 {
                self.fail_count.fetch_sub(1, Ordering::SeqCst);
                Err(AgentError::ParseError {
                    message: "Simulated failure".to_string(),
                    reason: ParseErrorReason::MarkdownExtractionFailed,
                })
            } else {
                Ok("success".to_string())
            }
        }
    }

    #[tokio::test]
    async fn test_retry_agent_success_first_try() {
        let base = FailingAgent::new(0); // Never fail
        let retry_agent = RetryAgent::new(base, 3);

        let result = retry_agent.execute(Payload::text("test")).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(retry_agent.inner().total_calls(), 1);
    }

    #[tokio::test]
    async fn test_retry_agent_success_after_retries() {
        let base = FailingAgent::new(2); // Fail twice, then succeed
        let retry_agent = RetryAgent::new(base, 3);

        let result = retry_agent.execute(Payload::text("test")).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(retry_agent.inner().total_calls(), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_agent_max_retries_exhausted() {
        let base = FailingAgent::new(10); // Always fail
        let retry_agent = RetryAgent::new(base, 2);

        let result = retry_agent.execute(Payload::text("test")).await;

        assert!(result.is_err());
        assert_eq!(retry_agent.inner().total_calls(), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_agent_name() {
        let base = FailingAgent::new(0);
        let retry_agent = RetryAgent::new(base, 3);

        // RetryAgent should be transparent - it returns the inner agent's name
        // This ensures orchestrator can find agents by their original name
        let name = retry_agent.name();
        assert_eq!(name, "FailingAgent");
    }

    #[tokio::test]
    async fn test_retry_agent_expertise_delegation() {
        let base = FailingAgent::new(0);
        let retry_agent = RetryAgent::new(base, 3);

        assert_eq!(
            retry_agent.expertise(),
            "Test agent that fails a configurable number of times"
        );
    }

    /// Mock agent that fails with 429 + retry_after for testing rate limiting
    struct RateLimitedAgent {
        fail_count: Arc<AtomicU32>,
        total_calls: Arc<AtomicU32>,
        retry_after: std::time::Duration,
    }

    impl RateLimitedAgent {
        fn new(fail_count: u32, retry_after: std::time::Duration) -> Self {
            Self {
                fail_count: Arc::new(AtomicU32::new(fail_count)),
                total_calls: Arc::new(AtomicU32::new(0)),
                retry_after,
            }
        }

        fn total_calls(&self) -> u32 {
            self.total_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Agent for RateLimitedAgent {
        type Output = String;

        fn expertise(&self) -> &str {
            "Test agent that simulates rate limiting with retry_after"
        }

        async fn execute(&self, _payload: Payload) -> Result<String, AgentError> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let remaining = self.fail_count.load(Ordering::SeqCst);

            if remaining > 0 {
                self.fail_count.fetch_sub(1, Ordering::SeqCst);
                Err(AgentError::ProcessError {
                    status_code: Some(429),
                    message: "Rate limited".to_string(),
                    is_retryable: true,
                    retry_after: Some(self.retry_after),
                })
            } else {
                Ok("success".to_string())
            }
        }
    }

    /// Mock agent that fails with 429 without retry_after
    struct RateLimited429Agent {
        fail_count: Arc<AtomicU32>,
        total_calls: Arc<AtomicU32>,
    }

    impl RateLimited429Agent {
        fn new(fail_count: u32) -> Self {
            Self {
                fail_count: Arc::new(AtomicU32::new(fail_count)),
                total_calls: Arc::new(AtomicU32::new(0)),
            }
        }

        fn total_calls(&self) -> u32 {
            self.total_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Agent for RateLimited429Agent {
        type Output = String;

        fn expertise(&self) -> &str {
            "Test agent that simulates 429 without retry_after"
        }

        async fn execute(&self, _payload: Payload) -> Result<String, AgentError> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let remaining = self.fail_count.load(Ordering::SeqCst);

            if remaining > 0 {
                self.fail_count.fetch_sub(1, Ordering::SeqCst);
                Err(AgentError::ProcessError {
                    status_code: Some(429),
                    message: "Rate limited".to_string(),
                    is_retryable: true,
                    retry_after: None,
                })
            } else {
                Ok("success".to_string())
            }
        }
    }

    #[tokio::test]
    async fn test_retry_agent_with_429_retry_after() {
        // Agent fails twice with 429 + retry_after (100ms), then succeeds
        let base = RateLimitedAgent::new(2, std::time::Duration::from_millis(100));
        let retry_agent = RetryAgent::new(base, 3);

        let start = std::time::Instant::now();
        let result = retry_agent.execute(Payload::text("test")).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(retry_agent.inner().total_calls(), 3); // 1 initial + 2 retries

        // Should have waited at least some time (with jitter, it's 0~100ms per retry)
        // We can't assert exact timing due to jitter, but it should be reasonable
        assert!(
            elapsed.as_millis() < 1000,
            "Should complete within 1 second with 100ms retry_after"
        );
    }

    #[tokio::test]
    async fn test_retry_agent_with_429_without_retry_after() {
        // Agent fails once with 429 (no retry_after), then succeeds
        let base = RateLimited429Agent::new(1);
        let retry_agent = RetryAgent::new(base, 3);

        let start = std::time::Instant::now();
        let result = retry_agent.execute(Payload::text("test")).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(retry_agent.inner().total_calls(), 2); // 1 initial + 1 retry

        // With exponential backoff (attempt 1 = 2^0 = 1 second) + jitter, should be under 2 seconds
        assert!(
            elapsed.as_secs() < 2,
            "Should complete within 2 seconds with exponential backoff"
        );
    }

    #[tokio::test]
    async fn test_retry_agent_respects_retry_after_duration() {
        // Agent fails once with 1-second retry_after
        let base = RateLimitedAgent::new(1, std::time::Duration::from_secs(1));
        let retry_agent = RetryAgent::new(base, 3);

        let start = std::time::Instant::now();
        let result = retry_agent.execute(Payload::text("test")).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert_eq!(retry_agent.inner().total_calls(), 2); // 1 initial + 1 retry

        // Should wait some portion of the 1 second due to jitter (0-1000ms)
        // Just verify it's reasonable
        assert!(
            elapsed.as_millis() < 1500,
            "Should complete within 1.5 seconds"
        );
    }

    #[tokio::test]
    async fn test_retry_agent_429_exhausts_retries() {
        // Agent always fails with 429 + retry_after
        let base = RateLimitedAgent::new(10, std::time::Duration::from_millis(50));
        let retry_agent = RetryAgent::new(base, 2); // Only 2 retries

        let result = retry_agent.execute(Payload::text("test")).await;

        assert!(result.is_err());
        assert_eq!(retry_agent.inner().total_calls(), 3); // 1 initial + 2 retries

        // Verify it's a 429 error with retry_after
        if let Err(AgentError::ProcessError {
            status_code,
            retry_after,
            ..
        }) = result
        {
            assert_eq!(status_code, Some(429));
            assert!(retry_after.is_some());
        } else {
            panic!("Expected ProcessError with 429");
        }
    }
}
