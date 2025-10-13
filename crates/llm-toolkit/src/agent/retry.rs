//! Core retry logic for agent operations.
//!
//! This module provides the unified retry mechanism that can be used by both
//! `RetryAgent` and macro-generated code, following the DRY (Don't Repeat Yourself)
//! principle.

use super::{AgentError, Payload};
use std::future::Future;

/// Executes an operation with retry logic.
///
/// This is the core retry function that implements:
/// - Configurable max retries
/// - Automatic retry delay based on error type (with jitter)
/// - Only retries errors marked as retryable
/// - Unified retry counter across all error types
///
/// # Design Philosophy
///
/// This function follows the DRY principle by centralizing retry logic
/// that was previously duplicated in `RetryAgent` and macro-generated code.
///
/// # Arguments
///
/// * `max_retries` - Maximum number of retry attempts (not including the first attempt)
/// * `payload` - The payload to pass to the operation
/// * `operation` - An async function that performs the operation
///
/// # Returns
///
/// Returns the result of the operation, or the last error if all retries are exhausted.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::retry::retry_execution;
/// use llm_toolkit::agent::{Payload, AgentError};
///
/// async fn my_operation(payload: &Payload) -> Result<String, AgentError> {
///     // Your operation here
///     Ok("success".to_string())
/// }
///
/// let result = retry_execution(3, &payload, my_operation).await;
/// ```
pub async fn retry_execution<F, Fut, T>(
    max_retries: u32,
    payload: &Payload,
    operation: F,
) -> Result<T, AgentError>
where
    F: Fn(&Payload) -> Fut + Send + Sync,
    Fut: Future<Output = Result<T, AgentError>> + Send,
    T: Send,
{
    let mut attempts = 0;

    loop {
        attempts += 1;

        match operation(payload).await {
            Ok(output) => {
                if attempts > 1 {
                    log::info!(
                        "✅ Operation succeeded on attempt {}/{}",
                        attempts,
                        max_retries + 1
                    );
                }
                return Ok(output);
            }
            Err(e) if e.is_retryable() && attempts <= max_retries => {
                let delay = e.retry_delay(attempts);
                log::warn!(
                    "⚠️ Operation failed (attempt {}/{}): {}. Retrying in {:?}...",
                    attempts,
                    max_retries + 1,
                    e,
                    delay
                );
                tokio::time::sleep(delay).await;
                continue;
            }
            Err(e) => {
                if e.is_retryable() {
                    log::error!(
                        "❌ Operation failed after {} attempts (max retries exhausted): {}",
                        attempts,
                        e
                    );
                } else {
                    log::error!("❌ Operation failed with non-retryable error: {}", e);
                }
                return Err(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::error::ParseErrorReason;

    #[tokio::test]
    async fn test_retry_execution_success_first_try() {
        let payload = Payload::text("test");

        let operation = |_payload: &Payload| async { Ok::<_, AgentError>("success".to_string()) };

        let result = retry_execution(3, &payload, operation).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_retry_execution_success_after_retry() {
        let payload = Payload::text("test");
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let operation = move |_payload: &Payload| {
            let count = call_count_clone.clone();
            async move {
                let current = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if current < 2 {
                    Err(AgentError::ParseError {
                        message: "Retryable error".to_string(),
                        reason: ParseErrorReason::MarkdownExtractionFailed,
                    })
                } else {
                    Ok("success".to_string())
                }
            }
        };

        let result = retry_execution(3, &payload, operation).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "Should have retried twice before succeeding"
        );
    }

    #[tokio::test]
    async fn test_retry_execution_non_retryable_error() {
        let payload = Payload::text("test");
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let operation = move |_payload: &Payload| {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err::<String, _>(AgentError::ExecutionFailed(
                    "Non-retryable error".to_string(),
                ))
            }
        };

        let result = retry_execution(3, &payload, operation).await;

        assert!(result.is_err());
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "Should not retry non-retryable errors"
        );
    }

    #[tokio::test]
    async fn test_retry_execution_max_retries_exhausted() {
        let payload = Payload::text("test");
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let operation = move |_payload: &Payload| {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err::<String, _>(AgentError::ParseError {
                    message: "Always fails".to_string(),
                    reason: ParseErrorReason::MarkdownExtractionFailed,
                })
            }
        };

        let result = retry_execution(2, &payload, operation).await;

        assert!(result.is_err());
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "Should try once + 2 retries"
        );
    }
}
