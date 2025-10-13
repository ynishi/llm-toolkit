//! Error types for the agent module.

use std::time::Duration;
use thiserror::Error;

/// Reason for parse errors, used to determine retry strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorReason {
    /// JSON is structurally invalid (retrying unlikely to help)
    InvalidJson,
    /// Stream was truncated or incomplete (retry may help)
    UnexpectedEof,
    /// JSON is valid but doesn't match expected schema (retrying unlikely to help)
    SchemaMismatch,
    /// Failed to extract JSON from markdown/text (retry may help with clearer prompt)
    MarkdownExtractionFailed,
}

/// Errors that can occur during agent execution.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The agent execution failed with a specific error message.
    #[error("Agent execution failed: {0}")]
    ExecutionFailed(String),

    /// Failed to parse the agent's output into the expected type.
    ///
    /// This variant is structured to provide better retry decisions based on
    /// the specific reason for the parse failure.
    #[error("Failed to parse agent output: {message} (reason: {reason:?})")]
    ParseError {
        message: String,
        reason: ParseErrorReason,
    },

    /// The agent process spawning or communication failed.
    ///
    /// This variant is structured to allow better retry decisions based on
    /// HTTP status codes, explicit retry flags, and server-provided retry delays.
    #[error("Process error: {message}{}{}",
        status_code.map(|c| format!(" (status: {})", c)).unwrap_or_default(),
        retry_after.map(|d| format!(" (retry after: {}s)", d.as_secs())).unwrap_or_default()
    )]
    ProcessError {
        status_code: Option<u16>,
        message: String,
        is_retryable: bool,
        /// Server-provided retry delay (e.g., from Retry-After header)
        /// For 429 errors, this typically ranges from 60 seconds to several minutes
        retry_after: Option<Duration>,
    },

    /// I/O error occurred during agent execution.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Failed to serialize output to JSON value.
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    /// A generic error for other cases.
    #[error("Agent error: {0}")]
    Other(String),
}

impl AgentError {
    /// Creates a new ProcessError with basic information.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let error = AgentError::process_error(500, "Internal server error", true);
    /// ```
    pub fn process_error(status_code: u16, message: impl Into<String>, is_retryable: bool) -> Self {
        AgentError::ProcessError {
            status_code: Some(status_code),
            message: message.into(),
            is_retryable,
            retry_after: None,
        }
    }

    /// Creates a new ProcessError with retry_after duration.
    ///
    /// This is particularly useful for 429 rate limiting errors where the server
    /// specifies how long to wait before retrying.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::time::Duration;
    ///
    /// // Server says to wait 90 seconds before retrying
    /// let error = AgentError::process_error_with_retry_after(
    ///     429,
    ///     "Rate limit exceeded",
    ///     true,
    ///     Duration::from_secs(90)
    /// );
    /// ```
    pub fn process_error_with_retry_after(
        status_code: u16,
        message: impl Into<String>,
        is_retryable: bool,
        retry_after: Duration,
    ) -> Self {
        AgentError::ProcessError {
            status_code: Some(status_code),
            message: message.into(),
            is_retryable,
            retry_after: Some(retry_after),
        }
    }

    /// Check if this error is transient (likely to succeed on retry).
    ///
    /// **Deprecated:** Use `is_retryable()` instead.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            AgentError::ProcessError { .. } | AgentError::IoError(_)
        )
    }

    /// Check if this error should trigger an automatic retry.
    ///
    /// Returns `true` for errors that are likely transient and may succeed on retry:
    /// - `ParseError`: Only certain parse error types (UnexpectedEof, MarkdownExtractionFailed)
    /// - `ProcessError`: Based on status code and is_retryable flag
    /// - `IoError`: Temporary I/O failures
    ///
    /// Returns `false` for errors that are unlikely to be resolved by retry:
    /// - `ExecutionFailed`: LLM logical errors (should be reported to orchestrator)
    /// - `SerializationFailed`: Code-level issues
    /// - `Other`: Unknown errors (better to fail fast)
    ///
    /// # Design Philosophy
    ///
    /// Agent-level retries should be simple and limited (2-3 attempts). If retries
    /// are exhausted, the error should be reported to the orchestrator, which has
    /// broader context and can make better decisions (try different agents, redesign
    /// strategy, escalate to human, etc.).
    pub fn is_retryable(&self) -> bool {
        match self {
            // Process errors: check status code and explicit flag
            AgentError::ProcessError {
                is_retryable,
                status_code,
                ..
            } => *is_retryable || matches!(status_code, Some(429) | Some(503) | Some(500)),
            // Parse errors: only retry if the reason suggests it might help
            AgentError::ParseError { reason, .. } => matches!(
                reason,
                ParseErrorReason::UnexpectedEof | ParseErrorReason::MarkdownExtractionFailed
            ),
            // I/O errors are generally transient
            AgentError::IoError(_) => true,
            // All other errors should not be retried
            _ => false,
        }
    }

    /// Calculate the delay before the next retry attempt.
    ///
    /// This implements different backoff strategies based on the error type and
    /// server-provided retry information:
    ///
    /// # Priority Order
    ///
    /// 1. **Server-provided `retry_after`** (highest priority)
    ///    - Uses the exact duration specified by the server (e.g., Retry-After header)
    ///    - Common for 429 rate limiting: 60s to several minutes
    ///    - Applied with Full Jitter (randomize between 0 and retry_after)
    ///
    /// 2. **Rate limiting (429) fallback**
    ///    - Exponential backoff: min(60s, 2^attempt seconds) with full jitter
    ///    - Caps at 60 seconds to avoid excessive delays
    ///
    /// 3. **Other errors**
    ///    - Linear backoff: (100ms * attempt) with full jitter
    ///
    /// # Full Jitter Strategy
    ///
    /// Randomizes the delay between 0 and the base delay to prevent the
    /// "thundering herd" problem in distributed systems where many clients
    /// retry simultaneously.
    ///
    /// # Arguments
    ///
    /// * `attempt` - The current retry attempt number (1-indexed)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::time::Duration;
    ///
    /// // Example 1: Server-provided retry_after (highest priority)
    /// let error = AgentError::ProcessError {
    ///     status_code: Some(429),
    ///     message: "Rate limited".to_string(),
    ///     is_retryable: true,
    ///     retry_after: Some(Duration::from_secs(90)),
    /// };
    /// let delay = error.retry_delay(1); // Random delay between 0-90 seconds
    ///
    /// // Example 2: 429 without retry_after (fallback to exponential)
    /// let error = AgentError::ProcessError {
    ///     status_code: Some(429),
    ///     message: "Rate limited".to_string(),
    ///     is_retryable: true,
    ///     retry_after: None,
    /// };
    /// let delay = error.retry_delay(1); // Random delay between 0-2 seconds
    /// let delay = error.retry_delay(6); // Random delay between 0-60 seconds (capped)
    /// ```
    pub fn retry_delay(&self, attempt: u32) -> Duration {
        let base_delay = match self {
            // Priority 1: Server-provided retry_after (e.g., from Retry-After header)
            AgentError::ProcessError {
                retry_after: Some(duration),
                ..
            } => *duration,

            // Priority 2: Rate limiting (429) fallback - exponential backoff capped at 60s
            // LLM APIs typically require 60s+ waits for rate limiting
            AgentError::ProcessError {
                status_code: Some(429),
                ..
            } => {
                let exponential_delay = 2_u64.pow(attempt.saturating_sub(1));
                Duration::from_secs(exponential_delay.min(60))
            }

            // Priority 3: Other errors - linear backoff (100ms * attempt)
            _ => Duration::from_millis(100 * attempt as u64),
        };

        // Full Jitter: random delay between 0 and base_delay
        // This prevents thundering herd problem in distributed systems
        use rand::Rng;
        let jitter_ms = rand::thread_rng().gen_range(0..=base_delay.as_millis() as u64);
        Duration::from_millis(jitter_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_retryable_parse_error_unexpected_eof() {
        let err = AgentError::ParseError {
            message: "Stream ended prematurely".to_string(),
            reason: ParseErrorReason::UnexpectedEof,
        };
        assert!(
            err.is_retryable(),
            "ParseError with UnexpectedEof should be retryable"
        );
    }

    #[test]
    fn test_is_retryable_parse_error_markdown_extraction() {
        let err = AgentError::ParseError {
            message: "Could not extract JSON from markdown".to_string(),
            reason: ParseErrorReason::MarkdownExtractionFailed,
        };
        assert!(
            err.is_retryable(),
            "ParseError with MarkdownExtractionFailed should be retryable"
        );
    }

    #[test]
    fn test_is_not_retryable_parse_error_invalid_json() {
        let err = AgentError::ParseError {
            message: "Invalid JSON syntax".to_string(),
            reason: ParseErrorReason::InvalidJson,
        };
        assert!(
            !err.is_retryable(),
            "ParseError with InvalidJson should NOT be retryable"
        );
    }

    #[test]
    fn test_is_not_retryable_parse_error_schema_mismatch() {
        let err = AgentError::ParseError {
            message: "JSON doesn't match schema".to_string(),
            reason: ParseErrorReason::SchemaMismatch,
        };
        assert!(
            !err.is_retryable(),
            "ParseError with SchemaMismatch should NOT be retryable"
        );
    }

    #[test]
    fn test_is_retryable_process_error_with_flag() {
        let err = AgentError::ProcessError {
            status_code: None,
            message: "process crashed".to_string(),
            is_retryable: true,
            retry_after: None,
        };
        assert!(
            err.is_retryable(),
            "ProcessError with is_retryable=true should be retryable"
        );
    }

    #[test]
    fn test_is_retryable_process_error_429() {
        let err = AgentError::ProcessError {
            status_code: Some(429),
            message: "Rate limited".to_string(),
            is_retryable: false, // Even if flag is false, 429 should be retryable
            retry_after: None,
        };
        assert!(
            err.is_retryable(),
            "ProcessError with status 429 should be retryable"
        );
    }

    #[test]
    fn test_is_retryable_process_error_503() {
        let err = AgentError::ProcessError {
            status_code: Some(503),
            message: "Service unavailable".to_string(),
            is_retryable: false,
            retry_after: None,
        };
        assert!(
            err.is_retryable(),
            "ProcessError with status 503 should be retryable"
        );
    }

    #[test]
    fn test_is_not_retryable_process_error() {
        let err = AgentError::ProcessError {
            status_code: Some(400),
            message: "Bad request".to_string(),
            is_retryable: false,
            retry_after: None,
        };
        assert!(
            !err.is_retryable(),
            "ProcessError with status 400 and is_retryable=false should NOT be retryable"
        );
    }

    #[test]
    fn test_is_retryable_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused");
        let err = AgentError::IoError(io_err);
        assert!(
            err.is_retryable(),
            "IoError should be retryable (temporary I/O failures)"
        );
    }

    #[test]
    fn test_is_not_retryable_execution_failed() {
        let err = AgentError::ExecutionFailed("LLM logical error".to_string());
        assert!(
            !err.is_retryable(),
            "ExecutionFailed should NOT be retryable (report to orchestrator)"
        );
    }

    #[test]
    fn test_is_not_retryable_serialization_failed() {
        let err = AgentError::SerializationFailed("invalid type".to_string());
        assert!(
            !err.is_retryable(),
            "SerializationFailed should NOT be retryable (code-level issue)"
        );
    }

    #[test]
    fn test_is_not_retryable_other() {
        let err = AgentError::Other("unknown error".to_string());
        assert!(
            !err.is_retryable(),
            "Other should NOT be retryable (unknown error, fail fast)"
        );
    }

    #[test]
    fn test_is_transient_backward_compatibility() {
        // is_transient() should still work for backward compatibility
        let process_err = AgentError::ProcessError {
            status_code: None,
            message: "process issue".to_string(),
            is_retryable: true,
            retry_after: None,
        };
        assert!(
            process_err.is_transient(),
            "ProcessError should be transient (backward compatibility)"
        );

        let io_err =
            AgentError::IoError(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout"));
        assert!(
            io_err.is_transient(),
            "IoError should be transient (backward compatibility)"
        );

        // But ParseError should NOT be transient (only retryable)
        let parse_err = AgentError::ParseError {
            message: "malformed".to_string(),
            reason: ParseErrorReason::InvalidJson,
        };
        assert!(
            !parse_err.is_transient(),
            "ParseError is not transient (only certain types are retryable)"
        );
    }

    #[test]
    fn test_retry_delay_with_retry_after() {
        // Priority 1: Server-provided retry_after should be used
        let err = AgentError::ProcessError {
            status_code: Some(429),
            message: "Rate limited".to_string(),
            is_retryable: true,
            retry_after: Some(Duration::from_secs(90)),
        };

        // retry_after takes priority over exponential backoff
        for attempt in 1..=5 {
            let delay = err.retry_delay(attempt);
            assert!(
                delay.as_secs() <= 90,
                "Delay should be <= 90 seconds (retry_after value)"
            );
        }
    }

    #[test]
    fn test_retry_delay_rate_limiting_fallback() {
        // Priority 2: 429 without retry_after uses exponential backoff (capped at 60s)
        let err = AgentError::ProcessError {
            status_code: Some(429),
            message: "Rate limited".to_string(),
            is_retryable: true,
            retry_after: None,
        };

        // Test exponential backoff with jitter, capped at 60s
        let delay1 = err.retry_delay(1);
        assert!(delay1.as_secs() <= 1, "Attempt 1: delay <= 1s");

        let delay2 = err.retry_delay(2);
        assert!(delay2.as_secs() <= 2, "Attempt 2: delay <= 2s");

        let delay6 = err.retry_delay(6);
        assert!(delay6.as_secs() <= 60, "Attempt 6: delay <= 60s (capped)");

        let delay10 = err.retry_delay(10);
        assert!(delay10.as_secs() <= 60, "Attempt 10: delay <= 60s (capped)");
    }

    #[test]
    fn test_retry_delay_linear_backoff() {
        // Priority 3: Other errors use linear backoff
        let err = AgentError::ProcessError {
            status_code: None,
            message: "Generic error".to_string(),
            is_retryable: true,
            retry_after: None,
        };

        // Test linear backoff with jitter
        for attempt in 1..=5 {
            let delay = err.retry_delay(attempt);
            let max_delay_ms = 100 * attempt as u64;
            assert!(
                delay.as_millis() <= max_delay_ms as u128,
                "Delay for attempt {} should be <= {} ms (got: {} ms)",
                attempt,
                max_delay_ms,
                delay.as_millis()
            );
        }
    }

    #[test]
    fn test_process_error_builder() {
        // Test basic builder
        let err = AgentError::process_error(500, "Internal server error", true);
        assert!(matches!(
            err,
            AgentError::ProcessError {
                status_code: Some(500),
                is_retryable: true,
                retry_after: None,
                ..
            }
        ));

        // Test builder with retry_after
        let err = AgentError::process_error_with_retry_after(
            429,
            "Rate limit exceeded",
            true,
            Duration::from_secs(90),
        );
        assert!(matches!(
            err,
            AgentError::ProcessError {
                status_code: Some(429),
                is_retryable: true,
                retry_after: Some(_),
                ..
            }
        ));

        if let AgentError::ProcessError { retry_after, .. } = err {
            assert_eq!(retry_after, Some(Duration::from_secs(90)));
        }
    }
}
