//! Error types for the agent module.

use std::time::Duration;
use thiserror::Error;

/// Rich contextual information for error debugging and observability.
///
/// This struct captures additional metadata about where and when an error occurred,
/// enabling better debugging, tracing, and error analysis. It integrates seamlessly
/// with the tracing infrastructure for observability.
#[derive(Debug, Clone)]
pub struct ErrorMetadata {
    /// Agent that encountered the error (if applicable)
    pub agent_name: Option<String>,

    /// Agent expertise/role
    pub agent_expertise: Option<String>,

    /// Operation or phase where error occurred
    pub operation: Option<String>,

    /// Timestamp when error occurred
    pub timestamp: Option<std::time::SystemTime>,

    /// Span/trace ID for correlation with tracing logs
    pub span_id: Option<String>,

    /// Additional structured context (e.g., input size, model params)
    pub context: Option<serde_json::Value>,

    /// Chain of causation (for nested errors) - stored as string for simplicity
    /// Use `caused_by_description` to store the Display representation of the cause
    pub caused_by_description: Option<String>,
}

impl ErrorMetadata {
    /// Creates a new ErrorMetadata with timestamp initialized to now.
    pub fn new() -> Self {
        Self {
            agent_name: None,
            agent_expertise: None,
            operation: None,
            timestamp: Some(std::time::SystemTime::now()),
            span_id: None,
            context: None,
            caused_by_description: None,
        }
    }

    /// Sets the agent name.
    pub fn with_agent(mut self, name: impl Into<String>) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    /// Sets the agent expertise/role.
    pub fn with_expertise(mut self, expertise: impl Into<String>) -> Self {
        self.agent_expertise = Some(expertise.into());
        self
    }

    /// Sets the operation where the error occurred.
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    /// Sets the span/trace ID for correlation.
    pub fn with_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    /// Adds a key-value pair to the structured context.
    pub fn with_context(mut self, key: &str, value: serde_json::Value) -> Self {
        let mut ctx = self.context.unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = ctx.as_object_mut() {
            obj.insert(key.to_string(), value);
        }
        self.context = Some(ctx);
        self
    }

    /// Sets the cause of this error (for error chaining).
    ///
    /// The error is converted to its string representation for storage.
    pub fn with_cause(mut self, cause: &AgentError) -> Self {
        self.caused_by_description = Some(format!("{}", cause));
        self
    }
}

impl Default for ErrorMetadata {
    fn default() -> Self {
        Self::new()
    }
}

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

    // ========== Rich Variants (with ErrorMetadata) ==========
    /// Rich execution error with full contextual metadata.
    ///
    /// This variant provides enhanced debugging and observability by including
    /// agent information, timestamps, operation context, and error chaining.
    #[error("Agent execution failed: {message} [agent: {}, operation: {}]",
        metadata.agent_name.as_deref().unwrap_or("unknown"),
        metadata.operation.as_deref().unwrap_or("unknown")
    )]
    ExecutionFailedRich {
        message: String,
        metadata: ErrorMetadata,
    },

    /// Rich parse error with full contextual metadata.
    #[error("Failed to parse agent output: {message} (reason: {reason:?}) [agent: {}, operation: {}]",
        metadata.agent_name.as_deref().unwrap_or("unknown"),
        metadata.operation.as_deref().unwrap_or("unknown")
    )]
    ParseErrorRich {
        message: String,
        reason: ParseErrorReason,
        metadata: ErrorMetadata,
    },

    /// Rich process error with full contextual metadata.
    #[error("Process error: {message}{}{} [agent: {}, operation: {}]",
        status_code.map(|c| format!(" (status: {})", c)).unwrap_or_default(),
        retry_after.map(|d| format!(" (retry after: {}s)", d.as_secs())).unwrap_or_default(),
        metadata.agent_name.as_deref().unwrap_or("unknown"),
        metadata.operation.as_deref().unwrap_or("unknown")
    )]
    ProcessErrorRich {
        status_code: Option<u16>,
        message: String,
        is_retryable: bool,
        retry_after: Option<Duration>,
        metadata: ErrorMetadata,
    },
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
            // Rich variants: same logic as simple variants
            AgentError::ProcessErrorRich {
                is_retryable,
                status_code,
                ..
            } => *is_retryable || matches!(status_code, Some(429) | Some(503) | Some(500)),
            AgentError::ParseErrorRich { reason, .. } => matches!(
                reason,
                ParseErrorReason::UnexpectedEof | ParseErrorReason::MarkdownExtractionFailed
            ),
            AgentError::ExecutionFailedRich { .. } => false,
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
            AgentError::ProcessErrorRich {
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
            AgentError::ProcessErrorRich {
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

    /// Creates a rich execution error with a builder for adding metadata.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::AgentError;
    ///
    /// let error = AgentError::execution_failed_rich("Task failed")
    ///     .agent("MyAgent")
    ///     .expertise("General-purpose AI")
    ///     .operation("execute")
    ///     .build();
    /// ```
    pub fn execution_failed_rich(message: impl Into<String>) -> RichErrorBuilder {
        RichErrorBuilder::new(message.into())
    }

    /// Extracts metadata from this error, if it's a Rich variant.
    ///
    /// Returns `Some(&ErrorMetadata)` for Rich variants, `None` for simple variants.
    pub fn metadata(&self) -> Option<&ErrorMetadata> {
        match self {
            AgentError::ExecutionFailedRich { metadata, .. } => Some(metadata),
            AgentError::ParseErrorRich { metadata, .. } => Some(metadata),
            AgentError::ProcessErrorRich { metadata, .. } => Some(metadata),
            _ => None,
        }
    }

    /// Converts a simple error variant to a Rich variant with metadata.
    ///
    /// For simple variants, this creates the corresponding Rich variant.
    /// For Rich variants, this replaces the existing metadata.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::{AgentError, ErrorMetadata};
    ///
    /// let simple_err = AgentError::ExecutionFailed("failed".to_string());
    /// let rich_err = simple_err.with_metadata(
    ///     ErrorMetadata::new()
    ///         .with_agent("MyAgent")
    ///         .with_operation("execute")
    /// );
    /// ```
    pub fn with_metadata(self, metadata: ErrorMetadata) -> Self {
        match self {
            AgentError::ExecutionFailed(msg) => AgentError::ExecutionFailedRich {
                message: msg,
                metadata,
            },
            AgentError::ParseError { message, reason } => AgentError::ParseErrorRich {
                message,
                reason,
                metadata,
            },
            AgentError::ProcessError {
                status_code,
                message,
                is_retryable,
                retry_after,
            } => AgentError::ProcessErrorRich {
                status_code,
                message,
                is_retryable,
                retry_after,
                metadata,
            },
            // Already rich - replace metadata
            AgentError::ExecutionFailedRich { message, .. } => {
                AgentError::ExecutionFailedRich { message, metadata }
            }
            AgentError::ParseErrorRich {
                message, reason, ..
            } => AgentError::ParseErrorRich {
                message,
                reason,
                metadata,
            },
            AgentError::ProcessErrorRich {
                status_code,
                message,
                is_retryable,
                retry_after,
                ..
            } => AgentError::ProcessErrorRich {
                status_code,
                message,
                is_retryable,
                retry_after,
                metadata,
            },
            // Other variants remain unchanged
            other => other,
        }
    }

    /// Logs this error with tracing, including all available metadata.
    ///
    /// This method provides structured logging that integrates with the observability
    /// system. Rich variants log detailed metadata including agent name, operation,
    /// span ID, and context. Simple variants log basic error information.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::AgentError;
    ///
    /// let err = AgentError::execution_failed_rich("Model timeout")
    ///     .agent("GeminiAgent")
    ///     .operation("execute")
    ///     .build();
    ///
    /// err.trace_error(); // Logs structured error with all metadata
    /// ```
    pub fn trace_error(&self) {
        match self {
            // Rich variants with full metadata
            AgentError::ExecutionFailedRich { message, metadata } => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "ExecutionFailedRich",
                    error_message = %message,
                    agent_name = ?metadata.agent_name,
                    agent_expertise = ?metadata.agent_expertise,
                    operation = ?metadata.operation,
                    span_id = ?metadata.span_id,
                    caused_by = ?metadata.caused_by_description,
                    context = ?metadata.context,
                    "Agent execution failed with context"
                );
            }
            AgentError::ParseErrorRich {
                message,
                reason,
                metadata,
            } => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "ParseErrorRich",
                    error_message = %message,
                    error_reason = ?reason,
                    agent_name = ?metadata.agent_name,
                    agent_expertise = ?metadata.agent_expertise,
                    operation = ?metadata.operation,
                    span_id = ?metadata.span_id,
                    caused_by = ?metadata.caused_by_description,
                    context = ?metadata.context,
                    "Parse error with context"
                );
            }
            AgentError::ProcessErrorRich {
                status_code,
                message,
                is_retryable,
                retry_after,
                metadata,
            } => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "ProcessErrorRich",
                    error_message = %message,
                    error_status_code = ?status_code,
                    error_is_retryable = %is_retryable,
                    error_retry_after = ?retry_after,
                    agent_name = ?metadata.agent_name,
                    agent_expertise = ?metadata.agent_expertise,
                    operation = ?metadata.operation,
                    span_id = ?metadata.span_id,
                    caused_by = ?metadata.caused_by_description,
                    context = ?metadata.context,
                    "Process error with context"
                );
            }
            // Simple variants - basic logging
            AgentError::ExecutionFailed(msg) => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "ExecutionFailed",
                    error_message = %msg,
                    "Agent execution failed (no metadata)"
                );
            }
            AgentError::ParseError { message, reason } => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "ParseError",
                    error_message = %message,
                    error_reason = ?reason,
                    "Parse error (no metadata)"
                );
            }
            AgentError::ProcessError {
                status_code,
                message,
                is_retryable,
                retry_after,
            } => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "ProcessError",
                    error_message = %message,
                    error_status_code = ?status_code,
                    error_is_retryable = %is_retryable,
                    error_retry_after = ?retry_after,
                    "Process error (no metadata)"
                );
            }
            AgentError::IoError(err) => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "IoError",
                    error_message = %err,
                    "I/O error"
                );
            }
            AgentError::JsonError(err) => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "JsonError",
                    error_message = %err,
                    "JSON serialization/deserialization error"
                );
            }
            AgentError::SerializationFailed(msg) => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "SerializationFailed",
                    error_message = %msg,
                    "Serialization failed"
                );
            }
            AgentError::Other(msg) => {
                tracing::error!(
                    target: "llm_toolkit::agent::error",
                    error_type = "Other",
                    error_message = %msg,
                    "Generic agent error"
                );
            }
        }
    }
}

/// Builder for creating rich errors with metadata.
///
/// This provides an ergonomic API for building errors with contextual information.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::AgentError;
/// use serde_json::json;
///
/// let error = AgentError::execution_failed_rich("Model timeout")
///     .agent("GeminiAgent")
///     .expertise("Fast inference agent")
///     .operation("execute")
///     .context("model", json!("gemini-2.5-flash"))
///     .context("timeout_ms", json!(30000))
///     .build();
/// ```
pub struct RichErrorBuilder {
    message: String,
    metadata: ErrorMetadata,
}

impl RichErrorBuilder {
    /// Creates a new builder with the given error message.
    pub fn new(message: String) -> Self {
        Self {
            message,
            metadata: ErrorMetadata::new(),
        }
    }

    /// Sets the agent name.
    pub fn agent(mut self, name: impl Into<String>) -> Self {
        self.metadata = self.metadata.with_agent(name);
        self
    }

    /// Sets the agent expertise/role.
    pub fn expertise(mut self, expertise: impl Into<String>) -> Self {
        self.metadata = self.metadata.with_expertise(expertise);
        self
    }

    /// Sets the operation where the error occurred.
    pub fn operation(mut self, op: impl Into<String>) -> Self {
        self.metadata = self.metadata.with_operation(op);
        self
    }

    /// Adds a key-value pair to the structured context.
    pub fn context(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata = self.metadata.with_context(key, value);
        self
    }

    /// Sets the span/trace ID for correlation.
    pub fn span_id(mut self, span_id: impl Into<String>) -> Self {
        self.metadata = self.metadata.with_span_id(span_id);
        self
    }

    /// Sets the cause of this error (for error chaining).
    pub fn caused_by(mut self, cause: &AgentError) -> Self {
        self.metadata = self.metadata.with_cause(cause);
        self
    }

    /// Builds the final rich error.
    pub fn build(self) -> AgentError {
        AgentError::ExecutionFailedRich {
            message: self.message,
            metadata: self.metadata,
        }
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

    // ========== Rich Error Tests ==========

    #[test]
    fn test_error_metadata_builder() {
        let metadata = ErrorMetadata::new()
            .with_agent("TestAgent")
            .with_expertise("Test expertise")
            .with_operation("test_op")
            .with_span_id("span-123")
            .with_context("key1", serde_json::json!("value1"))
            .with_context("key2", serde_json::json!(42));

        assert_eq!(metadata.agent_name, Some("TestAgent".to_string()));
        assert_eq!(metadata.agent_expertise, Some("Test expertise".to_string()));
        assert_eq!(metadata.operation, Some("test_op".to_string()));
        assert_eq!(metadata.span_id, Some("span-123".to_string()));
        assert!(metadata.context.is_some());
        assert!(metadata.timestamp.is_some());
    }

    #[test]
    fn test_rich_error_builder() {
        let err = AgentError::execution_failed_rich("Test failure")
            .agent("MyAgent")
            .expertise("General AI")
            .operation("execute")
            .context("model", serde_json::json!("claude-sonnet-4.5"))
            .build();

        match err {
            AgentError::ExecutionFailedRich { message, metadata } => {
                assert_eq!(message, "Test failure");
                assert_eq!(metadata.agent_name, Some("MyAgent".to_string()));
                assert_eq!(metadata.agent_expertise, Some("General AI".to_string()));
                assert_eq!(metadata.operation, Some("execute".to_string()));
            }
            _ => panic!("Expected ExecutionFailedRich variant"),
        }
    }

    #[test]
    fn test_with_metadata_conversion() {
        let simple_err = AgentError::ExecutionFailed("simple error".to_string());
        let metadata = ErrorMetadata::new()
            .with_agent("ConvertedAgent")
            .with_operation("conversion_test");

        let rich_err = simple_err.with_metadata(metadata);

        match rich_err {
            AgentError::ExecutionFailedRich { message, metadata } => {
                assert_eq!(message, "simple error");
                assert_eq!(metadata.agent_name, Some("ConvertedAgent".to_string()));
                assert_eq!(metadata.operation, Some("conversion_test".to_string()));
            }
            _ => panic!("Expected ExecutionFailedRich variant"),
        }
    }

    #[test]
    fn test_metadata_getter() {
        let rich_err = AgentError::execution_failed_rich("test")
            .agent("Agent1")
            .build();

        assert!(rich_err.metadata().is_some());
        assert_eq!(
            rich_err.metadata().unwrap().agent_name,
            Some("Agent1".to_string())
        );

        let simple_err = AgentError::ExecutionFailed("test".to_string());
        assert!(simple_err.metadata().is_none());
    }

    #[test]
    fn test_rich_error_is_retryable() {
        // ExecutionFailedRich should not be retryable
        let err = AgentError::execution_failed_rich("test")
            .agent("Agent1")
            .build();
        assert!(!err.is_retryable());

        // ParseErrorRich with UnexpectedEof should be retryable
        let err = AgentError::ParseErrorRich {
            message: "EOF".to_string(),
            reason: ParseErrorReason::UnexpectedEof,
            metadata: ErrorMetadata::new(),
        };
        assert!(err.is_retryable());

        // ProcessErrorRich with 429 should be retryable
        let err = AgentError::ProcessErrorRich {
            status_code: Some(429),
            message: "Rate limited".to_string(),
            is_retryable: false, // Even if false, 429 should be retryable
            retry_after: None,
            metadata: ErrorMetadata::new(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_rich_error_retry_delay() {
        // ProcessErrorRich with retry_after
        let err = AgentError::ProcessErrorRich {
            status_code: Some(429),
            message: "Rate limited".to_string(),
            is_retryable: true,
            retry_after: Some(Duration::from_secs(90)),
            metadata: ErrorMetadata::new(),
        };

        let delay = err.retry_delay(1);
        assert!(delay.as_secs() <= 90, "Delay should be <= 90 seconds");

        // ProcessErrorRich with 429 but no retry_after (exponential backoff)
        let err = AgentError::ProcessErrorRich {
            status_code: Some(429),
            message: "Rate limited".to_string(),
            is_retryable: true,
            retry_after: None,
            metadata: ErrorMetadata::new(),
        };

        let delay1 = err.retry_delay(1);
        assert!(delay1.as_secs() <= 2, "Attempt 1: delay <= 2s");

        let delay6 = err.retry_delay(6);
        assert!(delay6.as_secs() <= 60, "Attempt 6: delay <= 60s (capped)");
    }

    #[test]
    fn test_error_chaining() {
        let inner_err = AgentError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));

        let outer_err = AgentError::execution_failed_rich("Failed to load config")
            .agent("ConfigLoader")
            .caused_by(&inner_err)
            .build();

        match outer_err {
            AgentError::ExecutionFailedRich { metadata, .. } => {
                assert!(metadata.caused_by_description.is_some());
                let cause_desc = metadata.caused_by_description.unwrap();
                assert!(cause_desc.contains("I/O error"));
            }
            _ => panic!("Expected ExecutionFailedRich"),
        }
    }

    #[test]
    fn test_rich_error_display() {
        let err = AgentError::execution_failed_rich("Something went wrong")
            .agent("DisplayTestAgent")
            .operation("test_display")
            .build();

        let display_string = format!("{}", err);
        assert!(display_string.contains("Something went wrong"));
        assert!(display_string.contains("DisplayTestAgent"));
        assert!(display_string.contains("test_display"));
    }

    #[test]
    fn test_trace_error_does_not_panic() {
        // Test that trace_error() doesn't panic for various error types
        // Note: We can't easily test the actual tracing output without a subscriber,
        // but we can ensure the method doesn't panic

        let rich_err = AgentError::execution_failed_rich("test")
            .agent("TestAgent")
            .build();
        rich_err.trace_error(); // Should not panic

        let simple_err = AgentError::ExecutionFailed("test".to_string());
        simple_err.trace_error(); // Should not panic

        let parse_err = AgentError::ParseError {
            message: "test".to_string(),
            reason: ParseErrorReason::InvalidJson,
        };
        parse_err.trace_error(); // Should not panic

        let process_err = AgentError::ProcessError {
            status_code: Some(500),
            message: "test".to_string(),
            is_retryable: true,
            retry_after: None,
        };
        process_err.trace_error(); // Should not panic
    }
}
