//! Error types for the agent module.

use thiserror::Error;

/// Errors that can occur during agent execution.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The agent execution failed with a specific error message.
    #[error("Agent execution failed: {0}")]
    ExecutionFailed(String),

    /// Failed to parse the agent's output into the expected type.
    #[error("Failed to parse agent output: {0}")]
    ParseError(String),

    /// The agent process spawning or communication failed.
    #[error("Process error: {0}")]
    ProcessError(String),

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
    /// Check if this error is transient (likely to succeed on retry).
    ///
    /// **Deprecated:** Use `is_retryable()` instead.
    pub fn is_transient(&self) -> bool {
        matches!(self, AgentError::ProcessError(_) | AgentError::IoError(_))
    }

    /// Check if this error should trigger an automatic retry.
    ///
    /// Returns `true` for errors that are likely transient and may succeed on retry:
    /// - `ParseError`: LLM output might be malformed, retry with clearer prompt
    /// - `ProcessError`: Process communication issues (network, etc.)
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
        matches!(
            self,
            AgentError::ParseError(_) | AgentError::ProcessError(_) | AgentError::IoError(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_retryable_parse_error() {
        let err = AgentError::ParseError("malformed JSON".to_string());
        assert!(
            err.is_retryable(),
            "ParseError should be retryable (LLM output might be malformed)"
        );
    }

    #[test]
    fn test_is_retryable_process_error() {
        let err = AgentError::ProcessError("process crashed".to_string());
        assert!(
            err.is_retryable(),
            "ProcessError should be retryable (transient process issues)"
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
        let process_err = AgentError::ProcessError("process issue".to_string());
        assert!(
            process_err.is_transient(),
            "ProcessError should be transient (backward compatibility)"
        );

        let io_err = AgentError::IoError(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "timeout",
        ));
        assert!(
            io_err.is_transient(),
            "IoError should be transient (backward compatibility)"
        );

        // But ParseError should NOT be transient (only retryable)
        let parse_err = AgentError::ParseError("malformed".to_string());
        assert!(
            !parse_err.is_transient(),
            "ParseError is retryable but not transient (semantic difference)"
        );
    }
}
