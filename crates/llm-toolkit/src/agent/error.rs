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
    pub fn is_transient(&self) -> bool {
        matches!(self, AgentError::ProcessError(_) | AgentError::IoError(_))
    }
}
