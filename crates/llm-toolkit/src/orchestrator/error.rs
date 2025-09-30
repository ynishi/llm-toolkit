//! Error types for orchestrator operations.

use crate::agent::AgentError;
use thiserror::Error;

/// Errors that can occur during orchestrator operations.
#[derive(Debug, Error)]
pub enum OrchestratorError {
    /// An agent execution failed.
    #[error("Agent error: {0}")]
    AgentError(#[from] AgentError),

    /// No agent with the specified name was found in the registry.
    #[error("No agent found with name: {0}")]
    AgentNotFound(String),

    /// Strategy generation failed (e.g., LLM call error, parsing error).
    #[error("Strategy generation failed: {0}")]
    StrategyGenerationFailed(String),

    /// Strategy execution failed.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Redesign determination failed.
    #[error("Redesign determination failed: {0}")]
    RedesignFailed(String),

    /// JSON parsing error during strategy generation.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Generic error for other cases.
    #[error("Orchestrator error: {0}")]
    Other(String),
}

impl OrchestratorError {
    /// Creates an error indicating no strategy is available.
    pub fn no_strategy() -> Self {
        Self::Other("No strategy available".to_string())
    }

    /// Creates an error indicating the blueprint is invalid.
    pub fn invalid_blueprint(reason: impl Into<String>) -> Self {
        Self::Other(format!("Invalid blueprint: {}", reason.into()))
    }
}
