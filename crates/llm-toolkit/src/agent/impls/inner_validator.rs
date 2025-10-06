//! InnerValidatorAgent - A built-in agent for validating step outputs against task goals.
//!
//! This agent validates the output of a previous step against the overall task goal
//! and specific quality criteria. It returns a JSON string with validation results.

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

/// A built-in agent that validates step outputs against quality criteria.
///
/// Returns a JSON string with validation results in the format:
/// ```json
/// {
///   "status": "PASS" | "FAIL",
///   "reason": "explanation of the validation result"
/// }
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Agent, InnerValidatorAgent};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let agent = InnerValidatorAgent::new();
///
///     let result = agent.execute(
///         "Validate if the code output follows Rust best practices".to_string()
///     ).await?;
///
///     println!("{}", result);
///     Ok(())
/// }
/// ```
pub struct InnerValidatorAgent;

impl InnerValidatorAgent {
    /// Creates a new InnerValidatorAgent.
    pub fn new() -> Self {
        Self
    }
}

impl Default for InnerValidatorAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for InnerValidatorAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "A built-in agent that validates the output of a previous step against the overall task goal and specific quality criteria. It returns a JSON string with a 'status' ('PASS' or 'FAIL') and a 'reason'."
    }

    async fn execute(&self, _intent: Payload) -> Result<Self::Output, AgentError> {
        // Placeholder implementation
        Ok(String::new())
    }

    fn name(&self) -> String {
        "InnerValidatorAgent".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_validator_agent_creation() {
        let agent = InnerValidatorAgent::new();
        assert_eq!(agent.name(), "InnerValidatorAgent");
        assert!(!agent.expertise().is_empty());
    }

    #[test]
    #[allow(clippy::default_constructed_unit_structs)]
    fn test_inner_validator_agent_default() {
        let agent = InnerValidatorAgent::default();
        assert_eq!(agent.name(), "InnerValidatorAgent");
    }
}
