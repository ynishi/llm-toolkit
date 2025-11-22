//! InnerValidatorAgent - A built-in agent for validating step outputs against task goals.
//!
//! This agent validates the output of a previous step against the overall task goal
//! and specific quality criteria. It returns a JSON string with validation results.

use crate::agent::impls::ClaudeCodeAgent;
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
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"A built-in agent that validates the output of a previous step against the overall task goal and specific quality criteria. It returns a JSON string with a 'status' ('PASS' or 'FAIL') and a 'reason'."
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        // Use ClaudeCodeAgent as the underlying LLM
        let claude = ClaudeCodeAgent::new();

        // Build validation prompt
        let validation_prompt = format!(
            r#"You are a validation agent. Your task is to validate whether the OUTPUT meets the requirements specified in the TASK.

Carefully analyze:
1. Does the OUTPUT address what was asked in the TASK?
2. Does the OUTPUT meet any quality criteria mentioned in the TASK?
3. Is the OUTPUT complete and well-formed?

TASK:
{}

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{
  "status": "PASS" or "FAIL",
  "reason": "Brief explanation of why the validation passed or failed"
}}"#,
            intent.to_text()
        );

        claude.execute(validation_prompt.into()).await
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
