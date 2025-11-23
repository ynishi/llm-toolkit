//! ExpertiseAgent: Automatic context-aware expertise rendering
//!
//! This module provides a wrapper agent that automatically applies context-aware
//! expertise rendering by extracting RenderContext from Payload and applying it
//! to Expertise prompt generation.

use super::expertise::Expertise;
use super::{Agent, AgentError, Capability, Payload, ToExpertise};
use crate::prompt::{PromptPart, ToPrompt};
use async_trait::async_trait;

/// Agent wrapper that automatically applies context-aware expertise rendering.
///
/// This wrapper composes an inner agent with an Expertise definition,
/// automatically extracting render context from Payload and applying it
/// to expertise prompt generation.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::{ExpertiseAgent, impls::ClaudeCodeAgent};
/// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
/// use llm_toolkit::{Priority, ContextProfile, TaskHealth};
///
/// // Define expertise
/// let expertise = Expertise::new("rust-reviewer", "1.0")
///     .with_fragment(
///         WeightedFragment::new(KnowledgeFragment::Text(
///             "Always verify code compiles".to_string()
///         ))
///         .with_priority(Priority::Critical)
///     )
///     .with_fragment(
///         WeightedFragment::new(KnowledgeFragment::Text(
///             "SLOW DOWN. Extra vigilance required.".to_string()
///         ))
///         .with_priority(Priority::Critical)
///         .with_context(ContextProfile::Conditional {
///             task_types: vec![],
///             user_states: vec![],
///             task_health: Some(TaskHealth::AtRisk),
///         })
///     );
///
/// // Wrap with agent
/// let agent = ExpertiseAgent::new(
///     ClaudeCodeAgent::new(),
///     expertise
/// );
///
/// // Execute with context
/// let payload = Payload::text("Review this code")
///     .with_task_type("security-review")
///     .with_task_health(TaskHealth::AtRisk);
///
/// let result = agent.execute(payload).await?;
/// // Automatically includes "SLOW DOWN" fragment due to AtRisk health
/// ```
pub struct ExpertiseAgent<T: Agent> {
    inner_agent: T,
    expertise: Expertise,
}

impl<T: Agent> ExpertiseAgent<T> {
    /// Creates a new ExpertiseAgent.
    ///
    /// # Arguments
    ///
    /// * `inner_agent` - The underlying agent that will execute the enriched payload
    /// * `expertise` - The expertise definition with conditional fragments
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::{ExpertiseAgent, impls::ClaudeCodeAgent};
    /// use llm_toolkit::agent::expertise::Expertise;
    ///
    /// let expertise = Expertise::new("code-reviewer", "1.0");
    /// let agent = ExpertiseAgent::new(ClaudeCodeAgent::new(), expertise);
    /// ```
    pub fn new(inner_agent: T, expertise: Expertise) -> Self {
        Self {
            inner_agent,
            expertise,
        }
    }

    /// Returns a reference to the inner agent.
    pub fn inner(&self) -> &T {
        &self.inner_agent
    }

    /// Returns a reference to the expertise.
    pub fn expertise_ref(&self) -> &Expertise {
        &self.expertise
    }
}

#[async_trait]
impl<T> Agent for ExpertiseAgent<T>
where
    T: Agent + Send + Sync,
    T::Output: Send,
{
    type Output = T::Output;
    type Expertise = Expertise;

    fn expertise(&self) -> &Expertise {
        &self.expertise
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        // Extract render context from Payload
        let render_context = intent.render_context().cloned().unwrap_or_default();

        // Generate context-aware expertise prompt
        let expertise_prompt = self.expertise.to_prompt_with_context(&render_context);

        // Prepend expertise to payload as system context
        let enriched_payload = intent.prepend_system(expertise_prompt);

        // Execute inner agent
        self.inner_agent.execute(enriched_payload).await
    }

    fn name(&self) -> String {
        format!("ExpertiseAgent<{}>", self.inner_agent.name())
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        self.inner_agent.is_available().await
    }
}

// Implement ToPrompt (required by ToExpertise)
impl<T: Agent> ToPrompt for ExpertiseAgent<T> {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        self.expertise.to_prompt_parts()
    }

    fn to_prompt(&self) -> String {
        self.expertise.to_prompt()
    }
}

// Implement ToExpertise for convenience
impl<T: Agent> ToExpertise for ExpertiseAgent<T> {
    fn description(&self) -> &str {
        self.expertise.description()
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.expertise.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::expertise::{KnowledgeFragment, WeightedFragment};
    use crate::context::{ContextProfile, TaskHealth};

    // Mock agent for testing
    struct MockAgent {
        response: String,
    }

    #[async_trait]
    impl Agent for MockAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            &"Mock agent"
        }

        async fn execute(&self, _intent: Payload) -> Result<Self::Output, AgentError> {
            Ok(self.response.clone())
        }

        fn name(&self) -> String {
            "MockAgent".to_string()
        }

        async fn is_available(&self) -> Result<(), AgentError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_expertise_agent_basic() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let agent = ExpertiseAgent::new(mock, expertise);

        let payload = Payload::text("Question");
        let result = agent.execute(payload).await.unwrap();

        assert_eq!(result, "Mock response");
    }

    #[tokio::test]
    async fn test_expertise_agent_with_context() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test", "1.0")
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("Always visible".to_string()))
                    .with_context(ContextProfile::Always),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("AtRisk only".to_string()))
                    .with_context(ContextProfile::Conditional {
                        task_types: vec![],
                        user_states: vec![],
                        task_health: Some(TaskHealth::AtRisk),
                    }),
            );

        let agent = ExpertiseAgent::new(mock, expertise);

        let payload = Payload::text("Question").with_task_health(TaskHealth::AtRisk);

        let result = agent.execute(payload).await.unwrap();
        assert_eq!(result, "Mock response");
    }

    #[tokio::test]
    async fn test_expertise_agent_name() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test", "1.0");
        let agent = ExpertiseAgent::new(mock, expertise);

        assert_eq!(agent.name(), "ExpertiseAgent<MockAgent>");
    }

    #[tokio::test]
    async fn test_expertise_agent_is_available() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test", "1.0");
        let agent = ExpertiseAgent::new(mock, expertise);

        assert!(agent.is_available().await.is_ok());
    }

    #[tokio::test]
    async fn test_expertise_agent_inner_access() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test", "1.0");
        let agent = ExpertiseAgent::new(mock, expertise);

        assert_eq!(agent.inner().name(), "MockAgent");
    }

    #[tokio::test]
    async fn test_expertise_agent_expertise_ref() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test-expertise", "1.0");
        let agent = ExpertiseAgent::new(mock, expertise);

        assert_eq!(agent.expertise_ref().id, "test-expertise");
    }

    #[test]
    fn test_expertise_agent_to_expertise() {
        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise =
            Expertise::new("test-desc", "1.0").with_description("Test description for routing");

        let agent = ExpertiseAgent::new(mock, expertise);

        // ToExpertise implementation
        assert_eq!(
            ToExpertise::description(&agent),
            "Test description for routing"
        );
        assert_eq!(ToExpertise::capabilities(&agent).len(), 0);
    }

    #[test]
    fn test_expertise_agent_to_prompt() {
        use crate::agent::expertise::{KnowledgeFragment, WeightedFragment};

        let mock = MockAgent {
            response: "Mock response".to_string(),
        };

        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let agent = ExpertiseAgent::new(mock, expertise);

        let prompt = agent.to_prompt();
        assert!(prompt.contains("Test content"));
    }
}
