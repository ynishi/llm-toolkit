use super::Agent;
use crate::agent::dialogue::joining_strategy::JoiningStrategy;
use crate::agent::history::HistoryAwareAgent;
use crate::agent::persona::{Persona, PersonaAgent};

/// A builder for creating conversational agents with optional persona and history management.
///
/// The `Chat` builder provides a fluent interface for constructing agents with different
/// decorator layers. It supports:
/// - Adding a persona to shape the agent's communication style
/// - Enabling/disabling conversation history tracking
///
/// # Type Parameters
///
/// * `A` - The underlying agent type that implements `Agent`
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::agent::chat::Chat;
/// use llm_toolkit::agent::impls::ClaudeCodeAgent;
///
/// // Simple agent with history
/// let chat = Chat::new(ClaudeCodeAgent::new()).build();
///
/// // Agent with persona and history
/// let persona = Persona {
///     name: "Alice".to_string(),
///     role: "Helpful Assistant".to_string(),
///     background: "Expert in Rust programming".to_string(),
///     communication_style: "Clear and concise".to_string(),
///     visual_identity: None,
///     capabilities: None,
/// };
/// let chat_with_persona = Chat::new(ClaudeCodeAgent::new())
///     .with_persona(persona)
///     .build();
///
/// // Agent with persona but without history
/// let stateless_chat = Chat::new(ClaudeCodeAgent::new())
///     .with_persona(persona)
///     .with_history(false)
///     .build();
/// ```
pub struct Chat<A: Agent> {
    agent: A,
    with_history: bool,
    /// Identity information for history attribution (if persona is set)
    identity: Option<(String, String)>, // (name, role)
    joining_strategy: Option<JoiningStrategy>,
}

impl<A: Agent> Chat<A> {
    /// Creates a new `Chat` builder with the given base agent.
    ///
    /// By default, history tracking is enabled.
    ///
    /// # Arguments
    ///
    /// * `agent` - The base agent to wrap
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chat = Chat::new(ClaudeCodeAgent::new());
    /// ```
    pub fn new(agent: A) -> Self {
        Self {
            agent,
            with_history: true,
            identity: None,
            joining_strategy: None,
        }
    }

    /// Wraps the current agent with a `PersonaAgent` that shapes its communication style.
    ///
    /// This method changes the generic type parameter from `A` to `PersonaAgent<A>`,
    /// allowing further builder calls to operate on the persona-wrapped agent.
    ///
    /// # Arguments
    ///
    /// * `persona` - The persona configuration to apply
    ///
    /// # Example
    ///
    /// ```ignore
    /// let persona = Persona {
    ///     name: "Bob".to_string(),
    ///     role: "Code Reviewer".to_string(),
    ///     background: "Senior software engineer".to_string(),
    ///     communication_style: "Detailed and constructive".to_string(),
    ///     visual_identity: None,
    ///     capabilities: None,
    /// };
    /// let chat = Chat::new(agent).with_persona(persona);
    /// ```
    pub fn with_persona(self, persona: Persona) -> Chat<PersonaAgent<A>>
    where
        A::Output: Send,
    {
        let identity = Some((persona.name.clone(), persona.role.clone()));
        Chat {
            agent: PersonaAgent::new(self.agent, persona),
            with_history: self.with_history,
            identity,
            joining_strategy: self.joining_strategy,
        }
    }

    /// Enables or disables conversation history tracking.
    ///
    /// When enabled (default), the agent will maintain a history of all interactions
    /// and include that context in subsequent requests. When disabled, each request
    /// is processed independently.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable history tracking
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Disable history for stateless interactions
    /// let chat = Chat::new(agent).with_history(false);
    /// ```
    pub fn with_history(mut self, enabled: bool) -> Self {
        self.with_history = enabled;
        self
    }

    /// Sets a custom joining strategy for this chat agent.
    ///
    /// This method configures how much conversation history the agent receives
    /// when responding. It's primarily used for mid-dialogue participation scenarios
    /// through [`Dialogue::join_in_progress`], but can also be set directly for
    /// standalone chat agents with specific history requirements.
    ///
    /// # Arguments
    ///
    /// * `joining_strategy` - Optional strategy for filtering conversation history.
    ///   `None` means default behavior (all history if history is enabled).
    ///
    /// # Use Cases
    ///
    /// - **Mid-dialogue join**: Participant joining ongoing conversation needs
    ///   controlled history context
    /// - **Memory optimization**: Limit history for agents with token constraints
    /// - **Fresh perspective**: Remove historical bias for specific analysis tasks
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Create an agent that only sees recent 10 turns
    /// let chat = Chat::new(agent)
    ///     .with_history(true)
    ///     .with_joining_strategy(Some(JoiningStrategy::recent_with_turns(10)))
    ///     .build();
    ///
    /// // Create an agent with no history (fresh perspective)
    /// let chat = Chat::new(agent)
    ///     .with_history(true)
    ///     .with_joining_strategy(Some(JoiningStrategy::fresh()))
    ///     .build();
    /// ```
    ///
    /// # Note
    ///
    /// This setting only takes effect when `with_history(true)` is set.
    /// If history is disabled, this strategy is ignored.
    ///
    /// [`Dialogue::join_in_progress`]: crate::agent::dialogue::Dialogue::join_in_progress
    pub fn with_joining_strategy(mut self, joining_strategy: Option<JoiningStrategy>) -> Self {
        self.joining_strategy = joining_strategy;
        self
    }

    /// Finalizes the builder and returns the configured agent.
    ///
    /// This method constructs the final agent based on the builder's configuration:
    /// - If history is enabled, wraps the agent in `HistoryAwareAgent`
    /// - Returns a boxed trait object to allow different configurations to be
    ///   treated uniformly
    ///
    /// # Returns
    ///
    /// A boxed agent that implements the `Agent` trait with the configured output type.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chat = Chat::new(agent)
    ///     .with_persona(persona)
    ///     .build();
    ///
    /// // Use the built agent
    /// let response = chat.execute("Hello!".into()).await?;
    /// ```
    pub fn build(self) -> Box<crate::agent::AnyAgent<A::Output>>
    where
        A: 'static,
        A::Output: 'static + Send,
    {
        if self.with_history {
            match self.identity {
                Some((name, role)) => crate::agent::AnyAgent::boxed(
                    HistoryAwareAgent::new_with_identity(self.agent, name, role),
                ),
                None => crate::agent::AnyAgent::boxed(HistoryAwareAgent::new(self.agent)),
            }
        } else {
            crate::agent::AnyAgent::boxed(self.agent)
        }
    }
}

/// PersonaAgent-specific builder methods for Chat.
///
/// These methods are only available when the agent has been wrapped with a Persona
/// via `with_persona()`.
impl<A: Agent> Chat<PersonaAgent<A>>
where
    A::Output: Send,
{
    /// Configures the context placement strategy for the PersonaAgent.
    ///
    /// This allows you to customize how context, participants, and trailing prompts
    /// are positioned in the generated prompts to prevent confusion in long conversations.
    ///
    /// # Arguments
    ///
    /// * `config` - The ContextConfig with strategy options
    ///
    /// # Example
    ///
    /// ```ignore
    /// use llm_toolkit::agent::chat::Chat;
    /// use llm_toolkit::agent::persona::ContextConfig;
    ///
    /// let config = ContextConfig {
    ///     long_conversation_threshold: 5000,
    ///     recent_messages_count: 10,
    ///     participants_after_context: true,
    ///     include_trailing_prompt: true,
    /// };
    ///
    /// let chat = Chat::new(agent)
    ///     .with_persona(persona)
    ///     .with_context_config(config)
    ///     .build();
    /// ```
    pub fn with_context_config(mut self, config: crate::agent::persona::ContextConfig) -> Self {
        self.agent = self.agent.with_context_config(config);
        self
    }
}

// Note: Box<AnyAgent<T>> automatically implements Agent via Deref,
// so no explicit impl is needed.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentError, Payload, PayloadMessage};
    use async_trait::async_trait;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// A simple test agent that records calls and returns a fixed response.
    #[derive(Clone)]
    struct TestAgent {
        calls: Arc<Mutex<Vec<String>>>,
        response: String,
    }

    impl TestAgent {
        fn new(response: &str) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                response: response.to_string(),
            }
        }

        async fn get_calls(&self) -> Vec<String> {
            self.calls.lock().await.clone()
        }
    }

    #[async_trait]
    impl Agent for TestAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            const EXPERTISE: &str = "Test agent for Chat builder";
            &EXPERTISE
        }

        async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
            self.calls.lock().await.push(intent.to_text());
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_chat_builder_with_history() {
        let test_agent = TestAgent::new("response");
        let chat = Chat::new(test_agent.clone()).build();

        // First call - use from_messages instead of text
        let result1 = chat
            .execute(Payload::from_messages(vec![PayloadMessage::user(
                "User", "User", "Hello",
            )]))
            .await
            .unwrap();
        assert_eq!(result1, "response");

        // Second call should include history
        let result2 = chat
            .execute(Payload::from_messages(vec![PayloadMessage::user(
                "User",
                "User",
                "How are you?",
            )]))
            .await
            .unwrap();
        assert_eq!(result2, "response");

        // The inner agent should have been called twice
        let calls = test_agent.get_calls().await;
        assert_eq!(calls.len(), 2);

        // Second call should have history context
        assert!(calls[1].contains("Previous conversation"));
        assert!(calls[1].contains("Hello"));
    }

    #[tokio::test]
    async fn test_chat_builder_without_history() {
        let test_agent = TestAgent::new("response");
        let chat = Chat::new(test_agent.clone()).with_history(false).build();

        // First call
        let result1 = chat.execute(Payload::text("Hello")).await.unwrap();
        assert_eq!(result1, "response");

        // Second call should NOT include history
        let result2 = chat.execute(Payload::text("How are you?")).await.unwrap();
        assert_eq!(result2, "response");

        // The inner agent should have been called twice
        let calls = test_agent.get_calls().await;
        assert_eq!(calls.len(), 2);

        // Second call should NOT have history context
        assert!(!calls[1].contains("Previous conversation"));
        assert_eq!(calls[1], "How are you?");
    }

    #[tokio::test]
    async fn test_chat_builder_with_persona() {
        let test_agent = TestAgent::new("response");
        let persona = Persona {
            name: "TestBot".to_string(),
            role: "Test Assistant".to_string(),
            background: "A helpful test bot".to_string(),
            communication_style: "Direct and clear".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let chat = Chat::new(test_agent.clone())
            .with_persona(persona)
            .with_history(false) // Disable history to make testing simpler
            .build();

        let result = chat.execute(Payload::text("Hello")).await.unwrap();
        assert_eq!(result, "response");

        // Verify persona was applied
        let calls = test_agent.get_calls().await;
        assert_eq!(calls.len(), 1);
        assert!(calls[0].contains("Persona Profile"));
        assert!(calls[0].contains("TestBot"));
        assert!(calls[0].contains("Test Assistant"));
    }

    #[tokio::test]
    async fn test_chat_builder_with_persona_and_history() {
        let test_agent = TestAgent::new("response");
        let persona = Persona {
            name: "Alice".to_string(),
            role: "Assistant".to_string(),
            background: "Helpful AI".to_string(),
            communication_style: "Friendly".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let chat = Chat::new(test_agent.clone()).with_persona(persona).build();

        // First call - use from_messages
        let _ = chat
            .execute(Payload::from_messages(vec![PayloadMessage::user(
                "User", "User", "Hi",
            )]))
            .await
            .unwrap();

        // Second call should have both persona and history
        let _ = chat
            .execute(Payload::from_messages(vec![PayloadMessage::user(
                "User", "User", "Bye",
            )]))
            .await
            .unwrap();

        let calls = test_agent.get_calls().await;
        assert_eq!(calls.len(), 2);

        // Debug: print second call content
        println!("=== Second call ===\n{}\n=== End ===", calls[1]);

        // Second call should include both persona and history
        assert!(calls[1].contains("Previous conversation"));
        assert!(calls[1].contains("Persona Profile"));
        assert!(calls[1].contains("Alice"));
    }

    #[tokio::test]
    async fn test_chat_builder_expertise_delegation() {
        let test_agent = TestAgent::new("response");
        let chat = Chat::new(test_agent).build();

        // Without persona, should delegate to inner agent
        assert_eq!(chat.expertise(), "Test agent for Chat builder");
    }

    #[tokio::test]
    async fn test_chat_builder_expertise_with_persona() {
        let test_agent = TestAgent::new("response");
        let persona = Persona {
            name: "Bob".to_string(),
            role: "Expert Coder".to_string(),
            background: "Senior developer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let chat = Chat::new(test_agent)
            .with_persona(persona)
            .with_history(false)
            .build();

        // With persona, should use persona's role
        assert_eq!(chat.expertise(), "Expert Coder");
    }

    #[tokio::test]
    async fn test_chat_builder_with_context_config() {
        use crate::agent::persona::ContextConfig;

        let test_agent = TestAgent::new("response");
        let persona = Persona {
            name: "Alice".to_string(),
            role: "Assistant".to_string(),
            background: "Helpful assistant".to_string(),
            communication_style: "Friendly".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let config = ContextConfig {
            long_conversation_threshold: 100,
            recent_messages_count: 5,
            participants_after_context: true,
            include_trailing_prompt: true,
        };

        // Build chat with custom ContextConfig
        let chat = Chat::new(test_agent)
            .with_persona(persona)
            .with_context_config(config)
            .build();

        // Execute to verify it works
        let result = chat.execute(Payload::text("Test message")).await.unwrap();

        assert_eq!(result, "response");
    }
}
