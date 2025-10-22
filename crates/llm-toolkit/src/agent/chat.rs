use super::{Agent, AgentError, Payload};
use crate::agent::history::HistoryAwareAgent;
use crate::agent::persona::{Persona, PersonaAgent};
use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};

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
///     name: "Alice",
///     role: "Helpful Assistant",
///     background: "Expert in Rust programming",
///     communication_style: "Clear and concise",
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
    ///     name: "Bob",
    ///     role: "Code Reviewer",
    ///     background: "Senior software engineer",
    ///     communication_style: "Detailed and constructive",
    /// };
    /// let chat = Chat::new(agent).with_persona(persona);
    /// ```
    pub fn with_persona(self, persona: Persona) -> Chat<PersonaAgent<A>>
    where
        A::Output: Send,
    {
        Chat {
            agent: PersonaAgent::new(self.agent, persona),
            with_history: self.with_history,
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
    pub fn build(self) -> Box<dyn Agent<Output = A::Output>>
    where
        A: 'static,
        A::Output: 'static + Send,
    {
        if self.with_history {
            Box::new(HistoryAwareAgent::new(self.agent))
        } else {
            Box::new(self.agent)
        }
    }
}

/// Implementation of the `Agent` trait for boxed agents.
///
/// This allows the boxed result from `build()` to be used directly as an agent.
#[async_trait]
impl<T> Agent for Box<dyn Agent<Output = T>>
where
    T: Serialize + DeserializeOwned + Send + Sync,
{
    type Output = T;

    fn expertise(&self) -> &str {
        (**self).expertise()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        (**self).execute(intent).await
    }

    fn name(&self) -> String {
        (**self).name()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        (**self).is_available().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        fn expertise(&self) -> &str {
            "Test agent for Chat builder"
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

        // First call
        let result1 = chat.execute(Payload::text("Hello")).await.unwrap();
        assert_eq!(result1, "response");

        // Second call should include history
        let result2 = chat.execute(Payload::text("How are you?")).await.unwrap();
        assert_eq!(result2, "response");

        // The inner agent should have been called twice
        let calls = test_agent.get_calls().await;
        assert_eq!(calls.len(), 2);

        // Second call should have history context
        assert!(calls[1].contains("Previous Conversation"));
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
        assert!(!calls[1].contains("Previous Conversation"));
        assert_eq!(calls[1], "How are you?");
    }

    #[tokio::test]
    async fn test_chat_builder_with_persona() {
        let test_agent = TestAgent::new("response");
        let persona = Persona {
            name: "TestBot",
            role: "Test Assistant",
            background: "A helpful test bot",
            communication_style: "Direct and clear",
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
            name: "Alice",
            role: "Assistant",
            background: "Helpful AI",
            communication_style: "Friendly",
        };

        let chat = Chat::new(test_agent.clone()).with_persona(persona).build();

        // First call
        let _ = chat.execute(Payload::text("Hi")).await.unwrap();

        // Second call should have both persona and history
        let _ = chat.execute(Payload::text("Bye")).await.unwrap();

        let calls = test_agent.get_calls().await;
        assert_eq!(calls.len(), 2);

        // Second call should include both persona and history
        assert!(calls[1].contains("Previous Conversation"));
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
            name: "Bob",
            role: "Expert Coder",
            background: "Senior developer",
            communication_style: "Technical",
        };

        let chat = Chat::new(test_agent)
            .with_persona(persona)
            .with_history(false)
            .build();

        // With persona, should use persona's role
        assert_eq!(chat.expertise(), "Expert Coder");
    }
}
