use super::{Agent, AgentError, Payload, PayloadContent};
use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

/// An agent wrapper that maintains dialogue history across multiple executions.
///
/// This agent wraps any inner agent and automatically maintains a history of all
/// interactions (user requests and agent responses). The history is prepended to
/// each new request, allowing the agent to have context of previous interactions.
///
/// # Example
///
/// ```ignore
/// use llm_toolkit::agent::{Agent, history::HistoryAwareAgent};
///
/// let base_agent = MyAgent::new();
/// let history_agent = HistoryAwareAgent::new(base_agent);
///
/// // First interaction
/// let response1 = history_agent.execute("What is Rust?".into()).await?;
///
/// // Second interaction - the agent will have context from the first
/// let response2 = history_agent.execute("Tell me more about it".into()).await?;
/// ```
pub struct HistoryAwareAgent<T: Agent> {
    inner_agent: T,
    dialogue_history: Arc<Mutex<Vec<String>>>,
}

impl<T: Agent> HistoryAwareAgent<T> {
    /// Creates a new history-aware agent wrapping the given inner agent.
    ///
    /// # Arguments
    ///
    /// * `inner_agent` - The agent to wrap with history tracking
    pub fn new(inner_agent: T) -> Self {
        Self {
            inner_agent,
            dialogue_history: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl<T> Agent for HistoryAwareAgent<T>
where
    T: Agent + Send + Sync,
    T::Output: Send,
{
    type Output = T::Output;

    fn expertise(&self) -> &str {
        self.inner_agent.expertise()
    }

    #[crate::tracing::instrument(
        name = "history_aware_agent.execute",
        skip(self, intent),
        fields(
            agent.expertise = self.inner_agent.expertise(),
            has_history = !self.dialogue_history.try_lock().map(|h| h.is_empty()).unwrap_or(true),
        )
    )]
    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        // Lock history and build context
        let history = self.dialogue_history.lock().await;
        let history_prompt = if history.is_empty() {
            String::new()
        } else {
            format!("# Previous Conversation\n{}\n\n", history.join("\n"))
        };
        drop(history);

        // Get user request text
        let user_request = intent.to_text();

        // Create combined prompt with history context
        let combined_prompt = if history_prompt.is_empty() {
            user_request.clone()
        } else {
            format!("{}{}", history_prompt, user_request)
        };

        // Create new payload with combined prompt while preserving attachments
        let mut final_payload = Payload::text(combined_prompt);

        for content in intent.contents() {
            if let PayloadContent::Attachment(attachment) = content {
                final_payload = final_payload.with_attachment(attachment.clone());
            }
        }

        // Execute the inner agent
        let response = self.inner_agent.execute(final_payload).await?;

        // Add to history after execution
        let mut history = self.dialogue_history.lock().await;
        history.push(format!("User: {}", user_request));
        history.push(format!(
            "Assistant: {}",
            format_response_for_history(&response)
        ));

        Ok(response)
    }
}

/// Helper function to format agent output for storage in history.
///
/// Converts the structured output to a string representation suitable for
/// inclusion in the conversation history.
fn format_response_for_history<T: Serialize>(output: &T) -> String {
    serde_json::to_string_pretty(output)
        .unwrap_or_else(|_| format!("{:?}", std::any::type_name::<T>()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentError, Payload};
    use async_trait::async_trait;
    use serde::de::DeserializeOwned;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// A test agent that records all calls and returns a predefined response.
    #[derive(Clone)]
    struct RecordingAgent<T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static> {
        calls: Arc<Mutex<Vec<Payload>>>,
        response: T,
    }

    impl<T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static> RecordingAgent<T> {
        fn new(response: T) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                response,
            }
        }

        async fn get_calls(&self) -> Vec<Payload> {
            self.calls.lock().await.clone()
        }

        async fn call_count(&self) -> usize {
            self.calls.lock().await.len()
        }
    }

    #[async_trait]
    impl<T> Agent for RecordingAgent<T>
    where
        T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        type Output = T;

        fn expertise(&self) -> &str {
            "Test recording agent"
        }

        async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
            self.calls.lock().await.push(intent);
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_history_tracking_across_multiple_calls() {
        let base_agent = RecordingAgent::new(String::from("Response 1"));
        let history_agent = HistoryAwareAgent::new(base_agent.clone());

        // First call
        let payload1 = Payload::text("What is Rust?");
        let response1 = history_agent.execute(payload1).await.unwrap();
        assert_eq!(response1, "Response 1");

        // Verify first call was recorded
        assert_eq!(base_agent.call_count().await, 1);

        // Second call - should include history
        let base_agent2 = RecordingAgent::new(String::from("Response 2"));
        let history_agent2 = HistoryAwareAgent {
            inner_agent: base_agent2.clone(),
            dialogue_history: history_agent.dialogue_history.clone(),
        };

        let payload2 = Payload::text("Tell me more");
        let response2 = history_agent2.execute(payload2).await.unwrap();
        assert_eq!(response2, "Response 2");

        // Verify the second agent received history in its prompt
        let calls = base_agent2.get_calls().await;
        assert_eq!(calls.len(), 1);
        let received_text = calls[0].to_text();

        // The second call should include the previous conversation
        assert!(received_text.contains("Previous Conversation"));
        assert!(received_text.contains("User: What is Rust?"));
        assert!(received_text.contains("Assistant: \"Response 1\""));
        assert!(received_text.contains("Tell me more"));
    }

    #[tokio::test]
    async fn test_history_preserves_attachments() {
        use crate::attachment::Attachment;

        let base_agent = RecordingAgent::new(String::from("ok"));
        let history_agent = HistoryAwareAgent::new(base_agent.clone());

        // First call with attachment
        let attachment = Attachment::in_memory(vec![1, 2, 3]);
        let payload = Payload::text("Analyze this data").with_attachment(attachment.clone());

        let _ = history_agent.execute(payload).await.unwrap();

        // Verify attachment was preserved
        let calls = base_agent.get_calls().await;
        assert_eq!(calls.len(), 1);
        assert!(calls[0].has_attachments());

        let attachments = calls[0].attachments();
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0], &attachment);
    }

    #[tokio::test]
    async fn test_empty_history_on_first_call() {
        let base_agent = RecordingAgent::new(String::from("First response"));
        let history_agent = HistoryAwareAgent::new(base_agent.clone());

        let payload = Payload::text("Hello");
        let _ = history_agent.execute(payload).await.unwrap();

        // First call should not have history prefix
        let calls = base_agent.get_calls().await;
        assert_eq!(calls.len(), 1);
        let received_text = calls[0].to_text();

        // Should not contain "Previous Conversation" since it's the first call
        assert!(!received_text.contains("Previous Conversation"));
        assert_eq!(received_text, "Hello");
    }

    #[tokio::test]
    async fn test_expertise_delegation() {
        let base_agent = RecordingAgent::new(String::from("response"));
        let history_agent = HistoryAwareAgent::new(base_agent);

        assert_eq!(history_agent.expertise(), "Test recording agent");
    }
}
