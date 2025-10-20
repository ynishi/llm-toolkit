use super::{Agent, AgentError, Payload, PayloadContent};
use crate::ToPrompt;
use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(ToPrompt, Serialize, Clone, Debug)]
#[prompt(template = "
# Persona Profile
**Name**: {{ name }}
**Role**: {{ role }}

## Background
{{ background }}

## Communication Style
{{ communication_style }}
")]
pub struct Persona {
    pub name: &'static str,
    pub role: &'static str,
    pub background: &'static str,
    pub communication_style: &'static str,
}

pub struct PersonaAgent<T: Agent> {
    inner_agent: T,
    persona: Persona,
    dialogue_history: Arc<Mutex<Vec<String>>>,
}

impl<T: Agent> PersonaAgent<T> {
    pub fn new(inner_agent: T, persona: Persona) -> Self {
        Self {
            inner_agent,
            persona,
            dialogue_history: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl<T> Agent for PersonaAgent<T>
where
    T: Agent + Send + Sync,
    T::Output: Send,
{
    type Output = T::Output;

    fn expertise(&self) -> &str {
        self.persona.role
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let system_prompt = self.persona.to_prompt();
        let user_request = intent.to_text();

        let history_prompt = {
            let history = self.dialogue_history.lock().await;
            history.join("\n")
        };

        let final_prompt = format!(
            "{}\n\n# Conversation History\n{}\n\n# New Request\n{}",
            system_prompt, history_prompt, user_request
        );

        let mut final_payload = Payload::text(final_prompt);

        for content in intent.contents() {
            if let PayloadContent::Attachment(attachment) = content {
                final_payload = final_payload.with_attachment(attachment.clone());
            }
        }

        let response = self.inner_agent.execute(final_payload).await?;

        let mut history = self.dialogue_history.lock().await;

        if !user_request.is_empty() {
            history.push(format!("User: {}", user_request));
        }

        history.push(format!(
            "Assistant: {}",
            format_response_for_history(&response)
        ));

        Ok(response)
    }
}

fn format_response_for_history<T: Serialize>(response: &T) -> String {
    match serde_json::to_value(response) {
        Ok(serde_json::Value::String(text)) => text,
        Ok(other) => other.to_string(),
        Err(_) => "<non-serializable response>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentError, Payload};
    use crate::attachment::Attachment;
    use async_trait::async_trait;
    use serde::Deserialize;
    use serde::de::DeserializeOwned;
    use std::sync::Arc;
    use tokio::sync::Mutex;

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

        async fn last_call(&self) -> Option<Payload> {
            self.calls.lock().await.last().cloned()
        }
    }

    #[async_trait]
    impl<T> Agent for RecordingAgent<T>
    where
        T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        type Output = T;

        fn expertise(&self) -> &str {
            "Test agent"
        }

        async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
            self.calls.lock().await.push(intent);
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn persona_agent_preserves_attachments() {
        let persona = Persona {
            name: "Tester",
            role: "Attachment Checker",
            background: "Validates payload handling.",
            communication_style: "Direct and concise.",
        };

        let base_agent = RecordingAgent::new(String::from("ok"));
        let persona_agent = PersonaAgent::new(base_agent.clone(), persona);

        let attachment = Attachment::in_memory(vec![1, 2, 3]);
        let payload = Payload::text("Please inspect the data").with_attachment(attachment.clone());

        let _ = persona_agent.execute(payload).await.unwrap();

        let recorded_payload = base_agent.last_call().await.expect("call recorded");
        assert!(
            recorded_payload.has_attachments(),
            "attachments should be preserved"
        );
        let attachments = recorded_payload.attachments();
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0], &attachment);
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct StructuredResponse {
        message: String,
        value: u32,
    }

    #[tokio::test]
    async fn persona_agent_records_structured_history() {
        let persona = Persona {
            name: "Tester",
            role: "Structured Recorder",
            background: "Ensures structured outputs are captured.",
            communication_style: "Analytical.",
        };

        let response = StructuredResponse {
            message: "done".to_string(),
            value: 42,
        };
        let base_agent = RecordingAgent::new(response.clone());
        let persona_agent = PersonaAgent::new(base_agent, persona);

        let payload = Payload::text("Provide structured response");
        let _ = persona_agent.execute(payload).await.unwrap();

        let history = persona_agent.dialogue_history.lock().await.clone();
        assert_eq!(history.len(), 2);
        assert!(history[1].contains("\"message\":\"done\""));
        assert!(history[1].contains("\"value\":42"));
        assert!(
            history[1].starts_with("Assistant: "),
            "Assistant responses should be prefixed"
        );
    }
}
