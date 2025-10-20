use super::{Agent, AgentError, Payload};
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
impl<T: Agent<Output = String> + Send + Sync> Agent for PersonaAgent<T> {
    type Output = String;

    fn expertise(&self) -> &str {
        self.persona.role
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let system_prompt = self.persona.to_prompt();
        let user_request = intent.to_text();

        let mut history = self.dialogue_history.lock().await;

        let history_prompt = history.join("\n");

        let final_prompt = format!(
            "{}\n\n# Conversation History\n{}\n\n# New Request\n{}",
            system_prompt, history_prompt, user_request
        );

        let response = self
            .inner_agent
            .execute(Payload::text(final_prompt))
            .await?;

        history.push(format!("User: {}", user_request));
        history.push(format!("Assistant: {}", response));

        Ok(response)
    }
}
