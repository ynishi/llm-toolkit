//! GenaiAgent - Unified multi-provider LLM agent powered by the `genai` crate.
//!
//! This agent provides a single implementation that can talk to **any** provider
//! supported by genai (OpenAI, Anthropic, Gemini, xAI, Ollama, Groq, DeepSeek,
//! Cohere, and more) through a normalized Chat Completion API.
//!
//! The provider is automatically resolved from the model name string
//! (e.g. `"gpt-5"` → OpenAI, `"claude-sonnet-4-6"` → Anthropic).
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_toolkit::agent::impls::GenaiAgent;
//! use llm_toolkit::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Provider is inferred from the model name
//! let agent = GenaiAgent::new("claude-sonnet-4-6");
//! let response = agent.execute("Hello, world!".into()).await?;
//!
//! // With options
//! let agent = GenaiAgent::new("gpt-5")
//!     .with_system("You are a helpful assistant")
//!     .with_max_tokens(4096)
//!     .with_temperature(0.7);
//! # Ok(())
//! # }
//! ```

use crate::agent::{Agent, AgentError, Payload};
use crate::attachment::Attachment;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use genai::Client;
use genai::chat::{Binary, ChatMessage, ChatOptions, ChatRequest, ContentPart, MessageContent};

/// Unified multi-provider LLM agent.
///
/// Uses the `genai` crate to normalize Chat Completion calls across providers.
/// The provider is resolved automatically from the model name.
#[derive(Clone)]
pub struct GenaiAgent {
    client: Client,
    model: String,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl GenaiAgent {
    /// Creates a new agent for the given model name.
    ///
    /// The provider is inferred from the model string:
    /// - `"gpt-*"` / `"o1-*"` / `"o3-*"` → OpenAI
    /// - `"claude-*"` → Anthropic
    /// - `"gemini-*"` → Google
    /// - `"deepseek-*"` → DeepSeek
    /// - `"grok-*"` → xAI
    /// - etc.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: Client::default(),
            model: model.into(),
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Creates a new agent with a custom `genai::Client`.
    ///
    /// Use this when you need custom authentication, endpoints, or adapter
    /// configuration beyond the defaults.
    pub fn with_client(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Sets the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the top-p (nucleus sampling) parameter.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Overrides the model after construction.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Returns the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns a reference to the underlying `genai::Client`.
    pub fn client(&self) -> &Client {
        &self.client
    }

    fn build_chat_options(&self) -> Option<ChatOptions> {
        if self.max_tokens.is_none() && self.temperature.is_none() && self.top_p.is_none() {
            return None;
        }

        let mut opts = ChatOptions::default();
        if let Some(max_tokens) = self.max_tokens {
            opts = opts.with_max_tokens(max_tokens);
        }
        if let Some(temperature) = self.temperature {
            opts = opts.with_temperature(temperature);
        }
        if let Some(top_p) = self.top_p {
            opts = opts.with_top_p(top_p);
        }
        Some(opts)
    }

    async fn build_chat_request(&self, payload: &Payload) -> Result<ChatRequest, AgentError> {
        let mut messages = Vec::new();

        if let Some(ref system) = self.system {
            messages.push(ChatMessage::system(system.as_str()));
        }

        let text = payload.to_text();
        let attachments = payload.attachments();

        if attachments.is_empty() {
            messages.push(ChatMessage::user(text));
        } else {
            let mut parts: Vec<ContentPart> = Vec::new();

            if !text.trim().is_empty() {
                parts.push(ContentPart::Text(text));
            }

            for attachment in attachments {
                let binary = Self::attachment_to_binary(attachment).await?;
                parts.push(ContentPart::Binary(binary));
            }

            let content = MessageContent::from_parts(parts);
            messages.push(ChatMessage::user(content));
        }

        Ok(ChatRequest::new(messages))
    }

    async fn attachment_to_binary(attachment: &Attachment) -> Result<Binary, AgentError> {
        match attachment {
            Attachment::Local(path) => Binary::from_file(path).map_err(|e| {
                AgentError::ExecutionFailed(format!(
                    "Failed to load local attachment for genai: {e}"
                ))
            }),
            Attachment::Remote(url) => {
                let mime = attachment
                    .mime_type()
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                Ok(Binary::from_url(mime, url.as_str(), attachment.file_name()))
            }
            Attachment::InMemory {
                bytes,
                file_name,
                mime_type,
            } => {
                let mime = mime_type
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                let b64 = BASE64_STANDARD.encode(bytes);
                Ok(Binary::from_base64(mime, b64, file_name.clone()))
            }
        }
    }
}

#[async_trait]
impl Agent for GenaiAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"Unified multi-provider LLM agent powered by genai"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let chat_req = self.build_chat_request(&payload).await?;
        let chat_options = self.build_chat_options();

        let chat_res = self
            .client
            .exec_chat(&self.model, chat_req, chat_options.as_ref())
            .await
            .map_err(|e| AgentError::ProcessError {
                status_code: None,
                message: format!("genai exec_chat failed: {e}"),
                is_retryable: false,
                retry_after: None,
            })?;

        chat_res
            .first_text()
            .ok_or_else(|| {
                AgentError::ExecutionFailed(
                    "genai returned no text content in the response".to_string(),
                )
            })
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genai_agent_creation() {
        let agent = GenaiAgent::new("gpt-5");
        assert_eq!(agent.model(), "gpt-5");
        assert!(agent.system.is_none());
        assert!(agent.max_tokens.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let agent = GenaiAgent::new("claude-sonnet-4-6")
            .with_system("You are helpful")
            .with_max_tokens(4096)
            .with_temperature(0.7)
            .with_top_p(0.9);

        assert_eq!(agent.model(), "claude-sonnet-4-6");
        assert_eq!(agent.system.as_deref(), Some("You are helpful"));
        assert_eq!(agent.max_tokens, Some(4096));
        assert_eq!(agent.temperature, Some(0.7));
        assert_eq!(agent.top_p, Some(0.9));
    }

    #[test]
    fn test_with_model_override() {
        let agent = GenaiAgent::new("gpt-5").with_model("gemini-2.5-flash");
        assert_eq!(agent.model(), "gemini-2.5-flash");
    }

    #[test]
    fn test_chat_options_none_when_defaults() {
        let agent = GenaiAgent::new("gpt-5");
        assert!(agent.build_chat_options().is_none());
    }

    #[test]
    fn test_chat_options_some_when_configured() {
        let agent = GenaiAgent::new("gpt-5").with_max_tokens(1000);
        assert!(agent.build_chat_options().is_some());
    }

    #[tokio::test]
    async fn test_chat_request_with_system() {
        let agent = GenaiAgent::new("gpt-5").with_system("Be concise");
        let payload = Payload::text("Hello");
        let req = agent.build_chat_request(&payload).await.unwrap();
        // ChatRequest should have 2 messages (system + user)
        assert_eq!(req.messages.len(), 2);
    }

    #[tokio::test]
    async fn test_chat_request_without_system() {
        let agent = GenaiAgent::new("gpt-5");
        let payload = Payload::text("Hello");
        let req = agent.build_chat_request(&payload).await.unwrap();
        // ChatRequest should have 1 message (user only)
        assert_eq!(req.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_chat_request_with_in_memory_attachment() {
        let agent = GenaiAgent::new("gpt-5");
        let payload = Payload::text("What is in this image?").with_attachment(
            Attachment::in_memory_with_meta(
                vec![0x89, 0x50, 0x4E, 0x47],
                Some("test.png".to_string()),
                Some("image/png".to_string()),
            ),
        );
        let req = agent.build_chat_request(&payload).await.unwrap();
        assert_eq!(req.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_chat_request_with_remote_attachment() {
        let agent = GenaiAgent::new("gpt-5");
        let payload = Payload::text("Describe this")
            .with_attachment(Attachment::remote("https://example.com/image.png"));
        let req = agent.build_chat_request(&payload).await.unwrap();
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_attachment_to_binary_in_memory() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let attachment = Attachment::in_memory_with_meta(
            vec![1, 2, 3],
            Some("data.bin".to_string()),
            Some("application/octet-stream".to_string()),
        );
        let binary = rt
            .block_on(GenaiAgent::attachment_to_binary(&attachment))
            .unwrap();
        assert_eq!(binary.content_type, "application/octet-stream");
        assert_eq!(binary.name, Some("data.bin".to_string()));
    }

    #[test]
    fn test_attachment_to_binary_remote() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let attachment = Attachment::remote("https://example.com/photo.jpg");
        let binary = rt
            .block_on(GenaiAgent::attachment_to_binary(&attachment))
            .unwrap();
        assert_eq!(binary.content_type, "application/octet-stream");
    }
}
