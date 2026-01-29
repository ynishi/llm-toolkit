//! OllamaApiAgent - Ollama HTTP API implementation.
//!
//! This agent calls the Ollama REST API for local LLM inference.
//! Supports models like Llama 3, Qwen, CodeLlama, and others.
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_toolkit::agent::impls::OllamaApiAgent;
//! use llm_toolkit::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Default configuration (localhost:11434, llama3)
//! let agent = OllamaApiAgent::new();
//! let response = agent.execute("Hello, world!".into()).await?;
//!
//! // Custom model
//! let agent = OllamaApiAgent::new().with_model("qwen2.5-coder:1.5b");
//!
//! // Custom endpoint
//! let agent = OllamaApiAgent::new()
//!     .with_endpoint("http://192.168.1.100:11434")
//!     .with_model("codellama");
//! # Ok(())
//! # }
//! ```

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use std::env;
use std::sync::Arc;
use tokio::sync::RwLock;

const DEFAULT_MODEL: &str = "llama3";
const DEFAULT_HOST: &str = "http://localhost";
const DEFAULT_PORT: u16 = 11434;

/// Agent implementation that talks to the Ollama HTTP API.
#[derive(Clone)]
pub struct OllamaApiAgent {
    client: Arc<RwLock<Ollama>>,
    model: String,
    endpoint: String,
    system_prompt: Option<String>,
}

impl Default for OllamaApiAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaApiAgent {
    /// Creates a new agent with default configuration.
    ///
    /// Defaults:
    /// - Endpoint: `http://localhost:11434`
    /// - Model: `llama3`
    pub fn new() -> Self {
        let client = Ollama::new(DEFAULT_HOST.to_string(), DEFAULT_PORT);
        Self {
            client: Arc::new(RwLock::new(client)),
            model: DEFAULT_MODEL.to_string(),
            endpoint: format!("{}:{}", DEFAULT_HOST, DEFAULT_PORT),
            system_prompt: None,
        }
    }

    /// Loads configuration from environment variables.
    ///
    /// Environment variables:
    /// - `OLLAMA_HOST` (optional, defaults to `http://localhost:11434`)
    /// - `OLLAMA_MODEL` (optional, defaults to `llama3`)
    pub fn from_env() -> Self {
        let endpoint = env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| format!("{}:{}", DEFAULT_HOST, DEFAULT_PORT));
        let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        Self::new().with_endpoint(&endpoint).with_model(&model)
    }

    /// Overrides the model after construction.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the endpoint after construction.
    ///
    /// Endpoint format: `http://host:port` (e.g., `http://localhost:11434`)
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        let endpoint_str = endpoint.into();
        let (host, port) = Self::parse_endpoint(&endpoint_str);
        let client = Ollama::new(host, port);
        self.client = Arc::new(RwLock::new(client));
        self.endpoint = endpoint_str;
        self
    }

    /// Adds a system prompt that will be prepended to every request.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Returns the current model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the current endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Parse endpoint string into host and port.
    ///
    /// Input: "http://localhost:11434" -> ("http://localhost", 11434)
    fn parse_endpoint(endpoint: &str) -> (String, u16) {
        // Find the last colon that's followed by digits (port)
        if let Some(pos) = endpoint.rfind(':') {
            let port_str = &endpoint[pos + 1..];
            if let Ok(port) = port_str.parse::<u16>() {
                let host = &endpoint[..pos];
                return (host.to_string(), port);
            }
        }
        // Default if parsing fails
        (DEFAULT_HOST.to_string(), DEFAULT_PORT)
    }

    /// Check if the Ollama server is healthy.
    pub async fn is_healthy(&self) -> bool {
        let client = self.client.read().await;
        client.list_local_models().await.is_ok()
    }

    /// List available models on the Ollama server.
    pub async fn list_models(&self) -> Result<Vec<String>, AgentError> {
        let client = self.client.read().await;
        let models = client
            .list_local_models()
            .await
            .map_err(|e| AgentError::ExecutionFailed(format!("Failed to list models: {}", e)))?;
        Ok(models.into_iter().map(|m| m.name).collect())
    }

    /// Call Ollama API with the given prompt.
    async fn call_ollama(&self, prompt: &str) -> Result<String, AgentError> {
        let client = self.client.read().await;

        let full_prompt = if let Some(system) = &self.system_prompt {
            format!("{}\n\n{}", system, prompt)
        } else {
            prompt.to_string()
        };

        let request = GenerationRequest::new(self.model.clone(), full_prompt);

        match client.generate(request).await {
            Ok(response) => Ok(response.response),
            Err(e) => {
                let message = e.to_string();
                // Determine if error is retryable
                let is_retryable = message.contains("connection")
                    || message.contains("timeout")
                    || message.contains("temporarily");

                Err(AgentError::ProcessError {
                    status_code: None,
                    message: format!("Ollama API error: {}", message),
                    is_retryable,
                    retry_after: None,
                })
            }
        }
    }
}

#[async_trait]
impl Agent for OllamaApiAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"Ollama API agent for local LLM inference"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let text = payload.to_text();
        if text.trim().is_empty() {
            return Err(AgentError::ExecutionFailed(
                "Ollama payload must include text".into(),
            ));
        }

        self.call_ollama(&text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_agent_creation() {
        let agent = OllamaApiAgent::new();
        assert_eq!(agent.model(), "llama3");
        assert_eq!(agent.endpoint(), "http://localhost:11434");
        assert!(agent.system_prompt.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let agent = OllamaApiAgent::new()
            .with_model("qwen2.5-coder:1.5b")
            .with_endpoint("http://192.168.1.100:11434")
            .with_system_prompt("You are a helpful assistant.");

        assert_eq!(agent.model(), "qwen2.5-coder:1.5b");
        assert_eq!(agent.endpoint(), "http://192.168.1.100:11434");
        assert_eq!(
            agent.system_prompt,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_parse_endpoint() {
        // Standard format
        let (host, port) = OllamaApiAgent::parse_endpoint("http://localhost:11434");
        assert_eq!(host, "http://localhost");
        assert_eq!(port, 11434);

        // Custom port
        let (host, port) = OllamaApiAgent::parse_endpoint("http://192.168.1.100:8080");
        assert_eq!(host, "http://192.168.1.100");
        assert_eq!(port, 8080);

        // Invalid format (no port)
        let (host, port) = OllamaApiAgent::parse_endpoint("http://localhost");
        assert_eq!(host, DEFAULT_HOST);
        assert_eq!(port, DEFAULT_PORT);

        // Invalid port
        let (host, port) = OllamaApiAgent::parse_endpoint("http://localhost:abc");
        assert_eq!(host, DEFAULT_HOST);
        assert_eq!(port, DEFAULT_PORT);
    }

    #[test]
    fn test_from_env_defaults() {
        // Clear env vars for test
        unsafe {
            std::env::remove_var("OLLAMA_HOST");
            std::env::remove_var("OLLAMA_MODEL");
        }

        let agent = OllamaApiAgent::from_env();
        assert_eq!(agent.model(), "llama3");
        assert_eq!(agent.endpoint(), "http://localhost:11434");
    }

    #[test]
    fn test_default_trait() {
        let agent = OllamaApiAgent::default();
        assert_eq!(agent.model(), "llama3");
    }
}
