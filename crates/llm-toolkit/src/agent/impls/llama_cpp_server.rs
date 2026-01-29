//! LlamaCppServerAgent - llama-server HTTP API implementation.
//!
//! This agent calls the llama.cpp server (llama-server) REST API for local LLM inference.
//! The server must be started separately with a model loaded.
//!
//! # Server Setup
//!
//! ```bash
//! # Start llama-server with a model
//! llama-server -m model.gguf --host 0.0.0.0 --port 8080
//!
//! # With multiple slots for concurrent requests
//! llama-server -m model.gguf --host 0.0.0.0 --port 8080 -np 4
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_toolkit::agent::impls::LlamaCppServerAgent;
//! use llm_toolkit::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Default configuration (localhost:8080)
//! let agent = LlamaCppServerAgent::new();
//!
//! // Custom endpoint and chat template
//! let agent = LlamaCppServerAgent::new()
//!     .with_endpoint("http://192.168.1.100:8080")
//!     .with_chat_template(ChatTemplate::Llama3)
//!     .with_temperature(0.7);
//!
//! let response = agent.execute("Hello, world!".into()).await?;
//! # Ok(())
//! # }
//! ```

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

const DEFAULT_ENDPOINT: &str = "http://localhost:8080";
const DEFAULT_MAX_TOKENS: usize = 512;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.9;
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Chat template format for different model families.
#[derive(Debug, Clone)]
pub enum ChatTemplate {
    /// Llama 3 format
    Llama3,
    /// Qwen format (Qwen, Qwen2, Qwen2.5)
    Qwen,
    /// LFM2 format (Sakana AI)
    Lfm2,
    /// Mistral/Mixtral format
    Mistral,
    /// ChatML format (general)
    ChatMl,
    /// No template (raw prompt)
    None,
    /// Custom template
    Custom {
        user_prefix: String,
        user_suffix: String,
        assistant_prefix: String,
    },
}

impl Default for ChatTemplate {
    fn default() -> Self {
        ChatTemplate::Llama3
    }
}

impl ChatTemplate {
    /// Format a prompt with this template.
    pub fn format(&self, prompt: &str) -> String {
        match self {
            ChatTemplate::Llama3 => {
                format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    prompt
                )
            }
            ChatTemplate::Qwen => {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    prompt
                )
            }
            ChatTemplate::Lfm2 => {
                format!("<|user|>\n{}\n<|assistant|>\n", prompt)
            }
            ChatTemplate::Mistral => {
                format!("[INST] {} [/INST]", prompt)
            }
            ChatTemplate::ChatMl => {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    prompt
                )
            }
            ChatTemplate::None => prompt.to_string(),
            ChatTemplate::Custom {
                user_prefix,
                user_suffix,
                assistant_prefix,
            } => {
                format!("{}{}{}{}", user_prefix, prompt, user_suffix, assistant_prefix)
            }
        }
    }

    /// Get stop tokens for this template.
    pub fn stop_tokens(&self) -> Vec<String> {
        match self {
            ChatTemplate::Llama3 => {
                vec!["<|eot_id|>".to_string(), "<|start_header_id|>".to_string()]
            }
            ChatTemplate::Qwen | ChatTemplate::ChatMl => {
                vec![
                    "<|im_end|>".to_string(),
                    "<|im_start|>".to_string(),
                    "<|endoftext|>".to_string(),
                ]
            }
            ChatTemplate::Lfm2 => {
                vec!["<|user|>".to_string(), "<|endoftext|>".to_string()]
            }
            ChatTemplate::Mistral => {
                vec!["[INST]".to_string(), "</s>".to_string()]
            }
            ChatTemplate::None => vec![],
            ChatTemplate::Custom { .. } => vec![],
        }
    }
}

/// Configuration for LlamaCppServerAgent.
#[derive(Debug, Clone)]
pub struct LlamaCppServerConfig {
    /// Server endpoint (e.g., "http://localhost:8080")
    pub endpoint: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Chat template to use
    pub chat_template: ChatTemplate,
    /// System prompt (optional)
    pub system_prompt: Option<String>,
}

impl Default for LlamaCppServerConfig {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            chat_template: ChatTemplate::default(),
            system_prompt: None,
        }
    }
}

/// Agent implementation for llama-server HTTP API.
#[derive(Clone)]
pub struct LlamaCppServerAgent {
    config: LlamaCppServerConfig,
    client: Client,
}

impl Default for LlamaCppServerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaCppServerAgent {
    /// Creates a new agent with default configuration.
    ///
    /// Defaults:
    /// - Endpoint: `http://localhost:8080`
    /// - Max tokens: 512
    /// - Temperature: 0.7
    /// - Chat template: Llama3
    pub fn new() -> Self {
        let config = LlamaCppServerConfig::default();
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Creates an agent from environment variables.
    ///
    /// Environment variables:
    /// - `LLAMA_SERVER_ENDPOINT` (optional, defaults to `http://localhost:8080`)
    /// - `LLAMA_SERVER_MAX_TOKENS` (optional, defaults to 512)
    /// - `LLAMA_SERVER_TEMPERATURE` (optional, defaults to 0.7)
    pub fn from_env() -> Self {
        let endpoint = env::var("LLAMA_SERVER_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        let max_tokens = env::var("LLAMA_SERVER_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let temperature = env::var("LLAMA_SERVER_TEMPERATURE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TEMPERATURE);

        Self::new()
            .with_endpoint(endpoint)
            .with_max_tokens(max_tokens)
            .with_temperature(temperature)
    }

    /// Sets the server endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.endpoint = endpoint.into();
        self
    }

    /// Sets the maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    /// Sets the temperature for sampling.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Sets the top-p (nucleus) sampling parameter.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = top_p;
        self
    }

    /// Sets the request timeout in seconds.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self.client = Client::builder()
            .timeout(Duration::from_secs(secs))
            .build()
            .expect("Failed to create HTTP client");
        self
    }

    /// Sets the chat template.
    pub fn with_chat_template(mut self, template: ChatTemplate) -> Self {
        self.config.chat_template = template;
        self
    }

    /// Sets a system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Returns the current endpoint.
    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    /// Checks if the server is healthy.
    pub async fn is_healthy(&self) -> bool {
        let url = format!("{}/health", self.config.endpoint);
        match self.client.get(&url).send().await {
            Ok(response) => {
                if let Ok(health) = response.json::<HealthResponse>().await {
                    health.status == "ok"
                } else {
                    // Some versions return 200 without JSON body
                    true
                }
            }
            Err(_) => false,
        }
    }

    /// Gets the number of available slots on the server.
    pub async fn available_slots(&self) -> Result<usize, AgentError> {
        let url = format!("{}/slots", self.config.endpoint);
        let response = self.client.get(&url).send().await.map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to get slots: {}", e))
        })?;

        let slots: Vec<serde_json::Value> = response.json().await.map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to parse slots response: {}", e))
        })?;

        Ok(slots.len())
    }

    /// Calls the completion API.
    async fn call_completion(&self, prompt: &str) -> Result<String, AgentError> {
        // Apply system prompt if set
        let full_prompt = if let Some(ref system) = self.config.system_prompt {
            format!("{}\n\n{}", system, prompt)
        } else {
            prompt.to_string()
        };

        // Apply chat template
        let formatted_prompt = self.config.chat_template.format(&full_prompt);
        let stop_tokens = self.config.chat_template.stop_tokens();

        let request = CompletionRequest {
            prompt: formatted_prompt,
            n_predict: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            stream: false,
            stop: stop_tokens,
        };

        let url = format!("{}/completion", self.config.endpoint);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                let is_retryable = e.is_timeout() || e.is_connect();
                AgentError::ProcessError {
                    status_code: None,
                    message: format!("llama-server request failed: {}", e),
                    is_retryable,
                    retry_after: None,
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::ProcessError {
                status_code: Some(status.as_u16()),
                message: format!("llama-server error: {}", body),
                is_retryable: status.is_server_error(),
                retry_after: None,
            });
        }

        let completion: CompletionResponse = response.json().await.map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to parse response: {}", e))
        })?;

        Ok(completion.content)
    }
}

#[async_trait]
impl Agent for LlamaCppServerAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"llama-server agent for local LLM inference with llama.cpp"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let text = payload.to_text();
        if text.trim().is_empty() {
            return Err(AgentError::ExecutionFailed(
                "Payload must include text".into(),
            ));
        }

        self.call_completion(&text).await
    }
}

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Debug, Serialize)]
struct CompletionRequest {
    prompt: String,
    n_predict: usize,
    temperature: f32,
    top_p: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let agent = LlamaCppServerAgent::new();
        assert_eq!(agent.endpoint(), "http://localhost:8080");
        assert_eq!(agent.config.max_tokens, 512);
        assert!((agent.config.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder() {
        let agent = LlamaCppServerAgent::new()
            .with_endpoint("http://192.168.1.100:9000")
            .with_max_tokens(1024)
            .with_temperature(0.5)
            .with_top_p(0.95)
            .with_system_prompt("You are a helpful assistant.");

        assert_eq!(agent.endpoint(), "http://192.168.1.100:9000");
        assert_eq!(agent.config.max_tokens, 1024);
        assert!((agent.config.temperature - 0.5).abs() < f32::EPSILON);
        assert!((agent.config.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(
            agent.config.system_prompt,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_chat_template_llama3() {
        let template = ChatTemplate::Llama3;
        let formatted = template.format("Hello");
        assert!(formatted.contains("<|start_header_id|>user"));
        assert!(formatted.contains("<|eot_id|>"));
        assert!(formatted.contains("<|start_header_id|>assistant"));
    }

    #[test]
    fn test_chat_template_qwen() {
        let template = ChatTemplate::Qwen;
        let formatted = template.format("Hello");
        assert!(formatted.contains("<|im_start|>user"));
        assert!(formatted.contains("<|im_end|>"));
        assert!(formatted.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_chat_template_mistral() {
        let template = ChatTemplate::Mistral;
        let formatted = template.format("Hello");
        assert_eq!(formatted, "[INST] Hello [/INST]");
    }

    #[test]
    fn test_chat_template_none() {
        let template = ChatTemplate::None;
        let formatted = template.format("Hello");
        assert_eq!(formatted, "Hello");
    }

    #[test]
    fn test_chat_template_custom() {
        let template = ChatTemplate::Custom {
            user_prefix: "[USER]".to_string(),
            user_suffix: "[/USER]".to_string(),
            assistant_prefix: "[ASSISTANT]".to_string(),
        };
        let formatted = template.format("Hello");
        assert_eq!(formatted, "[USER]Hello[/USER][ASSISTANT]");
    }

    #[test]
    fn test_stop_tokens() {
        let llama3 = ChatTemplate::Llama3;
        assert!(!llama3.stop_tokens().is_empty());

        let none = ChatTemplate::None;
        assert!(none.stop_tokens().is_empty());
    }

    #[test]
    fn test_from_env_defaults() {
        unsafe {
            std::env::remove_var("LLAMA_SERVER_ENDPOINT");
            std::env::remove_var("LLAMA_SERVER_MAX_TOKENS");
            std::env::remove_var("LLAMA_SERVER_TEMPERATURE");
        }

        let agent = LlamaCppServerAgent::from_env();
        assert_eq!(agent.endpoint(), "http://localhost:8080");
    }

    #[test]
    fn test_request_serialization() {
        let request = CompletionRequest {
            prompt: "Hello".to_string(),
            n_predict: 100,
            temperature: 0.7,
            top_p: 0.9,
            stream: false,
            stop: vec!["<|eot_id|>".to_string()],
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"prompt\":\"Hello\""));
        assert!(json.contains("\"n_predict\":100"));
        assert!(json.contains("\"stop\":[\"<|eot_id|>\"]"));
    }

    #[test]
    fn test_request_serialization_empty_stop() {
        let request = CompletionRequest {
            prompt: "Hello".to_string(),
            n_predict: 100,
            temperature: 0.7,
            top_p: 0.9,
            stream: false,
            stop: vec![],
        };

        let json = serde_json::to_string(&request).unwrap();
        // stop should be omitted when empty
        assert!(!json.contains("\"stop\""));
    }
}
