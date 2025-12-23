//! AnthropicApiAgent - Direct REST API implementation for Claude (Anthropic).
//!
//! This agent calls the Claude REST API directly without CLI dependency.
//! API key can be provided directly or loaded from environment variables.
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_toolkit::agent::impls::AnthropicApiAgent;
//! use llm_toolkit::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // From environment variable (ANTHROPIC_API_KEY)
//! let agent = AnthropicApiAgent::try_from_env()?;
//! let response = agent.execute("Hello, world!".into()).await?;
//!
//! // Direct API key
//! let agent = AnthropicApiAgent::new("your-api-key", "claude-sonnet-4-20250514");
//!
//! // With options
//! let agent = AnthropicApiAgent::new("your-api-key", "claude-sonnet-4-20250514")
//!     .with_system("You are a helpful assistant")
//!     .with_max_tokens(4096);
//! # Ok(())
//! # }
//! ```

use crate::agent::{Agent, AgentError, Payload};
use crate::attachment::Attachment;
use crate::models::ClaudeModel;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use reqwest::{Client, StatusCode, header::HeaderValue};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;
const BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Agent implementation that talks to the Claude (Anthropic) HTTP API.
#[derive(Clone)]
pub struct AnthropicApiAgent {
    client: Client,
    api_key: String,
    model: String,
    system: Option<String>,
    max_tokens: u32,
}

impl AnthropicApiAgent {
    /// Creates a new agent with the provided API key and model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            system: None,
            max_tokens: 4096,
        }
    }

    /// Loads configuration from environment variables.
    ///
    /// Environment variables:
    /// - `ANTHROPIC_API_KEY` (required)
    /// - `ANTHROPIC_MODEL` (optional, defaults to Claude Sonnet 4.5)
    pub fn try_from_env() -> Result<Self, AgentError> {
        let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| {
            AgentError::ExecutionFailed(
                "ANTHROPIC_API_KEY environment variable not set".to_string(),
            )
        })?;

        let model = env::var("ANTHROPIC_MODEL")
            .map(|s| s.parse::<ClaudeModel>().unwrap_or_default().as_api_id().to_string())
            .unwrap_or_else(|_| ClaudeModel::default().as_api_id().to_string());

        Ok(Self::new(api_key, model))
    }

    /// Overrides the model after construction using a string.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the model using a typed [`ClaudeModel`].
    pub fn with_claude_model(mut self, model: ClaudeModel) -> Self {
        self.model = model.as_api_id().to_string();
        self
    }

    /// Adds a system prompt that will be sent alongside every request.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    async fn build_content(&self, payload: &Payload) -> Result<Vec<ContentBlock>, AgentError> {
        let mut content_blocks = Vec::new();

        let text = payload.to_text();
        if !text.trim().is_empty() {
            content_blocks.push(ContentBlock::Text { text });
        }

        for attachment in payload.attachments() {
            if let Some(block) = Self::attachment_to_content_block(attachment).await? {
                content_blocks.push(block);
            }
        }

        if content_blocks.is_empty() {
            return Err(AgentError::ExecutionFailed(
                "Claude payload must include text or supported attachments".into(),
            ));
        }

        Ok(content_blocks)
    }

    async fn attachment_to_content_block(
        attachment: &Attachment,
    ) -> Result<Option<ContentBlock>, AgentError> {
        if let Attachment::Remote(_) = attachment {
            return Err(AgentError::ExecutionFailed(
                "Remote attachments are not supported for Claude API".into(),
            ));
        }

        let bytes = attachment.load_bytes().await.map_err(|err| {
            AgentError::ExecutionFailed(format!("Failed to load attachment for Claude API: {err}"))
        })?;

        let media_type = attachment
            .mime_type()
            .unwrap_or_else(|| "application/octet-stream".to_string());

        let data = BASE64_STANDARD.encode(bytes);

        Ok(Some(ContentBlock::Image {
            source: ImageSource {
                r#type: "base64".to_string(),
                media_type,
                data,
            },
        }))
    }

    async fn send_request(&self, body: &CreateMessageRequest) -> Result<String, AgentError> {
        let response = self
            .client
            .post(BASE_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|err| AgentError::ProcessError {
                status_code: None,
                message: format!("Claude API request failed: {err}"),
                is_retryable: err.is_connect() || err.is_timeout(),
                retry_after: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers().get("retry-after"));
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read Claude error body".to_string());
            return Err(map_http_error(status, body_text, retry_after));
        }

        let parsed: CreateMessageResponse = response
            .json()
            .await
            .map_err(|err| AgentError::Other(format!("Failed to parse Claude response: {err}")))?;

        extract_text_response(parsed)
    }
}

#[async_trait]
impl Agent for AnthropicApiAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"Claude API agent for advanced reasoning and coding tasks"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let content = self.build_content(&payload).await?;

        let messages = vec![Message {
            role: "user".to_string(),
            content,
        }];

        let request = CreateMessageRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            system: self.system.clone(),
        };

        self.send_request(&request).await
    }
}

#[derive(Serialize)]
struct CreateMessageRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: Vec<ContentBlock>,
}

enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
}

impl Serialize for ContentBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;

        match self {
            ContentBlock::Text { text } => {
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
            }
            ContentBlock::Image { source } => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", source)?;
            }
        }

        map.end()
    }
}

#[derive(Serialize)]
struct ImageSource {
    r#type: String,
    media_type: String,
    data: String,
}

#[derive(Deserialize)]
struct CreateMessageResponse {
    content: Vec<ContentBlockResponse>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlockResponse {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[allow(dead_code)]
    r#type: String,
    message: String,
}

fn extract_text_response(response: CreateMessageResponse) -> Result<String, AgentError> {
    response
        .content
        .into_iter()
        .map(|block| match block {
            ContentBlockResponse::Text { text } => text,
        })
        .next()
        .ok_or_else(|| {
            AgentError::ExecutionFailed(
                "Claude API returned no text in the response content".into(),
            )
        })
}

fn map_http_error(status: StatusCode, body: String, retry_after: Option<Duration>) -> AgentError {
    let message = serde_json::from_str::<ErrorResponse>(&body)
        .map(|wrapper| wrapper.error.message)
        .unwrap_or_else(|_| body.clone());

    let is_retryable = matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    );

    if let Some(delay) = retry_after {
        AgentError::process_error_with_retry_after(status.as_u16(), message, is_retryable, delay)
    } else {
        AgentError::ProcessError {
            status_code: Some(status.as_u16()),
            message,
            is_retryable,
            retry_after: None,
        }
    }
}

fn parse_retry_after(header: Option<&HeaderValue>) -> Option<Duration> {
    let value = header?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_api_agent_creation() {
        let agent = AnthropicApiAgent::new("test-key", "claude-sonnet-4-20250514");
        assert_eq!(agent.model, "claude-sonnet-4-20250514");
        assert!(agent.system.is_none());
        assert_eq!(agent.max_tokens, 4096);
    }

    #[test]
    fn test_builder_methods() {
        let agent = AnthropicApiAgent::new("test-key", "claude-sonnet-4-20250514")
            .with_model("claude-opus-4-20250514")
            .with_system("You are a helpful assistant")
            .with_max_tokens(8192);

        assert_eq!(agent.model, "claude-opus-4-20250514");
        assert_eq!(
            agent.system,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(agent.max_tokens, 8192);
    }

    #[test]
    fn test_request_serialization() {
        let request = CreateMessageRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Hello".to_string(),
                }],
            }],
            max_tokens: 4096,
            system: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"claude-sonnet-4-20250514\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"max_tokens\":4096"));
    }

    #[test]
    fn test_request_serialization_with_system() {
        let request = CreateMessageRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Hello".to_string(),
                }],
            }],
            max_tokens: 4096,
            system: Some("You are a helpful assistant".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"system\":\"You are a helpful assistant\""));
    }

    #[test]
    fn test_request_serialization_with_image() {
        let request = CreateMessageRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: vec![
                    ContentBlock::Text {
                        text: "What's in this image?".to_string(),
                    },
                    ContentBlock::Image {
                        source: ImageSource {
                            r#type: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "base64encodeddata".to_string(),
                        },
                    },
                ],
            }],
            max_tokens: 4096,
            system: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"type\":\"image\""));
        assert!(json.contains("\"media_type\":\"image/png\""));
    }

    #[test]
    fn test_response_parsing() {
        let json = r#"{
            "content": [{
                "type": "text",
                "text": "Hello, world!"
            }]
        }"#;

        let response: CreateMessageResponse = serde_json::from_str(json).unwrap();
        let text = extract_text_response(response).unwrap();
        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_response_parsing_empty_content() {
        let json = r#"{"content": []}"#;
        let response: CreateMessageResponse = serde_json::from_str(json).unwrap();
        let result = extract_text_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parsing() {
        let json = r#"{
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        }"#;

        let error = map_http_error(StatusCode::UNAUTHORIZED, json.to_string(), None);
        match error {
            AgentError::ProcessError { message, .. } => {
                assert!(message.contains("Invalid API key"));
            }
            _ => panic!("Expected ProcessError"),
        }
    }

    #[test]
    fn test_retryable_status_codes() {
        let retryable_statuses = [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ];

        for status in retryable_statuses {
            let error = map_http_error(status, "error".to_string(), None);
            match error {
                AgentError::ProcessError { is_retryable, .. } => {
                    assert!(is_retryable, "Status {:?} should be retryable", status);
                }
                _ => panic!("Expected ProcessError"),
            }
        }
    }

    #[test]
    fn test_non_retryable_status_codes() {
        let non_retryable_statuses = [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
        ];

        for status in non_retryable_statuses {
            let error = map_http_error(status, "error".to_string(), None);
            match error {
                AgentError::ProcessError { is_retryable, .. } => {
                    assert!(!is_retryable, "Status {:?} should not be retryable", status);
                }
                _ => panic!("Expected ProcessError"),
            }
        }
    }

    #[test]
    fn test_try_from_env_missing_key() {
        // SAFETY: This test runs single-threaded (--test-threads=1)
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
        let result = AnthropicApiAgent::try_from_env();
        assert!(result.is_err());
    }
}
