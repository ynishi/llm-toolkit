//! OpenAIApiAgent - Direct REST API implementation for OpenAI GPT.
//!
//! This agent calls the OpenAI Chat Completions API directly without CLI dependency.
//! API key can be provided directly or loaded from environment variables.
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_toolkit::agent::impls::OpenAIApiAgent;
//! use llm_toolkit::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // From environment variable (OPENAI_API_KEY)
//! let agent = OpenAIApiAgent::try_from_env()?;
//! let response = agent.execute("Hello, world!".into()).await?;
//!
//! // Direct API key
//! let agent = OpenAIApiAgent::new("your-api-key", "gpt-4o");
//!
//! // With options
//! let agent = OpenAIApiAgent::new("your-api-key", "gpt-4o")
//!     .with_max_tokens(4096);
//! # Ok(())
//! # }
//! ```

use crate::agent::{Agent, AgentError, Payload};
use crate::attachment::Attachment;
use crate::models::OpenAIModel;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use reqwest::{Client, StatusCode, header::HeaderValue};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

const BASE_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Agent implementation that talks to the OpenAI HTTP API.
#[derive(Clone)]
pub struct OpenAIApiAgent {
    client: Client,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
}

impl OpenAIApiAgent {
    /// Creates a new agent with the provided API key and model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: None,
        }
    }

    /// Loads configuration from environment variables.
    ///
    /// Environment variables:
    /// - `OPENAI_API_KEY` (required)
    /// - `OPENAI_MODEL` (optional, defaults to GPT-4o)
    pub fn try_from_env() -> Result<Self, AgentError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
            AgentError::ExecutionFailed("OPENAI_API_KEY environment variable not set".to_string())
        })?;

        let model = env::var("OPENAI_MODEL")
            .map(|s| {
                s.parse::<OpenAIModel>()
                    .unwrap_or_default()
                    .as_api_id()
                    .to_string()
            })
            .unwrap_or_else(|_| OpenAIModel::default().as_api_id().to_string());

        Ok(Self::new(api_key, model))
    }

    /// Overrides the model after construction using a string.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the model using a typed [`OpenAIModel`].
    pub fn with_openai_model(mut self, model: OpenAIModel) -> Self {
        self.model = model.as_api_id().to_string();
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    async fn build_messages(&self, payload: &Payload) -> Result<Vec<ChatMessage>, AgentError> {
        let mut content_parts = Vec::new();

        let text = payload.to_text();
        if !text.trim().is_empty() {
            content_parts.push(MessageContent::Text { text });
        }

        for attachment in payload.attachments() {
            if let Some(content) = Self::attachment_to_content(attachment).await? {
                content_parts.push(content);
            }
        }

        if content_parts.is_empty() {
            return Err(AgentError::ExecutionFailed(
                "OpenAI payload must include text or supported attachments".into(),
            ));
        }

        Ok(vec![ChatMessage {
            role: "user".to_string(),
            content: content_parts,
        }])
    }

    async fn attachment_to_content(
        attachment: &Attachment,
    ) -> Result<Option<MessageContent>, AgentError> {
        if let Attachment::Remote(url) = attachment {
            return Ok(Some(MessageContent::ImageUrl {
                image_url: ImageUrl {
                    url: url.to_string(),
                },
            }));
        }

        let bytes = attachment.load_bytes().await.map_err(|err| {
            AgentError::ExecutionFailed(format!("Failed to load attachment for OpenAI API: {err}"))
        })?;

        let mime_type = attachment
            .mime_type()
            .unwrap_or_else(|| "image/jpeg".to_string());

        let data_url = format!(
            "data:{};base64,{}",
            mime_type,
            BASE64_STANDARD.encode(bytes)
        );

        Ok(Some(MessageContent::ImageUrl {
            image_url: ImageUrl { url: data_url },
        }))
    }

    async fn send_request(&self, body: &ChatCompletionRequest) -> Result<String, AgentError> {
        let response = self
            .client
            .post(BASE_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|err| AgentError::ProcessError {
                status_code: None,
                message: format!("OpenAI API request failed: {err}"),
                is_retryable: err.is_connect() || err.is_timeout(),
                retry_after: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers().get("retry-after"));
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read OpenAI error body".to_string());
            return Err(map_http_error(status, body_text, retry_after));
        }

        let parsed: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|err| AgentError::Other(format!("Failed to parse OpenAI response: {err}")))?;

        extract_text_response(parsed)
    }
}

#[async_trait]
impl Agent for OpenAIApiAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"OpenAI GPT agent for general-purpose reasoning and coding tasks"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let messages = self.build_messages(&payload).await?;

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
        };

        self.send_request(&request).await
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: Vec<MessageContent>,
}

enum MessageContent {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

impl Serialize for MessageContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;

        match self {
            MessageContent::Text { text } => {
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
            }
            MessageContent::ImageUrl { image_url } => {
                map.serialize_entry("type", "image_url")?;
                map.serialize_entry("image_url", image_url)?;
            }
        }

        map.end()
    }
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Deserialize)]
struct ErrorBody {
    message: String,
    #[allow(dead_code)]
    r#type: Option<String>,
    #[allow(dead_code)]
    code: Option<String>,
}

fn extract_text_response(response: ChatCompletionResponse) -> Result<String, AgentError> {
    response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| {
            AgentError::ExecutionFailed("OpenAI API returned no content in the response".into())
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
    fn test_openai_api_agent_creation() {
        let agent = OpenAIApiAgent::new("test-key", "gpt-4o");
        assert_eq!(agent.model, "gpt-4o");
        assert!(agent.max_tokens.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let agent = OpenAIApiAgent::new("test-key", "gpt-4o")
            .with_model("gpt-4o-mini")
            .with_max_tokens(4096);

        assert_eq!(agent.model, "gpt-4o-mini");
        assert_eq!(agent.max_tokens, Some(4096));
    }

    #[test]
    fn test_request_serialization() {
        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: vec![MessageContent::Text {
                    text: "Hello".to_string(),
                }],
            }],
            max_tokens: Some(1000),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gpt-4o\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"max_tokens\":1000"));
    }

    #[test]
    fn test_request_serialization_with_image() {
        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: vec![
                    MessageContent::Text {
                        text: "What's in this image?".to_string(),
                    },
                    MessageContent::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.png".to_string(),
                        },
                    },
                ],
            }],
            max_tokens: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"type\":\"image_url\""));
        assert!(json.contains("https://example.com/image.png"));
    }

    #[test]
    fn test_response_parsing() {
        let json = r#"{
            "choices": [{
                "message": {
                    "content": "Hello, world!"
                }
            }]
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        let text = extract_text_response(response).unwrap();
        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_response_parsing_empty_choices() {
        let json = r#"{"choices": []}"#;
        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        let result = extract_text_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parsing() {
        let json = r#"{
            "error": {
                "message": "Invalid API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        }"#;

        let error = map_http_error(StatusCode::UNAUTHORIZED, json.to_string(), None);
        match error {
            AgentError::ProcessError { message, .. } => {
                assert!(message.contains("Invalid API key provided"));
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
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = OpenAIApiAgent::try_from_env();
        assert!(result.is_err());
    }
}
