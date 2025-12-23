//! GeminiApiAgent - Direct REST API implementation for Gemini.
//!
//! This agent calls the Gemini REST API directly without CLI dependency.
//! API key can be provided directly or loaded from environment variables.
//!
//! # Gemini 3 Support
//!
//! This implementation supports Gemini 3 with thinking capabilities and Google Search.
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_toolkit::agent::impls::GeminiApiAgent;
//! use llm_toolkit::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // From environment variable (GEMINI_API_KEY)
//! let agent = GeminiApiAgent::try_from_env()?;
//! let response = agent.execute("Hello, world!".into()).await?;
//!
//! // Direct API key
//! let agent = GeminiApiAgent::new("your-api-key", "gemini-2.5-flash");
//!
//! // Gemini 3 with thinking capabilities
//! let agent_3 = GeminiApiAgent::new("your-api-key", "gemini-3-pro-preview")
//!     .with_thinking_level("HIGH")
//!     .with_google_search(true);
//! # Ok(())
//! # }
//! ```

use crate::agent::{Agent, AgentError, Payload};
use crate::attachment::Attachment;
use crate::models::GeminiModel;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use reqwest::{Client, StatusCode, header::HeaderValue};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;
const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Agent implementation that talks to the Gemini HTTP API.
#[derive(Clone)]
pub struct GeminiApiAgent {
    client: Client,
    api_key: String,
    model: String,
    system_instruction: Option<String>,
    thinking_level: Option<String>,
    enable_google_search: bool,
}

impl GeminiApiAgent {
    /// Creates a new agent with the provided API key and model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            system_instruction: None,
            thinking_level: None,
            enable_google_search: false,
        }
    }

    /// Loads configuration from environment variables.
    ///
    /// Environment variables:
    /// - `GEMINI_API_KEY` (required)
    /// - `GEMINI_MODEL` (optional, defaults to Gemini 2.5 Flash)
    pub fn try_from_env() -> Result<Self, AgentError> {
        let api_key = env::var("GEMINI_API_KEY").map_err(|_| {
            AgentError::ExecutionFailed(
                "GEMINI_API_KEY environment variable not set".to_string(),
            )
        })?;

        let model = env::var("GEMINI_MODEL")
            .map(|s| s.parse::<GeminiModel>().unwrap_or_default().as_api_id().to_string())
            .unwrap_or_else(|_| GeminiModel::default().as_api_id().to_string());

        Ok(Self::new(api_key, model))
    }

    /// Creates a Gemini 3 agent with thinking capabilities from environment.
    ///
    /// This is a convenience method that:
    /// - Loads API key from GEMINI_API_KEY
    /// - Sets model to gemini-3-pro-preview
    /// - Enables HIGH thinking level
    /// - Optionally enables Google Search tool
    pub fn try_gemini_3_from_env(enable_search: bool) -> Result<Self, AgentError> {
        let api_key = env::var("GEMINI_API_KEY").map_err(|_| {
            AgentError::ExecutionFailed(
                "GEMINI_API_KEY environment variable not set".to_string(),
            )
        })?;

        let mut agent = Self::new(api_key, "gemini-3-pro-preview").with_thinking_level("HIGH");

        if enable_search {
            agent = agent.with_google_search(true);
        }

        Ok(agent)
    }

    /// Overrides the model after construction using a string.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the model using a typed [`GeminiModel`].
    pub fn with_gemini_model(mut self, model: GeminiModel) -> Self {
        self.model = model.as_api_id().to_string();
        self
    }

    /// Adds a system instruction that will be sent alongside every request.
    pub fn with_system_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = Some(instruction.into());
        self
    }

    /// Sets the thinking level for Gemini 3+ models.
    /// Valid values: "LOW", "MEDIUM", "HIGH"
    pub fn with_thinking_level(mut self, level: impl Into<String>) -> Self {
        self.thinking_level = Some(level.into());
        self
    }

    /// Enables Google Search tool for the agent.
    pub fn with_google_search(mut self, enable: bool) -> Self {
        self.enable_google_search = enable;
        self
    }

    async fn build_parts(&self, payload: &Payload) -> Result<Vec<Part>, AgentError> {
        let mut parts = Vec::new();
        let text = payload.to_text();
        if !text.trim().is_empty() {
            parts.push(Part::Text { text });
        }

        for attachment in payload.attachments() {
            if let Some(part) = Self::attachment_to_part(attachment).await? {
                parts.push(part);
            }
        }

        if parts.is_empty() {
            return Err(AgentError::ExecutionFailed(
                "Gemini payload must include text or supported attachments".into(),
            ));
        }

        Ok(parts)
    }

    async fn attachment_to_part(attachment: &Attachment) -> Result<Option<Part>, AgentError> {
        if let Attachment::Remote(_) = attachment {
            return Err(AgentError::ExecutionFailed(
                "Remote attachments are not supported for Gemini API".into(),
            ));
        }

        let bytes = attachment.load_bytes().await.map_err(|err| {
            AgentError::ExecutionFailed(format!("Failed to load attachment for Gemini API: {err}"))
        })?;

        let mime_type = attachment
            .mime_type()
            .unwrap_or_else(|| "application/octet-stream".to_string());

        let data = BASE64_STANDARD.encode(bytes);
        Ok(Some(Part::InlineData {
            inline_data: InlineDataPayload { mime_type, data },
        }))
    }

    async fn send_request(&self, body: &GenerateContentRequest) -> Result<String, AgentError> {
        let url = format!(
            "{}/{model}:generateContent?key={api_key}",
            BASE_URL,
            model = self.model,
            api_key = self.api_key
        );

        let response = self
            .client
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|err| AgentError::ProcessError {
                status_code: None,
                message: format!("Gemini API request failed: {err}"),
                is_retryable: err.is_connect() || err.is_timeout(),
                retry_after: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers().get("retry-after"));
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read Gemini error body".to_string());
            return Err(map_http_error(status, body_text, retry_after));
        }

        let body_text = response.text().await.map_err(|err| {
            AgentError::Other(format!("Failed to read Gemini response body: {err}"))
        })?;

        let parsed: GenerateContentResponse = serde_json::from_str(&body_text).map_err(|err| {
            let truncated_body = if body_text.len() > 500 {
                format!(
                    "{}... (truncated, total {} bytes)",
                    &body_text[..500],
                    body_text.len()
                )
            } else {
                body_text.clone()
            };
            AgentError::Other(format!(
                "Failed to parse Gemini response: {err}\n\nResponse body:\n{truncated_body}"
            ))
        })?;

        extract_text_response(parsed)
    }
}

#[async_trait]
impl Agent for GeminiApiAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"Gemini API agent for multimodal reasoning"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let contents = vec![Content {
            role: "user".to_string(),
            parts: self.build_parts(&payload).await?,
        }];

        let system_instruction = self.system_instruction.as_ref().map(|text| Content {
            role: "system".to_string(),
            parts: vec![Part::Text {
                text: text.to_string(),
            }],
        });

        let generation_config = self.thinking_level.as_ref().map(|level| GenerationConfig {
            thinking_config: ThinkingConfig {
                thinking_level: level.to_string(),
            },
        });

        let tools = if self.enable_google_search {
            Some(vec![Tool::GoogleSearch(GoogleSearchTool {})])
        } else {
            None
        };

        let request = GenerateContentRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
        };
        self.send_request(&request).await
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    thinking_config: ThinkingConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ThinkingConfig {
    thinking_level: String,
}

#[derive(Serialize)]
enum Tool {
    #[serde(rename = "googleSearch")]
    GoogleSearch(GoogleSearchTool),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleSearchTool {}

#[derive(Serialize)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineDataPayload,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineDataPayload {
    mime_type: String,
    data: String,
}

#[derive(Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<ContentResponse>,
}

#[derive(Deserialize)]
struct ContentResponse {
    parts: Vec<PartResponse>,
}

#[derive(Deserialize)]
struct PartResponse {
    text: Option<String>,
}

#[derive(Deserialize)]
struct ErrorWrapper {
    error: ErrorBody,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[allow(dead_code)]
    code: Option<i32>,
    message: Option<String>,
    status: Option<String>,
}

fn extract_text_response(response: GenerateContentResponse) -> Result<String, AgentError> {
    response
        .candidates
        .and_then(|mut candidates| candidates.pop())
        .and_then(|candidate| candidate.content)
        .and_then(|content| content.parts.into_iter().find_map(|part| part.text))
        .ok_or_else(|| {
            AgentError::ExecutionFailed(
                "Gemini API returned no text in the response candidates".into(),
            )
        })
}

fn map_http_error(status: StatusCode, body: String, retry_after: Option<Duration>) -> AgentError {
    let message = serde_json::from_str::<ErrorWrapper>(&body)
        .map(|wrapper| {
            let status_text = wrapper.error.status.unwrap_or_default();
            let msg = wrapper.error.message.unwrap_or_else(|| body.clone());
            if status_text.is_empty() {
                msg
            } else {
                format!("{status_text}: {msg}")
            }
        })
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
    fn test_gemini_api_agent_creation() {
        let agent = GeminiApiAgent::new("test-key", "gemini-2.5-flash");
        assert_eq!(agent.model, "gemini-2.5-flash");
        assert!(agent.system_instruction.is_none());
        assert!(agent.thinking_level.is_none());
        assert!(!agent.enable_google_search);
    }

    #[test]
    fn test_builder_methods() {
        let agent = GeminiApiAgent::new("test-key", "gemini-2.5-flash")
            .with_model("gemini-3-pro-preview")
            .with_system_instruction("You are a helpful assistant")
            .with_thinking_level("HIGH")
            .with_google_search(true);

        assert_eq!(agent.model, "gemini-3-pro-preview");
        assert_eq!(
            agent.system_instruction,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(agent.thinking_level, Some("HIGH".to_string()));
        assert!(agent.enable_google_search);
    }

    #[test]
    fn test_request_serialization_basic() {
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: "user".to_string(),
                parts: vec![Part::Text {
                    text: "Hello".to_string(),
                }],
            }],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_request_serialization_with_thinking() {
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: "user".to_string(),
                parts: vec![Part::Text {
                    text: "Solve this".to_string(),
                }],
            }],
            system_instruction: None,
            generation_config: Some(GenerationConfig {
                thinking_config: ThinkingConfig {
                    thinking_level: "HIGH".to_string(),
                },
            }),
            tools: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"thinkingLevel\":\"HIGH\""));
    }

    #[test]
    fn test_request_serialization_with_google_search() {
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: "user".to_string(),
                parts: vec![Part::Text {
                    text: "Search for news".to_string(),
                }],
            }],
            system_instruction: None,
            generation_config: None,
            tools: Some(vec![Tool::GoogleSearch(GoogleSearchTool {})]),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("googleSearch"));
    }

    #[test]
    fn test_response_parsing() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello, world!"}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let text = extract_text_response(response).unwrap();
        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_response_parsing_empty_candidates() {
        let json = r#"{"candidates": []}"#;
        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let result = extract_text_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parsing() {
        let json = r#"{
            "error": {
                "code": 400,
                "message": "Invalid API key",
                "status": "INVALID_ARGUMENT"
            }
        }"#;

        let error = map_http_error(StatusCode::BAD_REQUEST, json.to_string(), None);
        match error {
            AgentError::ProcessError { message, .. } => {
                assert!(message.contains("INVALID_ARGUMENT"));
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
        // Ensure GEMINI_API_KEY is not set for this test
        // SAFETY: This test runs single-threaded (--test-threads=1)
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
        let result = GeminiApiAgent::try_from_env();
        assert!(result.is_err());
    }
}
