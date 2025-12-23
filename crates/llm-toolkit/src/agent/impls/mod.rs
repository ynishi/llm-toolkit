//! Built-in agent implementations.

pub mod claude_code;
pub mod cli_agent;
mod cli_attachment;
pub mod codex_agent;
pub mod gemini;
pub mod inner_validator;
pub mod retry;

// API client implementations (direct HTTP API calls)
#[cfg(feature = "gemini-api")]
pub mod gemini_api;
#[cfg(feature = "openai-api")]
pub mod openai_api;
#[cfg(feature = "anthropic-api")]
pub mod anthropic_api;

pub use claude_code::{ClaudeCodeAgent, ClaudeCodeJsonAgent, ClaudeModel};
pub use codex_agent::{CodexAgent, CodexModel};
pub use gemini::{GeminiAgent, GeminiModel};
pub use inner_validator::InnerValidatorAgent;
pub use retry::RetryAgent;

// API client exports
#[cfg(feature = "gemini-api")]
pub use gemini_api::GeminiApiAgent;
#[cfg(feature = "openai-api")]
pub use openai_api::OpenAIApiAgent;
#[cfg(feature = "anthropic-api")]
pub use anthropic_api::AnthropicApiAgent;
