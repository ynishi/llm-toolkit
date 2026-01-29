//! Built-in agent implementations.

pub mod claude_code;
pub mod cli_agent;
mod cli_attachment;
pub mod codex_agent;
pub mod gemini;
pub mod inner_validator;
pub mod retry;

// API client implementations (direct HTTP API calls)
#[cfg(feature = "anthropic-api")]
pub mod anthropic_api;
#[cfg(feature = "gemini-api")]
pub mod gemini_api;
#[cfg(feature = "llama-cpp-server")]
pub mod llama_cpp_server;
#[cfg(feature = "ollama-api")]
pub mod ollama_api;
#[cfg(feature = "openai-api")]
pub mod openai_api;

pub use claude_code::{ClaudeCodeAgent, ClaudeCodeJsonAgent};
pub use codex_agent::CodexAgent;
pub use gemini::GeminiAgent;
pub use inner_validator::InnerValidatorAgent;
pub use retry::RetryAgent;

// Re-export model types from the models module for backward compatibility
pub use crate::models::{ClaudeModel, GeminiModel, OpenAIModel};

// Deprecated type alias for backward compatibility
#[allow(deprecated)]
pub use codex_agent::CodexModel;

// API client exports
#[cfg(feature = "anthropic-api")]
pub use anthropic_api::AnthropicApiAgent;
#[cfg(feature = "gemini-api")]
pub use gemini_api::GeminiApiAgent;
#[cfg(feature = "llama-cpp-server")]
pub use llama_cpp_server::{ChatTemplate, LlamaCppServerAgent, LlamaCppServerConfig};
#[cfg(feature = "ollama-api")]
pub use ollama_api::OllamaApiAgent;
#[cfg(feature = "openai-api")]
pub use openai_api::OpenAIApiAgent;
