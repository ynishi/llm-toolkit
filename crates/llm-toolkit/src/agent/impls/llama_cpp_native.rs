//! LlamaCppNativeAgent - Native llama.cpp binding for local LLM inference.
//!
//! This module provides a native llama.cpp integration using the `llama-cpp-2` crate,
//! allowing for local LLM inference without HTTP overhead.
//!
//! # Features
//!
//! - **Native Inference**: No HTTP overhead, runs in-process
//! - **HuggingFace Integration**: Automatic model download from HuggingFace Hub
//! - **GPU Acceleration**: Optional CUDA/Metal support via feature flags
//! - **Model Presets**: Pre-configured settings for popular models
//!
//! # Prerequisites
//!
//! 1. Enable the feature: `llm-toolkit = { features = ["llama-cpp-native"] }`
//! 2. For GPU: `features = ["llama-cpp-native", "metal"]` (macOS) or `"cuda"` (NVIDIA)
//!
//! # Example
//!
//! ```ignore
//! use llm_toolkit::agent::impls::{LlamaCppNativeAgent, LlamaCppNativeConfig};
//! use llm_toolkit::agent::Agent;
//!
//! // Use a preset model (auto-downloads from HuggingFace)
//! let agent = LlamaCppNativeAgent::try_new(LlamaCppNativeConfig::qwen_0_5b())?;
//!
//! // Or with GPU acceleration
//! let agent = LlamaCppNativeAgent::try_new(
//!     LlamaCppNativeConfig::lfm2_1b().with_gpu_layers(32)
//! )?;
//!
//! let response = agent.execute("Hello!".into()).await?;
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use async_trait::async_trait;

use crate::agent::{Agent, AgentError, Payload};

/// Chat template for formatting prompts.
///
/// Different models require different prompt formats for optimal results.
#[derive(Debug, Clone, Default)]
pub enum NativeChatTemplate {
    /// Llama 3 format: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|eot_id|>`
    #[default]
    Llama3,
    /// Qwen/Qwen2/Qwen2.5 format: `<|im_start|>user\n...<|im_end|>`
    Qwen,
    /// LiquidAI LFM2/LFM2.5 format: `<|user|>\n...\n<|assistant|>`
    Lfm2,
    /// Mistral/Mixtral format: `[INST] ... [/INST]`
    Mistral,
    /// Generic ChatML format: `<|im_start|>user\n...<|im_end|>`
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

impl NativeChatTemplate {
    /// Format a user prompt with this template.
    pub fn format(&self, prompt: &str) -> String {
        match self {
            Self::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                prompt
            ),
            Self::Qwen | Self::ChatMl => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            ),
            Self::Lfm2 => format!("<|user|>\n{}\n<|assistant|>\n", prompt),
            Self::Mistral => format!("[INST] {} [/INST]", prompt),
            Self::None => prompt.to_string(),
            Self::Custom {
                user_prefix,
                user_suffix,
                assistant_prefix,
            } => format!(
                "{}{}{}{}",
                user_prefix, prompt, user_suffix, assistant_prefix
            ),
        }
    }

    /// Format with system prompt.
    pub fn format_with_system(&self, system: &str, prompt: &str) -> String {
        match self {
            Self::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                system, prompt
            ),
            Self::Qwen | Self::ChatMl => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system, prompt
            ),
            Self::Lfm2 => format!(
                "<|system|>\n{}\n<|user|>\n{}\n<|assistant|>\n",
                system, prompt
            ),
            Self::Mistral => format!("[INST] {} {} [/INST]", system, prompt),
            Self::None => format!("{}\n\n{}", system, prompt),
            Self::Custom {
                user_prefix,
                user_suffix,
                assistant_prefix,
            } => format!(
                "{}{}{}{}",
                user_prefix, prompt, user_suffix, assistant_prefix
            ),
        }
    }
}

/// Configuration for LlamaCppNativeAgent.
#[derive(Debug, Clone)]
pub struct LlamaCppNativeConfig {
    /// Model path (HuggingFace repo ID or local path)
    pub model_path: String,
    /// GGUF filename (for HuggingFace downloads)
    pub gguf_file: Option<String>,
    /// Context size (default: 4096)
    pub n_ctx: u32,
    /// Batch size (default: 512)
    pub n_batch: u32,
    /// GPU layers (0 = CPU only)
    pub n_gpu_layers: u32,
    /// Maximum generation tokens (default: 256)
    pub max_tokens: usize,
    /// Temperature (default: 0.7)
    pub temperature: f32,
    /// Top-p (default: 0.9)
    pub top_p: f32,
    /// Thread count (None = auto)
    pub n_threads: Option<u32>,
    /// Chat template
    pub chat_template: NativeChatTemplate,
    /// System prompt
    pub system_prompt: Option<String>,
}

impl Default for LlamaCppNativeConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            gguf_file: None,
            n_ctx: 4096,
            n_batch: 512,
            n_gpu_layers: 0,
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            n_threads: None,
            chat_template: NativeChatTemplate::default(),
            system_prompt: None,
        }
    }
}

impl LlamaCppNativeConfig {
    /// Create config from HuggingFace repo.
    pub fn from_hf(repo_id: impl Into<String>, gguf_file: impl Into<String>) -> Self {
        Self {
            model_path: repo_id.into(),
            gguf_file: Some(gguf_file.into()),
            ..Default::default()
        }
    }

    /// Create config from local file.
    pub fn from_local(path: impl Into<String>) -> Self {
        Self {
            model_path: path.into(),
            gguf_file: None,
            ..Default::default()
        }
    }

    /// Set GPU layers.
    pub fn with_gpu_layers(mut self, n_layers: u32) -> Self {
        self.n_gpu_layers = n_layers;
        self
    }

    /// Set context size.
    pub fn with_context_size(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set chat template.
    pub fn with_chat_template(mut self, template: NativeChatTemplate) -> Self {
        self.chat_template = template;
        self
    }

    /// Set system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    // =========================================================================
    // Model Presets
    // =========================================================================

    /// LiquidAI LFM2.5 1.2B (Q4_K_M) - Lightweight, fast
    pub fn lfm2_1b() -> Self {
        Self::from_hf(
            "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
            "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        )
        .with_chat_template(NativeChatTemplate::Lfm2)
    }

    /// LiquidAI LFM2.5 1.2B (Q8_0) - Higher quality
    pub fn lfm2_1b_q8() -> Self {
        Self::from_hf(
            "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
            "LFM2.5-1.2B-Instruct-Q8_0.gguf",
        )
        .with_chat_template(NativeChatTemplate::Lfm2)
    }

    /// Qwen 2.5 0.5B (Q4_K_M) - Ultra lightweight
    pub fn qwen_0_5b() -> Self {
        Self::from_hf(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )
        .with_chat_template(NativeChatTemplate::Qwen)
    }

    /// Qwen 2.5 1.5B (Q4_K_M)
    pub fn qwen_1_5b() -> Self {
        Self::from_hf(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        )
        .with_chat_template(NativeChatTemplate::Qwen)
    }

    /// Qwen 2.5 3B (Q4_K_M)
    pub fn qwen_3b() -> Self {
        Self::from_hf(
            "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "qwen2.5-3b-instruct-q4_k_m.gguf",
        )
        .with_chat_template(NativeChatTemplate::Qwen)
    }

    /// Microsoft Phi-3 Mini (Q4_K_M)
    pub fn phi3_mini() -> Self {
        Self::from_hf(
            "microsoft/Phi-3-mini-4k-instruct-gguf",
            "Phi-3-mini-4k-instruct-q4.gguf",
        )
        .with_chat_template(NativeChatTemplate::ChatMl)
    }

    /// Llama 3.2 1B (Q4_K_M)
    pub fn llama3_1b() -> Self {
        Self::from_hf(
            "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
            "llama-3.2-1b-instruct-q4_k_m.gguf",
        )
        .with_chat_template(NativeChatTemplate::Llama3)
    }

    /// Llama 3.2 3B (Q4_K_M)
    pub fn llama3_3b() -> Self {
        Self::from_hf(
            "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
            "llama-3.2-3b-instruct-q4_k_m.gguf",
        )
        .with_chat_template(NativeChatTemplate::Llama3)
    }

    /// Get display name.
    pub fn display_name(&self) -> &str {
        &self.model_path
    }
}

/// Internal state (shared across threads).
///
/// # Self-referential Structure
///
/// `context` references `model` and `backend`. Field declaration order ensures
/// proper drop order: `context` -> `model` -> `backend`.
struct LlamaCppInner {
    // Note: Field order matters. context references model/backend, so it must
    // be declared first to ensure it's dropped first.
    context: Mutex<Option<LlamaContext<'static>>>,
    #[allow(dead_code)]
    backend: LlamaBackend,
    model: LlamaModel,
    config: LlamaCppNativeConfig,
}

// SAFETY: context is protected by Mutex, and LlamaCppInner is shared via Arc.
// Concurrent access from multiple threads is safe.
unsafe impl Send for LlamaCppInner {}
unsafe impl Sync for LlamaCppInner {}

/// Native llama.cpp agent for local LLM inference.
///
/// Uses `llama-cpp-2` crate for direct llama.cpp integration without HTTP overhead.
///
/// # Example
///
/// ```ignore
/// use llm_toolkit::agent::impls::{LlamaCppNativeAgent, LlamaCppNativeConfig};
/// use llm_toolkit::agent::Agent;
///
/// let agent = LlamaCppNativeAgent::try_new(LlamaCppNativeConfig::qwen_0_5b())?;
/// let response = agent.execute("What is Rust?".into()).await?;
/// println!("{}", response);
/// ```
#[derive(Clone)]
pub struct LlamaCppNativeAgent {
    inner: Arc<LlamaCppInner>,
}

impl LlamaCppNativeAgent {
    /// Create a new agent with the given configuration.
    ///
    /// This will initialize the llama.cpp backend, load the model (downloading
    /// from HuggingFace if necessary), and create the inference context.
    pub fn try_new(config: LlamaCppNativeConfig) -> Result<Self, AgentError> {
        // Initialize backend
        let backend = LlamaBackend::init().map_err(|e| AgentError::ProcessError {
            message: format!("Failed to init llama backend: {}", e),
            is_retryable: false,
            status_code: None,
            retry_after: None,
        })?;

        // Resolve model path
        let model_path = Self::resolve_model_path(&config)?;

        // Model parameters
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);

        // Load model
        let model =
            LlamaModel::load_from_file(&backend, &model_path, &model_params).map_err(|e| {
                AgentError::ProcessError {
                    message: format!("Failed to load model: {}", e),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                }
            })?;

        // Context parameters
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(config.n_ctx))
            .with_n_batch(config.n_batch);

        // Create context
        let context =
            model
                .new_context(&backend, ctx_params)
                .map_err(|e| AgentError::ProcessError {
                    message: format!("Failed to create context: {}", e),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                })?;

        // SAFETY: LlamaContext references backend and model, but LlamaCppInner
        // owns all of them. Field order ensures context is dropped first.
        // The 'static lifetime is a lie, but the context is only used within
        // LlamaCppInner's lifetime.
        let context: LlamaContext<'static> = unsafe { std::mem::transmute(context) };

        let inner = LlamaCppInner {
            context: Mutex::new(Some(context)),
            backend,
            model,
            config,
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.inner.config.display_name()
    }

    /// Check if the agent is healthy (always true for native inference).
    pub fn is_healthy(&self) -> bool {
        true
    }

    /// Resolve model path (HuggingFace or local).
    fn resolve_model_path(config: &LlamaCppNativeConfig) -> Result<PathBuf, AgentError> {
        if let Some(ref gguf_file) = config.gguf_file {
            // HuggingFace Hub download
            let cache_dir = dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("huggingface")
                .join("hub");

            // Build model directory name
            let model_dir_name = format!("models--{}", config.model_path.replace('/', "--"));
            let model_dir = cache_dir.join(&model_dir_name);

            // Check snapshots for cached model
            let snapshots_dir = model_dir.join("snapshots");
            if snapshots_dir.exists()
                && let Ok(entries) = std::fs::read_dir(&snapshots_dir)
            {
                for entry in entries.flatten() {
                    let snapshot_path = entry.path().join(gguf_file);
                    if snapshot_path.exists() {
                        tracing::info!("Using cached model: {:?}", snapshot_path);
                        return Ok(snapshot_path);
                    }
                }
            }

            // Download from HuggingFace
            tracing::info!(
                "Model not in cache, downloading from HuggingFace: {}",
                config.model_path
            );

            let api = hf_hub::api::sync::Api::new().map_err(|e| AgentError::ProcessError {
                message: format!("Failed to create HF API: {}", e),
                is_retryable: false,
                status_code: None,
                retry_after: None,
            })?;

            let repo = api.model(config.model_path.clone());
            let path = repo.get(gguf_file).map_err(|e| AgentError::ProcessError {
                message: format!("Failed to download model: {}", e),
                is_retryable: true,
                status_code: None,
                retry_after: None,
            })?;

            Ok(path)
        } else {
            // Local path
            let path = PathBuf::from(&config.model_path);
            if !path.exists() {
                return Err(AgentError::ProcessError {
                    message: format!("Model file not found: {}", config.model_path),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                });
            }
            Ok(path)
        }
    }

    /// Generate text synchronously.
    fn generate_sync(
        inner: &LlamaCppInner,
        context: &mut LlamaContext,
        prompt: &str,
    ) -> Result<String, AgentError> {
        // Clear context (remove previous KV cache)
        context.clear_kv_cache();

        // Tokenize prompt
        let tokens = inner
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| AgentError::ProcessError {
                message: format!("Tokenization error: {}", e),
                is_retryable: false,
                status_code: None,
                retry_after: None,
            })?;

        tracing::debug!(
            tokens_len = tokens.len(),
            prompt_len = prompt.len(),
            "Tokenized prompt"
        );

        // Create batch
        let mut batch = LlamaBatch::new(inner.config.n_batch as usize, 1);

        // Add tokens to batch
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| AgentError::ProcessError {
                    message: format!("Batch add error: {}", e),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                })?;
        }

        // Decode prompt
        context
            .decode(&mut batch)
            .map_err(|e| AgentError::ProcessError {
                message: format!("Decode error: {}", e),
                is_retryable: false,
                status_code: None,
                retry_after: None,
            })?;

        tracing::debug!("Prompt decoded, starting generation");

        // Configure sampler
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(inner.config.temperature),
            LlamaSampler::top_p(inner.config.top_p, 1),
            LlamaSampler::dist(42),
        ]);

        // Generation loop
        let mut output_tokens = Vec::new();
        let mut n_cur = tokens.len();

        for i in 0..inner.config.max_tokens {
            // Sample
            let new_token = sampler.sample(context, -1);

            // Check EOS
            if inner.model.is_eog_token(new_token) {
                tracing::debug!(iteration = i, "EOS token reached");
                break;
            }

            output_tokens.push(new_token);

            // Debug first few tokens
            if i < 5
                && let Ok(piece) =
                    inner
                        .model
                        .token_to_str_with_size(new_token, 256, Special::Tokenize)
            {
                tracing::trace!(token_idx = i, piece = ?piece, "Generated token");
            }

            // Next batch
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| AgentError::ProcessError {
                    message: format!("Batch add error: {}", e),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                })?;

            n_cur += 1;

            // Decode
            context
                .decode(&mut batch)
                .map_err(|e| AgentError::ProcessError {
                    message: format!("Decode error: {}", e),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                })?;
        }

        // Convert tokens to string
        let mut output = String::new();
        for token in &output_tokens {
            let piece = inner
                .model
                .token_to_str_with_size(*token, 256, Special::Tokenize)
                .map_err(|e| AgentError::ProcessError {
                    message: format!("Detokenization error: {}", e),
                    is_retryable: false,
                    status_code: None,
                    retry_after: None,
                })?;
            output.push_str(&piece);
        }

        Ok(output)
    }

    /// Call LLM (sync -> async wrapper).
    async fn call_llm(&self, prompt: &str) -> Result<String, AgentError> {
        let inner = Arc::clone(&self.inner);

        // Format prompt with chat template
        let formatted_prompt = if let Some(ref system) = inner.config.system_prompt {
            inner
                .config
                .chat_template
                .format_with_system(system, prompt)
        } else {
            inner.config.chat_template.format(prompt)
        };

        tokio::task::spawn_blocking(move || {
            let mut guard = inner.context.lock().unwrap();
            let context = guard.as_mut().expect("Context not initialized");
            Self::generate_sync(&inner, context, &formatted_prompt)
        })
        .await
        .map_err(|e| AgentError::ProcessError {
            message: format!("spawn_blocking failed: {}", e),
            is_retryable: false,
            status_code: None,
            retry_after: None,
        })?
    }
}

#[async_trait]
impl Agent for LlamaCppNativeAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"Native llama.cpp agent for local LLM inference"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let text = payload.to_text();
        if text.trim().is_empty() {
            return Err(AgentError::ExecutionFailed(
                "Payload must include text".into(),
            ));
        }
        self.call_llm(&text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_hf() {
        let config = LlamaCppNativeConfig::from_hf(
            "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
            "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        );
        assert_eq!(config.model_path, "LiquidAI/LFM2.5-1.2B-Instruct-GGUF");
        assert_eq!(
            config.gguf_file,
            Some("LFM2.5-1.2B-Instruct-Q4_K_M.gguf".to_string())
        );
    }

    #[test]
    fn test_config_from_local() {
        let config = LlamaCppNativeConfig::from_local("/path/to/model.gguf");
        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert!(config.gguf_file.is_none());
    }

    #[test]
    fn test_config_presets() {
        let lfm2 = LlamaCppNativeConfig::lfm2_1b();
        assert!(lfm2.model_path.contains("LFM2.5"));
        assert!(lfm2.gguf_file.as_ref().unwrap().contains("Q4_K_M"));

        let qwen = LlamaCppNativeConfig::qwen_0_5b();
        assert!(qwen.model_path.contains("Qwen"));

        let llama = LlamaCppNativeConfig::llama3_1b();
        assert!(llama.model_path.contains("Llama-3.2"));
    }

    #[test]
    fn test_config_builder() {
        let config = LlamaCppNativeConfig::lfm2_1b()
            .with_gpu_layers(32)
            .with_max_tokens(512)
            .with_temperature(0.5)
            .with_top_p(0.95)
            .with_context_size(8192)
            .with_system_prompt("Be helpful");

        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.max_tokens, 512);
        assert!((config.temperature - 0.5).abs() < f32::EPSILON);
        assert!((config.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(config.n_ctx, 8192);
        assert_eq!(config.system_prompt, Some("Be helpful".to_string()));
    }

    #[test]
    fn test_chat_template_format() {
        let prompt = "Hello";

        let llama3 = NativeChatTemplate::Llama3.format(prompt);
        assert!(llama3.contains("<|begin_of_text|>"));
        assert!(llama3.contains(prompt));

        let qwen = NativeChatTemplate::Qwen.format(prompt);
        assert!(qwen.contains("<|im_start|>"));
        assert!(qwen.contains(prompt));

        let lfm2 = NativeChatTemplate::Lfm2.format(prompt);
        assert!(lfm2.contains("<|user|>"));
        assert!(lfm2.contains("<|assistant|>"));

        let mistral = NativeChatTemplate::Mistral.format(prompt);
        assert!(mistral.contains("[INST]"));
        assert!(mistral.contains("[/INST]"));

        let none = NativeChatTemplate::None.format(prompt);
        assert_eq!(none, prompt);
    }

    #[test]
    fn test_chat_template_with_system() {
        let system = "You are helpful";
        let prompt = "Hello";

        let llama3 = NativeChatTemplate::Llama3.format_with_system(system, prompt);
        assert!(llama3.contains("system"));
        assert!(llama3.contains(system));
        assert!(llama3.contains(prompt));

        let qwen = NativeChatTemplate::Qwen.format_with_system(system, prompt);
        assert!(qwen.contains("system"));
        assert!(qwen.contains(system));

        let lfm2 = NativeChatTemplate::Lfm2.format_with_system(system, prompt);
        assert!(lfm2.contains("<|system|>"));
        assert!(lfm2.contains(system));
    }
}
