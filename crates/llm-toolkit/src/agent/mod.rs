//! Agent API for building multi-agent LLM systems.
//!
//! This module provides the core abstractions for defining and executing agents
//! in a swarm-based architecture. Agents are defined by their expertise and output type,
//! allowing for flexible composition and orchestration.
//!
//! # Design Philosophy
//!
//! The Agent API follows the principle of **capability and intent separation**:
//!
//! - **Capability**: An agent declares what it can do (`expertise`) and what it produces (`Output`)
//! - **Intent**: The orchestrator provides what needs to be done as a natural language string
//!
//! This separation enables maximum reusability and flexibility.
//!
//! # Use Cases
//!
//! ## 1. Simple Structured Output
//!
//! Use `#[derive(Agent)]` when you just need LLM → JSON parsing without additional logic:
//!
//! ```rust,ignore
//! use llm_toolkit::Agent;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct ArticleSummary {
//!     title: String,
//!     key_points: Vec<String>,
//! }
//!
//! #[derive(Agent)]
//! #[agent(
//!     expertise = "Summarizing articles and extracting key information",
//!     output = "ArticleSummary"
//! )]
//! struct ArticleSummarizerAgent;
//!
//! // Usage
//! let agent = ArticleSummarizerAgent;
//! let result: ArticleSummary = agent.execute("Summarize this article...".to_string()).await?;
//! ```
//!
//! **When to use:** Simple agents where the LLM generates JSON and you just need it parsed.
//!
//! ## 2. Post-Processing (Save, Validate, Transform)
//!
//! Manually implement `Agent` trait when you need custom logic after the LLM call.
//! This is the recommended pattern for production use cases.
//!
//! ```rust,ignore
//! use llm_toolkit::agent::{Agent, AgentError, impls::ClaudeCodeAgent};
//! use serde::{Serialize, Deserialize};
//! use std::path::PathBuf;
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct ArticleData {
//!     title: String,
//!     content: String,
//! }
//!
//! /// An agent that generates articles and saves them to disk
//! pub struct ArticleGeneratorAgent {
//!     inner: ClaudeCodeAgent,  // Reuse existing LLM agent
//!     save_dir: PathBuf,       // Custom state for post-processing
//! }
//!
//! impl ArticleGeneratorAgent {
//!     pub fn new(save_dir: PathBuf) -> Self {
//!         Self {
//!             inner: ClaudeCodeAgent::new(),
//!             save_dir,
//!         }
//!     }
//! }
//!
//! #[async_trait::async_trait]
//! impl Agent for ArticleGeneratorAgent {
//!     type Output = ArticleData;
//!
//!     fn expertise(&self) -> &str {
//!         "Generate articles and save them to disk"
//!     }
//!
//!     async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
//!         // 1. Call LLM
//!         let raw_output = self.inner.execute(intent).await?;
//!
//!         // 2. Parse JSON
//!         let json_str = crate::extract_json(&raw_output)
//!             .map_err(|e| AgentError::ParseError(e.to_string()))?;
//!         let data: ArticleData = serde_json::from_str(&json_str)
//!             .map_err(|e| AgentError::ParseError(e.to_string()))?;
//!
//!         // 3. Post-processing: Save to disk
//!         let filename = format!("{}.json", data.title.replace(" ", "_"));
//!         let path = self.save_dir.join(filename);
//!         tokio::fs::write(&path, &json_str).await
//!             .map_err(|e| AgentError::Other(format!("Failed to save: {}", e)))?;
//!
//!         // 4. Return typed output
//!         Ok(data)
//!     }
//! }
//! ```
//!
//! **When to use:**
//! - Saving outputs to database/filesystem
//! - Validating LLM output before returning
//! - Transforming data formats
//! - Chaining multiple operations
//! - Any custom business logic
//!
//! **Pattern benefits:**
//! - ✅ Type-safe: Output type is preserved
//! - ✅ Composable: Reuse existing agents like `ClaudeCodeAgent`
//! - ✅ Flexible: Full control over execution flow
//! - ✅ Testable: Easy to mock inner agent
//!
//! ## 3. Multi-Step Processing
//!
//! Build agents that perform multiple LLM calls or validation steps:
//!
//! ```rust,ignore
//! pub struct ValidatedArticleAgent {
//!     generator: ClaudeCodeAgent,
//!     validator: ClaudeCodeAgent,
//! }
//!
//! #[async_trait::async_trait]
//! impl Agent for ValidatedArticleAgent {
//!     type Output = ArticleData;
//!
//!     async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
//!         // Step 1: Generate article
//!         let article = self.generator.execute(intent).await?;
//!         let data: ArticleData = serde_json::from_str(&article)?;
//!
//!         // Step 2: Validate quality
//!         let validation_prompt = format!(
//!             "Validate this article meets quality standards:\n{}",
//!             serde_json::to_string_pretty(&data)?
//!         );
//!         let validation = self.validator.execute(validation_prompt).await?;
//!
//!         // Step 3: Parse validation result and handle failures
//!         if !validation.contains("PASS") {
//!             return Err(AgentError::ExecutionFailed(
//!                 format!("Validation failed: {}", validation)
//!             ));
//!         }
//!
//!         Ok(data)
//!     }
//! }
//! ```
//!
//! ## 4. Custom Backend Integration
//!
//! Wrap other LLM providers or APIs:
//!
//! ```rust,ignore
//! pub struct CustomLLMAgent {
//!     api_key: String,
//!     endpoint: String,
//! }
//!
//! #[async_trait::async_trait]
//! impl Agent for CustomLLMAgent {
//!     type Output = String;
//!
//!     fn expertise(&self) -> &str {
//!         "Custom LLM provider integration"
//!     }
//!
//!     async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
//!         // Your custom HTTP client logic here
//!         let client = reqwest::Client::new();
//!         let response = client.post(&self.endpoint)
//!             .header("Authorization", format!("Bearer {}", self.api_key))
//!             .json(&serde_json::json!({ "prompt": intent }))
//!             .send()
//!             .await
//!             .map_err(|e| AgentError::ProcessError(e.to_string()))?;
//!
//!         response.text().await
//!             .map_err(|e| AgentError::Other(e.to_string()))
//!     }
//! }
//! ```
//!
//! # Implementation Patterns
//!
//! The crate provides `ClaudeCodeJsonAgent` as a reference implementation of the post-processing
//! pattern. It composes `ClaudeCodeAgent` and adds JSON extraction/parsing:
//!
//! ```rust,ignore
//! // From llm_toolkit::agent::impls::claude_code
//! pub struct ClaudeCodeJsonAgent<T> {
//!     inner: ClaudeCodeAgent,
//!     _phantom: PhantomData<T>,
//! }
//!
//! #[async_trait]
//! impl<T: Serialize + DeserializeOwned> Agent for ClaudeCodeJsonAgent<T> {
//!     async fn execute(&self, intent: String) -> Result<T, AgentError> {
//!         let raw = self.inner.execute(intent).await?;
//!         let json = extract_json(&raw)?;
//!         serde_json::from_str(&json)
//!     }
//! }
//! ```
//!
//! This pattern is recommended for building your own custom agents.

pub mod error;
pub mod payload;

#[cfg(feature = "agent")]
pub mod impls;

#[cfg(feature = "agent")]
pub mod retry;

/// Defines the execution profile for an agent, controlling its behavior.
///
/// This enum provides a semantic way to configure agents for different tasks
/// without exposing model-specific parameters like temperature directly.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionProfile {
    /// For tasks requiring creativity and diverse outputs.
    Creative,
    /// A balanced profile for general use cases.
    #[default]
    Balanced,
    /// For tasks requiring precision, consistency, and predictable outputs.
    Deterministic,
}

pub use error::AgentError;
pub use payload::{Payload, PayloadContent};

use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};

/// The core trait for defining an agent.
///
/// An agent represents a reusable capability that can execute tasks based on
/// natural language intents. The agent's expertise and output type are statically
/// defined, while the specific task is provided dynamically at runtime.
#[async_trait]
pub trait Agent: Send + Sync {
    /// The type of output this agent produces.
    ///
    /// This type must be serializable and deserializable to enable
    /// communication between agents and persistence of results.
    type Output: Serialize + DeserializeOwned;

    /// Returns a natural language description of what this agent can do.
    ///
    /// This description is used by orchestrators to select the most appropriate
    /// agent for a given task. It should be clear, concise, and descriptive.
    ///
    /// # Example
    ///
    /// ```ignore
    /// "Analyze web content and extract structured information about technical topics"
    /// ```
    fn expertise(&self) -> &str;

    /// Execute the agent with a specific intent.
    ///
    /// The intent is a natural language description of what needs to be accomplished.
    /// The orchestrator is responsible for collecting all necessary context and
    /// formatting it into a high-quality prompt.
    ///
    /// # Arguments
    ///
    /// * `intent` - A payload containing the task to perform. Can be text, images,
    ///              or a combination. Use `.into()` to convert from String for backward
    ///              compatibility.
    ///
    /// # Returns
    ///
    /// A `Result` containing the structured output on success, or an `AgentError` on failure.
    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError>;

    /// Returns the name of this agent.
    ///
    /// By default, this returns the type name. Can be overridden for custom naming.
    fn name(&self) -> String {
        std::any::type_name::<Self>()
            .split("::")
            .last()
            .unwrap_or("UnknownAgent")
            .to_string()
    }

    /// Checks if the agent's backend is available and ready to use.
    ///
    /// This is useful for verifying that required CLI tools or APIs are accessible
    /// before attempting to execute tasks.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the agent is available and ready, or an `AgentError` describing
    /// the issue if the agent cannot be used.
    ///
    /// # Default Implementation
    ///
    /// By default, this always returns `Ok(())`. Implementations should override
    /// this to check for CLI availability, API keys, or other prerequisites.
    async fn is_available(&self) -> Result<(), AgentError> {
        Ok(())
    }
}

/// A boxed agent trait object for dynamic dispatch.
///
/// This type alias is useful for storing heterogeneous collections of agents.
pub type BoxedAgent<T> = Box<dyn Agent<Output = T>>;

/// Dynamic agent trait for type-erased agent execution.
///
/// This trait allows the orchestrator to work with agents of different output types
/// by converting all outputs to `serde_json::Value`. This enables heterogeneous
/// agent collections while maintaining type safety at the agent implementation level.
#[async_trait]
pub trait DynamicAgent: Send + Sync {
    /// Execute the agent and return the output as a JSON value.
    async fn execute_dynamic(&self, intent: Payload) -> Result<serde_json::Value, AgentError>;

    /// Returns the name of this agent.
    fn name(&self) -> String;

    /// Returns a natural language description of what this agent can do.
    fn expertise(&self) -> &str;

    /// Checks if the agent's backend is available.
    async fn is_available(&self) -> Result<(), AgentError> {
        Ok(())
    }

    /// Attempts to convert a JSON output to a prompt string if the underlying type implements ToPrompt.
    ///
    /// Returns `Some(String)` if the output type implements ToPrompt, `None` otherwise.
    /// This allows the orchestrator to use rich prompt representations when available.
    fn try_to_prompt(&self, _json: &serde_json::Value) -> Option<String> {
        None
    }
}

/// Type alias for the ToPrompt conversion function.
type ToPromptFn = Box<dyn Fn(&serde_json::Value) -> Option<String> + Send + Sync>;

/// Adapter that wraps any `Agent<Output = T>` to implement `DynamicAgent`.
///
/// This adapter performs type erasure by converting the agent's structured output
/// into `serde_json::Value`, allowing agents with different output types to be
/// stored in the same collection.
pub struct AgentAdapter<T: Serialize + DeserializeOwned> {
    inner: Box<dyn Agent<Output = T>>,
    try_to_prompt_fn: Option<ToPromptFn>,
}

impl<T: Serialize + DeserializeOwned> AgentAdapter<T> {
    /// Creates a new adapter wrapping the given agent.
    pub fn new(agent: impl Agent<Output = T> + 'static) -> Self {
        Self {
            inner: Box::new(agent),
            try_to_prompt_fn: None,
        }
    }

    /// Creates a new adapter with ToPrompt support.
    ///
    /// This constructor should be used when T implements ToPrompt, allowing the
    /// orchestrator to use rich prompt representations instead of plain JSON.
    pub fn with_to_prompt(
        agent: impl Agent<Output = T> + 'static,
        to_prompt_fn: impl Fn(&T) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Box::new(agent),
            try_to_prompt_fn: Some(Box::new(move |json| {
                serde_json::from_value::<T>(json.clone())
                    .ok()
                    .map(|output| to_prompt_fn(&output))
            })),
        }
    }
}

#[async_trait]
impl<T: Serialize + DeserializeOwned> DynamicAgent for AgentAdapter<T> {
    async fn execute_dynamic(&self, intent: Payload) -> Result<serde_json::Value, AgentError> {
        let output = self.inner.execute(intent).await?;
        serde_json::to_value(output).map_err(|e| AgentError::SerializationFailed(e.to_string()))
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    fn expertise(&self) -> &str {
        self.inner.expertise()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        self.inner.is_available().await
    }

    fn try_to_prompt(&self, json: &serde_json::Value) -> Option<String> {
        self.try_to_prompt_fn.as_ref().and_then(|f| f(json))
    }
}
