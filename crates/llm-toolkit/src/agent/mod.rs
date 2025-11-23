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

pub mod capability;
pub mod error;
pub mod payload;

#[cfg(feature = "agent")]
pub mod impls;

#[cfg(feature = "agent")]
pub mod retry;

#[cfg(feature = "agent")]
pub mod persona;

#[cfg(feature = "agent")]
pub mod history;

#[cfg(feature = "agent")]
pub mod retrieval;

#[cfg(feature = "agent")]
pub mod payload_message;

#[cfg(feature = "agent")]
pub mod chat;

#[cfg(feature = "agent")]
pub mod dialogue;

#[cfg(feature = "agent")]
pub mod expertise;

#[cfg(feature = "agent")]
pub mod expertise_agent;

#[cfg(feature = "agent")]
pub mod env_context;

#[cfg(feature = "agent")]
pub mod detected_context;

#[cfg(feature = "agent")]
pub mod execution_context;

#[cfg(feature = "agent")]
pub mod context_detector;

#[cfg(feature = "agent")]
pub mod rule_based_detector;

#[cfg(feature = "agent")]
pub mod agent_based_detector;

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

pub use capability::Capability;
pub use error::AgentError;
#[cfg(feature = "agent")]
pub use expertise_agent::ExpertiseAgent;
pub use payload::{Payload, PayloadContent};
#[cfg(feature = "agent")]
pub use payload_message::{
    PayloadMessage, RelatedParticipant, RelatedPayloadMessage, SpeakerRelation,
    participant_relation,
};

#[cfg(feature = "agent")]
pub use env_context::{EnvContext, JournalSummary, StepInfo};

#[cfg(feature = "agent")]
pub use detected_context::{ConfidenceScores, DetectedContext};

#[cfg(feature = "agent")]
pub use execution_context::{ExecutionContext, ExecutionContextExt};

#[cfg(feature = "agent")]
pub use context_detector::{ContextDetector, DetectContextExt};

#[cfg(feature = "agent")]
pub use rule_based_detector::RuleBasedDetector;

#[cfg(feature = "agent")]
pub use agent_based_detector::AgentBasedDetector;

use crate::prompt::ToPrompt;
use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;

/// A trait for types that can serve as agent expertise.
///
/// This trait bridges the gap between simple string-based expertise (suitable for most users)
/// and complex `Expertise` types with weighted fragments, priorities, and context-aware prompts.
///
/// # Design Philosophy
///
/// - **Simple for beginners**: Plain strings (`&str`, `String`) work out of the box
/// - **Powerful for advanced users**: `Expertise` type enables composition, priorities, and tool definitions
/// - **Type-safe delegation**: Agent methods automatically delegate to expertise
///
/// # Examples
///
/// ## Simple usage with plain strings
///
/// ```rust,ignore
/// #[agent(expertise = "GitHub operations and Rust code review specialist")]
/// struct SimpleAgent;
/// ```
///
/// ## Advanced usage with Expertise type
///
/// ```rust,ignore
/// use llm_toolkit_expertise::Expertise;
///
/// #[agent(expertise = self.expertise_def)]
/// struct AdvancedAgent {
///     expertise_def: Expertise,
/// }
/// ```
pub trait ToExpertise: ToPrompt {
    /// Returns a lightweight catalog description for Orchestrator routing.
    ///
    /// This should be a concise summary (1-2 sentences) that helps the orchestrator
    /// select the appropriate agent. For simple string expertise, this is the same
    /// as the prompt. For complex `Expertise` types, this is a separate description field.
    fn description(&self) -> &str;

    /// Returns the list of capabilities (tools/actions) this expertise provides.
    ///
    /// This is used for precise orchestrator strategy generation and dialogue coordination.
    /// The default implementation returns an empty vector.
    ///
    /// For `Expertise` types, this extracts capabilities from `ToolDefinition` fragments.
    fn capabilities(&self) -> Vec<Capability> {
        Vec::new()
    }
}

// Implement ToExpertise for plain strings (simple case - most users)
impl ToExpertise for &str {
    fn description(&self) -> &str {
        self
    }
}

impl ToExpertise for String {
    fn description(&self) -> &str {
        self.as_str()
    }
}

/// The output type for agent execution.
///
/// This enum represents the result of agent execution, allowing agents to either
/// return a successful result or request human approval before proceeding.
#[derive(Debug, Clone)]
pub enum AgentOutput {
    /// The agent completed successfully with the given output.
    Success(serde_json::Value),
    /// The agent requires human approval before proceeding.
    ///
    /// Contains a message for the human and the current state of the payload.
    RequiresApproval {
        message_for_human: String,
        current_payload: serde_json::Value,
    },
}

/// The core trait for defining an agent.
///
/// An agent represents a reusable capability that can execute tasks based on
/// natural language intents. The agent's expertise and output type are statically
/// defined, while the specific task is provided dynamically at runtime.
///
/// # Type Parameters
///
/// - `Output`: The structured output type this agent produces
/// - `Expertise`: The expertise type (can be `&str`, `String`, or `Expertise`)
///
/// # Examples
///
/// ## Simple agent with string expertise
///
/// ```rust,ignore
/// #[agent(expertise = "GitHub operations specialist")]
/// struct SimpleAgent;
/// // Generated: type Expertise = &'static str;
/// ```
///
/// ## Advanced agent with Expertise type
///
/// ```rust,ignore
/// #[agent(expertise = self.expertise_def)]
/// struct AdvancedAgent {
///     expertise_def: llm_toolkit_expertise::Expertise,
/// }
/// // Generated: type Expertise = llm_toolkit_expertise::Expertise;
/// ```
#[async_trait]
pub trait Agent: Send + Sync {
    /// The type of output this agent produces.
    ///
    /// This type must be serializable and deserializable to enable
    /// communication between agents and persistence of results.
    type Output: Serialize + DeserializeOwned;

    /// The type of expertise this agent uses.
    ///
    /// This can be a simple string (`&str`, `String`) for most cases, or a complex
    /// `Expertise` type with weighted fragments, priorities, and tool definitions.
    type Expertise: ToExpertise;

    /// Returns the expertise definition for this agent.
    ///
    /// The expertise defines:
    /// - **description()**: Lightweight catalog summary for Orchestrator routing
    /// - **to_prompt()**: Full system prompt for LLM execution (HEAVY)
    /// - **capabilities()**: Tool/action definitions for precise orchestration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Simple case: returns &str
    /// fn expertise(&self) -> &&str {
    ///     &"GitHub operations and Rust code review specialist"
    /// }
    ///
    /// // Advanced case: returns Expertise
    /// fn expertise(&self) -> &Expertise {
    ///     &self.expertise_def
    /// }
    /// ```
    fn expertise(&self) -> &Self::Expertise;

    /// Returns a lightweight catalog description for Orchestrator routing.
    ///
    /// This is automatically delegated to `expertise().description()`.
    /// Override only if you need custom logic.
    fn description(&self) -> &str {
        self.expertise().description()
    }

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

    /// Returns a list of capabilities (tools/actions) this agent can perform.
    ///
    /// This is used by Orchestrator and Dialogue to understand what concrete
    /// actions an agent can execute, in addition to its general expertise.
    /// Capabilities enable:
    ///
    /// - **Orchestrator precision**: Strategy generation can select agents based on
    ///   concrete capabilities rather than just natural language expertise.
    /// - **Dialogue coordination**: Agents can discover what other participants can do.
    /// - **Dynamic policy enforcement**: Dialogues can restrict which capabilities
    ///   are allowed in a given session.
    ///
    /// # Default Implementation
    ///
    /// By default, this delegates to `expertise().capabilities()` and returns `None`
    /// if the list is empty, `Some(vec)` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // For Expertise types with ToolDefinition fragments:
    /// // capabilities() automatically extracts from tool definitions
    ///
    /// // For manual override:
    /// fn capabilities(&self) -> Option<Vec<Capability>> {
    ///     Some(vec![
    ///         Capability::new("file:read"),
    ///         Capability::new("file:write")
    ///             .with_description("Write content to a file"),
    ///     ])
    /// }
    /// ```
    fn capabilities(&self) -> Option<Vec<Capability>> {
        let caps = self.expertise().capabilities();
        if caps.is_empty() { None } else { Some(caps) }
    }
}

/// A type-erased agent wrapper for easy dynamic dispatch.
///
/// This wrapper allows you to store agents with different expertise types
/// in the same collection by erasing the Expertise type parameter.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::AnyAgent;
///
/// // Create from any agent
/// let agent1 = AnyAgent::new(simple_agent);  // expertise = &str
/// let agent2 = AnyAgent::new(advanced_agent); // expertise = Expertise
///
/// // Store in collections
/// let agents: Vec<Arc<AnyAgent<String>>> = vec![
///     Arc::new(agent1),
///     Arc::new(agent2),
/// ];
/// ```
pub struct AnyAgent<T: Serialize + DeserializeOwned> {
    inner: Box<dyn DynamicAgentInternal<T>>,
    description: String,
    capabilities: Option<Vec<Capability>>,
}

impl<T: Serialize + DeserializeOwned> AnyAgent<T> {
    /// Create a new AnyAgent from any Agent implementation.
    pub fn new<A: Agent<Output = T> + 'static>(agent: A) -> Self {
        let description = agent.description().to_string();
        let capabilities = agent.capabilities();
        Self {
            inner: Box::new(agent),
            description,
            capabilities,
        }
    }

    /// Create a new AnyAgent by boxing an agent.
    pub fn boxed<A: Agent<Output = T> + 'static>(agent: A) -> Box<Self> {
        Box::new(Self::new(agent))
    }

    /// Create a new AnyAgent by arc-ing an agent.
    pub fn arc<A: Agent<Output = T> + 'static>(agent: A) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::new(agent))
    }
}

#[async_trait]
impl<T: Serialize + DeserializeOwned> Agent for AnyAgent<T> {
    type Output = T;
    type Expertise = String;

    fn expertise(&self) -> &String {
        &self.description
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn capabilities(&self) -> Option<Vec<Capability>> {
        self.capabilities.clone()
    }

    async fn execute(&self, intent: Payload) -> Result<T, AgentError> {
        self.inner.execute(intent).await
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        self.inner.is_available().await
    }
}

/// A boxed agent trait object for dynamic dispatch.
///
/// **Deprecated**: Use `AnyAgent<T>` instead for better ergonomics.
///
/// This type alias requires specifying the Expertise type, which is
/// inconvenient for dynamic dispatch. Use `Box<AnyAgent<T>>` instead.
#[deprecated(
    since = "0.56.0",
    note = "Use Box<AnyAgent<T>> or Arc<AnyAgent<T>> instead"
)]
pub type BoxedAgent<T, E = String> = Box<dyn Agent<Output = T, Expertise = E>>;

/// Normalize raw LLM responses for agents that output plain `String` values.
///
/// This helper mirrors the default extraction pipeline (tag stripping, brace detection,
/// etc.) but adds a final `OriginalText` fallback so that purely textual outputs are
/// returned untouched. If the extracted content is itself a JSON string, the outer
/// quotes are removed to match the behavior users expect from plain text agents.
pub fn normalize_string_output(response: &str) -> String {
    use crate::extract::{ExtractionStrategy, FlexibleExtractor};

    if let Ok(extracted) = crate::extract_json(response) {
        if let Ok(unquoted) = serde_json::from_str::<String>(&extracted) {
            return unquoted;
        }
        return extracted;
    }

    let extractor = FlexibleExtractor::new();
    let mut strategies = FlexibleExtractor::standard_extraction_strategies();
    strategies.push(ExtractionStrategy::OriginalText);

    let extracted = extractor
        .extract_with_strategies(response, &strategies)
        .unwrap_or_else(|_| response.to_string());

    if let Ok(unquoted) = serde_json::from_str::<String>(&extracted) {
        return unquoted;
    }

    extracted
}

/// Dynamic agent trait for type-erased agent execution.
///
/// This trait allows the orchestrator to work with agents of different output types
/// by converting all outputs to `serde_json::Value`. This enables heterogeneous
/// agent collections while maintaining type safety at the agent implementation level.
#[async_trait]
pub trait DynamicAgent: Send + Sync {
    /// Execute the agent and return the output, which may require human approval.
    async fn execute_dynamic(&self, intent: Payload) -> Result<AgentOutput, AgentError>;

    /// Returns the name of this agent.
    fn name(&self) -> String;

    /// Returns a lightweight catalog description for Orchestrator routing.
    ///
    /// This is the type-erased version of `Agent::description()`.
    fn description(&self) -> &str;

    /// Returns a natural language description of what this agent can do.
    ///
    /// **Deprecated**: Use `description()` instead. This method is kept for
    /// backward compatibility and delegates to `description()`.
    #[deprecated(since = "0.56.0", note = "Use description() instead")]
    fn expertise(&self) -> &str {
        self.description()
    }

    /// Checks if the agent's backend is available.
    async fn is_available(&self) -> Result<(), AgentError> {
        Ok(())
    }

    /// Returns the capabilities of this agent.
    ///
    /// This mirrors the `Agent::capabilities()` method for type-erased agents.
    fn capabilities(&self) -> Option<Vec<Capability>> {
        None
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
///
/// The expertise type is also erased to `String` for dynamic dispatch.
pub struct AgentAdapter<T: Serialize + DeserializeOwned> {
    inner: Box<dyn DynamicAgentInternal<T>>,
    try_to_prompt_fn: Option<ToPromptFn>,
}

/// Internal trait for type-erasing Agent with specific Output type.
///
/// This trait bridges Agent<Output = T, Expertise = E> to DynamicAgent.
#[async_trait]
trait DynamicAgentInternal<T>: Send + Sync {
    async fn execute(&self, intent: Payload) -> Result<T, AgentError>;
    fn name(&self) -> String;
    fn description(&self) -> &str;
    async fn is_available(&self) -> Result<(), AgentError>;
    fn capabilities(&self) -> Option<Vec<Capability>>;
}

/// Blanket implementation for all Agent types
#[async_trait]
impl<T, A> DynamicAgentInternal<T> for A
where
    T: Serialize + DeserializeOwned,
    A: Agent<Output = T> + Send + Sync,
{
    async fn execute(&self, intent: Payload) -> Result<T, AgentError> {
        Agent::execute(self, intent).await
    }

    fn name(&self) -> String {
        Agent::name(self)
    }

    fn description(&self) -> &str {
        Agent::description(self)
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        Agent::is_available(self).await
    }

    fn capabilities(&self) -> Option<Vec<Capability>> {
        Agent::capabilities(self)
    }
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
    async fn execute_dynamic(&self, intent: Payload) -> Result<AgentOutput, AgentError> {
        let output = self.inner.execute(intent).await?;
        let json_value = serde_json::to_value(output)
            .map_err(|e| AgentError::SerializationFailed(e.to_string()))?;
        Ok(AgentOutput::Success(json_value))
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        self.inner.is_available().await
    }

    fn capabilities(&self) -> Option<Vec<Capability>> {
        self.inner.capabilities()
    }

    fn try_to_prompt(&self, json: &serde_json::Value) -> Option<String> {
        self.try_to_prompt_fn.as_ref().and_then(|f| f(json))
    }
}

// Agent implementations for smart pointers (Box, Arc)
// These enable ergonomic use of boxed/arc-wrapped agents

#[async_trait]
impl<T: Agent + ?Sized> Agent for Box<T>
where
    T::Output: Send,
{
    type Output = T::Output;
    type Expertise = T::Expertise;

    fn expertise(&self) -> &T::Expertise {
        (**self).expertise()
    }

    fn description(&self) -> &str {
        (**self).description()
    }

    fn capabilities(&self) -> Option<Vec<Capability>> {
        (**self).capabilities()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        (**self).execute(intent).await
    }

    fn name(&self) -> String {
        (**self).name()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        (**self).is_available().await
    }
}

#[async_trait]
impl<T: Agent + ?Sized> Agent for Arc<T>
where
    T::Output: Send,
{
    type Output = T::Output;
    type Expertise = T::Expertise;

    fn expertise(&self) -> &T::Expertise {
        (**self).expertise()
    }

    fn description(&self) -> &str {
        (**self).description()
    }

    fn capabilities(&self) -> Option<Vec<Capability>> {
        (**self).capabilities()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        (**self).execute(intent).await
    }

    fn name(&self) -> String {
        (**self).name()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        (**self).is_available().await
    }
}
