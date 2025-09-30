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
//! # Example
//!
//! ```rust,ignore
//! use llm_toolkit::agent::{Agent, AgentError};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct ArticleDraft {
//!     title: String,
//!     body: String,
//! }
//!
//! struct ContentSynthesizerAgent;
//!
//! #[async_trait::async_trait]
//! impl Agent for ContentSynthesizerAgent {
//!     type Output = ArticleDraft;
//!
//!     fn expertise(&self) -> &str {
//!         "Generate well-structured article drafts from research topics"
//!     }
//!
//!     async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
//!         // Implementation details...
//!         todo!()
//!     }
//! }
//! ```

pub mod error;

#[cfg(feature = "agent")]
pub mod impls;

pub use error::AgentError;

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
    /// * `intent` - A natural language description of the task to perform.
    ///              This should include all necessary context and information.
    ///
    /// # Returns
    ///
    /// A `Result` containing the structured output on success, or an `AgentError` on failure.
    async fn execute(&self, intent: String) -> Result<Self::Output, AgentError>;

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
}

/// A boxed agent trait object for dynamic dispatch.
///
/// This type alias is useful for storing heterogeneous collections of agents.
pub type BoxedAgent<T> = Box<dyn Agent<Output = T>>;
