//! Multi-agent dialogue system with message-based distribution.
//!
//! This module provides a complete system for managing multi-agent conversations
//! with the following key features:
//!
//! - **Message as Entity**: Each message has a unique ID and lifecycle
//! - **Context Distribution**: Agents receive context from other participants
//! - **History Management**: Each agent manages its own conversation history
//! - **Flexible Formatting**: Adaptive formatting based on content length
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ Dialogue (Coordinator)                                   │
//! │ - MessageStore: Stores all messages                     │
//! │ - Distributes context to agents                         │
//! │ - Records distribution log                              │
//! └─────────────────────────────────────────────────────────┘
//!              ↓ Distributes TurnInput
//! ┌─────────────────────────────────────────────────────────┐
//! │ Agent (HistoryAwareAgent)                               │
//! │ - Receives: user_prompt + context                       │
//! │ - Manages: Own conversation history                     │
//! │ - Responds: Based on full context                       │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use llm_toolkit::agent::dialogue::Dialogue;
//! use llm_toolkit::agent::persona::Persona;
//!
//! let mut dialogue = Dialogue::broadcast()
//!     .add_participant(persona1, agent1)
//!     .add_participant(persona2, agent2);
//!
//! // Turn 1: Initial prompt
//! let turns = dialogue.run("Discuss architecture").await?;
//!
//! // Turn 2: With context from previous turn
//! let turns = dialogue.run("Focus on database design").await?;
//! // Each agent receives context from other agents' Turn 1 responses
//! ```

pub mod message;
pub mod store;
pub mod turn_input;

// Re-export key types
pub use message::{DialogueMessage, MessageId, MessageMetadata, Speaker};
pub use store::MessageStore;
pub use turn_input::{
    AdaptiveContextFormatter, ContextFormatter, ContextMessage, MultipartContextFormatter,
    SimpleContextFormatter, TurnInput,
};

// Legacy compatibility - keep existing dialogue.rs types
mod legacy;
pub use legacy::{
    BroadcastOrder, Dialogue, DialogueBlueprint, DialogueSession, DialogueTurn, ExecutionModel,
};
