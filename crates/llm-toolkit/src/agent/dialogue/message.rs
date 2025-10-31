//! Domain model for dialogue messages.
//!
//! This module defines the core entities and value objects for managing
//! dialogue messages with identity and lifecycle tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current Unix timestamp in seconds.
pub(super) fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time should be after UNIX_EPOCH")
        .as_secs()
}

/// Unique identifier for dialogue messages.
///
/// This provides entity identity for messages, allowing them to be
/// tracked and referenced across different contexts.
///
/// # Implementation Note
///
/// Currently uses an atomic counter for simplicity. Can be upgraded
/// to UUID if needed in the future.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(u64);

static MESSAGE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

impl MessageId {
    /// Creates a new unique message ID.
    pub fn new() -> Self {
        Self(MESSAGE_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Returns the inner ID value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// A single message in a dialogue (Entity).
///
/// This represents the canonical message that exists once in the system
/// and is referenced from multiple contexts (Dialogue, History, Agent).
///
/// # Design Notes
///
/// - **Entity**: Messages have identity via `MessageId`
/// - **Immutable**: Once created, messages should not be modified
/// - **Turn-based**: Messages are organized by turn number
/// - **Timestamped**: Each message records when it was created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueMessage {
    /// Unique identifier (Entity identity)
    pub id: MessageId,

    /// Turn number (1-indexed)
    pub turn: usize,

    /// Speaker of this message
    pub speaker: Speaker,

    /// Message content (what was actually said)
    pub content: String,

    /// Creation timestamp (Unix timestamp in seconds)
    pub timestamp: u64,

    /// Optional metadata
    #[serde(default)]
    pub metadata: MessageMetadata,
}

impl DialogueMessage {
    /// Creates a new dialogue message.
    pub fn new(turn: usize, speaker: Speaker, content: String) -> Self {
        Self {
            id: MessageId::new(),
            turn,
            speaker,
            content,
            timestamp: current_unix_timestamp(),
            metadata: MessageMetadata::default(),
        }
    }

    /// Returns the speaker's name.
    pub fn speaker_name(&self) -> &str {
        self.speaker.name()
    }

    /// Returns the speaker's role (if participant).
    pub fn speaker_role(&self) -> Option<&str> {
        self.speaker.role()
    }
}

/// Represents who spoke in a dialogue message.
///
/// # Design Notes
///
/// - System: Generated prompts/instructions
/// - Participant: Human or AI agent in the dialogue
///
/// TODO: Consider making role an enum for common roles (Engineer, Designer, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Speaker {
    /// System-generated prompt
    System,

    /// Participant in the dialogue
    Participant {
        /// Name of the participant
        name: String,

        /// Role/title of the participant
        role: String,
    },
}

impl Speaker {
    /// Returns the speaker's name.
    pub fn name(&self) -> &str {
        match self {
            Speaker::System => "System",
            Speaker::Participant { name, .. } => name,
        }
    }

    /// Returns the speaker's role (if participant).
    pub fn role(&self) -> Option<&str> {
        match self {
            Speaker::System => None,
            Speaker::Participant { role, .. } => Some(role),
        }
    }

    /// Creates a new participant speaker.
    pub fn participant(name: impl Into<String>, role: impl Into<String>) -> Self {
        Self::Participant {
            name: name.into(),
            role: role.into(),
        }
    }
}

/// Metadata associated with a dialogue message.
///
/// This can be extended with custom fields as needed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Estimated token count (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,

    /// Whether this message has attachments
    #[serde(default)]
    pub has_attachments: bool,

    /// Custom application data
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_id_uniqueness() {
        let id1 = MessageId::new();
        let id2 = MessageId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_message_creation() {
        let msg = DialogueMessage::new(1, Speaker::System, "Test message".to_string());

        assert_eq!(msg.turn, 1);
        assert_eq!(msg.speaker_name(), "System");
        assert_eq!(msg.content, "Test message");
    }

    #[test]
    fn test_participant_speaker() {
        let speaker = Speaker::participant("Alice", "Engineer");

        assert_eq!(speaker.name(), "Alice");
        assert_eq!(speaker.role(), Some("Engineer"));
    }

    #[test]
    fn test_system_speaker() {
        let speaker = Speaker::System;

        assert_eq!(speaker.name(), "System");
        assert_eq!(speaker.role(), None);
    }

    #[test]
    fn test_message_serialization() {
        let msg = DialogueMessage::new(
            1,
            Speaker::participant("Bob", "Designer"),
            "Hello".to_string(),
        );

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: DialogueMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, msg.id);
        assert_eq!(deserialized.turn, msg.turn);
        assert_eq!(deserialized.content, msg.content);
    }
}
