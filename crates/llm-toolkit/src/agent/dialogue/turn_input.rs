//! DTOs for turn-based input to dialogue participants.
//!
//! This module defines the data structures for distributing messages
//! to agents, including context from other participants.

use crate::agent::payload_message::PayloadMessage;
use serde::{Deserialize, Serialize};

/// Information about a dialogue participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantInfo {
    /// Participant's name
    pub name: String,

    /// Participant's role
    pub role: String,

    /// Background/description of the participant
    pub description: String,
}

impl ParticipantInfo {
    /// Creates a new participant info.
    pub fn new(
        name: impl Into<String>,
        role: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            role: role.into(),
            description: description.into(),
        }
    }
}

/// A single turn's input to an agent.
///
/// # Design
///
/// - `user_prompt`: The current system/user prompt for this turn (legacy, single text)
/// - `current_messages`: Multiple structured messages for this turn (new)
/// - `context`: Messages from other participants (recent history)
/// - `participants`: All dialogue participants (who is in the conversation)
/// - `current_participant`: The name of the agent receiving this input
///
/// This DTO separates the agent's own conversation history
/// (managed by HistoryAwareAgent) from the dialogue context
/// (managed by Dialogue).
///
/// # Migration
///
/// When `current_messages` is non-empty, it takes precedence over `user_prompt`.
/// This allows backward compatibility while supporting structured multi-message turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnInput {
    /// The current user/system prompt for this turn (legacy)
    pub user_prompt: String,

    /// Multiple structured messages for this turn (takes precedence over user_prompt)
    #[serde(default)]
    pub current_messages: Vec<PayloadMessage>,

    /// Context messages from other participants (recent history)
    #[serde(default)]
    pub context: Vec<ContextMessage>,

    /// All dialogue participants
    #[serde(default)]
    pub participants: Vec<ParticipantInfo>,

    /// The name of the current participant (the one receiving this input)
    #[serde(default)]
    pub current_participant: String,
}

impl TurnInput {
    /// Creates a new turn input with just a user prompt (no context).
    pub fn new(user_prompt: impl Into<String>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            current_messages: Vec::new(),
            context: Vec::new(),
            participants: Vec::new(),
            current_participant: String::new(),
        }
    }

    /// Creates a turn input with context from other participants.
    pub fn with_context(user_prompt: impl Into<String>, context: Vec<ContextMessage>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            current_messages: Vec::new(),
            context,
            participants: Vec::new(),
            current_participant: String::new(),
        }
    }

    /// Creates a complete turn input with all dialogue information.
    pub fn with_dialogue_context(
        user_prompt: impl Into<String>,
        context: Vec<ContextMessage>,
        participants: Vec<ParticipantInfo>,
        current_participant: impl Into<String>,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            current_messages: Vec::new(),
            context,
            participants,
            current_participant: current_participant.into(),
        }
    }

    /// Creates a turn input with structured messages and full dialogue context.
    ///
    /// This is the new API for supporting multiple messages in a single turn,
    /// such as multiple system notifications or a conversation history snippet.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let messages = vec![
    ///     PayloadMessage::system("Alice joined"),
    ///     PayloadMessage::system("Bob joined"),
    ///     PayloadMessage::system("Discuss the topic"),
    /// ];
    /// let input = TurnInput::with_messages_and_context(
    ///     messages,
    ///     context,
    ///     participants,
    ///     "Alice"
    /// );
    /// ```
    pub fn with_messages_and_context(
        current_messages: Vec<PayloadMessage>,
        context: Vec<ContextMessage>,
        participants: Vec<ParticipantInfo>,
        current_participant: impl Into<String>,
    ) -> Self {
        Self {
            user_prompt: String::new(),
            current_messages,
            context,
            participants,
            current_participant: current_participant.into(),
        }
    }

    /// Converts this TurnInput into a vector of Messages for structured dialogue.
    ///
    /// This extracts:
    /// - Context messages from other participants (converted to Speaker + content)
    /// - Current messages (if `current_messages` is non-empty, it takes precedence)
    /// - Otherwise, `user_prompt` as a single System message (legacy)
    pub fn to_messages(&self) -> Vec<PayloadMessage> {
        use crate::agent::dialogue::Speaker;

        let mut messages = Vec::new();

        // Add context messages first (history from other participants)
        for ctx in &self.context {
            let speaker = if ctx.speaker_role == "System" {
                Speaker::System
            } else {
                Speaker::agent(&ctx.speaker_name, &ctx.speaker_role)
            };
            messages.push(PayloadMessage::new(speaker, ctx.content.clone()));
        }

        // Add current messages or user_prompt
        if !self.current_messages.is_empty() {
            // New: use structured messages
            messages.extend(self.current_messages.clone());
        } else if !self.user_prompt.is_empty() {
            // Legacy: use single user_prompt as System message
            messages.push(PayloadMessage::system(self.user_prompt.clone()));
        }

        messages
    }
}

/// Context message from another participant in the dialogue.
///
/// This represents what other agents said in recent turns,
/// which is distributed to all agents as context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMessage {
    /// Name of the speaker
    pub speaker_name: String,

    /// Role/title of the speaker
    pub speaker_role: String,

    /// Message content
    pub content: String,

    /// Turn number when this message was sent
    #[serde(default)]
    pub turn: usize,

    /// Unix timestamp when this message was created
    #[serde(default)]
    pub timestamp: u64,
}

impl ContextMessage {
    /// Creates a new context message (backward compatible).
    pub fn new(
        speaker_name: impl Into<String>,
        speaker_role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            speaker_name: speaker_name.into(),
            speaker_role: speaker_role.into(),
            content: content.into(),
            turn: 0,
            timestamp: 0,
        }
    }

    /// Creates a new context message with full information.
    pub fn with_metadata(
        speaker_name: impl Into<String>,
        speaker_role: impl Into<String>,
        content: impl Into<String>,
        turn: usize,
        timestamp: u64,
    ) -> Self {
        Self {
            speaker_name: speaker_name.into(),
            speaker_role: speaker_role.into(),
            content: content.into(),
            turn,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::dialogue::Speaker;

    #[test]
    fn test_turn_input_no_context() {
        let input = TurnInput::new("Test prompt");

        assert_eq!(input.user_prompt, "Test prompt");
        assert!(input.context.is_empty());

        // Verify to_messages conversion
        let messages = input.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].speaker, Speaker::System);
        assert_eq!(messages[0].content, "Test prompt");
    }

    #[test]
    fn test_turn_input_with_context() {
        let context = vec![
            ContextMessage::new("Alice", "Engineer", "I think we should use Rust"),
            ContextMessage::new("Bob", "Designer", "The UI needs to be simple"),
        ];

        let input = TurnInput::with_context("Discuss implementation", context);

        assert_eq!(input.user_prompt, "Discuss implementation");
        assert_eq!(input.context.len(), 2);

        // Verify to_messages conversion includes context + current prompt
        let messages = input.to_messages();
        assert_eq!(messages.len(), 3); // 2 context + 1 current

        // First context message
        assert_eq!(messages[0].speaker, Speaker::agent("Alice", "Engineer"));
        assert_eq!(messages[0].content, "I think we should use Rust");

        // Second context message
        assert_eq!(messages[1].speaker, Speaker::agent("Bob", "Designer"));
        assert_eq!(messages[1].content, "The UI needs to be simple");

        // Current prompt as System message
        assert_eq!(messages[2].speaker, Speaker::System);
        assert_eq!(messages[2].content, "Discuss implementation");
    }

    #[test]
    fn test_turn_input_with_dialogue_context() {
        let participants = vec![
            ParticipantInfo::new("Alice", "Engineer", "Expert in system architecture"),
            ParticipantInfo::new("Bob", "Designer", "Focuses on user experience"),
        ];

        let context = vec![ContextMessage::with_metadata(
            "Alice",
            "Engineer",
            "I think we should use microservices architecture",
            1,
            1699000000,
        )];

        let turn_input = TurnInput::with_dialogue_context(
            "Discuss the implementation plan",
            context,
            participants.clone(),
            "Bob",
        );

        // Verify participants are stored
        assert_eq!(turn_input.participants.len(), 2);
        assert_eq!(turn_input.current_participant, "Bob");

        // Verify to_messages conversion
        let messages = turn_input.to_messages();
        assert_eq!(messages.len(), 2); // 1 context + 1 current
        assert_eq!(messages[0].speaker, Speaker::agent("Alice", "Engineer"));
        assert_eq!(
            messages[0].content,
            "I think we should use microservices architecture"
        );
        assert_eq!(messages[1].speaker, Speaker::System);
        assert_eq!(messages[1].content, "Discuss the implementation plan");
    }

    #[test]
    fn test_context_message_with_metadata() {
        let msg = ContextMessage::with_metadata("Alice", "Engineer", "Test message", 5, 1699000000);

        assert_eq!(msg.speaker_name, "Alice");
        assert_eq!(msg.speaker_role, "Engineer");
        assert_eq!(msg.content, "Test message");
        assert_eq!(msg.turn, 5);
        assert_eq!(msg.timestamp, 1699000000);
    }
}
