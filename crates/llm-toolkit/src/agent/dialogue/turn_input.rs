//! DTOs for turn-based input to dialogue participants.
//!
//! This module defines the data structures for distributing messages
//! to agents, including context from other participants.

use serde::{Deserialize, Serialize};

/// Information about a dialogue participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
/// - `user_prompt`: The current system/user prompt for this turn
/// - `context`: Messages from other participants (recent history)
/// - `participants`: All dialogue participants (who is in the conversation)
/// - `current_participant`: The name of the agent receiving this input
///
/// This DTO separates the agent's own conversation history
/// (managed by HistoryAwareAgent) from the dialogue context
/// (managed by Dialogue).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnInput {
    /// The current user/system prompt for this turn
    pub user_prompt: String,

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
            context: Vec::new(),
            participants: Vec::new(),
            current_participant: String::new(),
        }
    }

    /// Creates a turn input with context from other participants.
    pub fn with_context(user_prompt: impl Into<String>, context: Vec<ContextMessage>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
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
            context,
            participants,
            current_participant: current_participant.into(),
        }
    }

    /// Formats the turn input as a complete prompt.
    ///
    /// # Format
    ///
    /// - If no context: returns just the user prompt
    /// - If context exists:
    ///   ```text
    ///   # Context from other participants
    ///
    ///   ## Speaker Name (Role)
    ///   Message content...
    ///
    ///   # Current task
    ///   User prompt...
    ///   ```
    pub fn to_prompt(&self) -> String {
        self.to_prompt_with_formatter(&SimpleContextFormatter)
    }

    /// Formats with a custom formatter.
    pub fn to_prompt_with_formatter(&self, formatter: &dyn ContextFormatter) -> String {
        formatter.format(self)
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

/// Strategy for formatting turn input with context.
pub trait ContextFormatter {
    fn format(&self, input: &TurnInput) -> String;
}

/// Simple markdown format for context.
pub struct SimpleContextFormatter;

impl ContextFormatter for SimpleContextFormatter {
    fn format(&self, input: &TurnInput) -> String {
        let mut output = String::new();

        // Participants section (if available)
        if !input.participants.is_empty() {
            output.push_str("# Participants\n\n");
            for participant in &input.participants {
                let marker = if participant.name == input.current_participant {
                    " (YOU)"
                } else {
                    ""
                };
                output.push_str(&format!(
                    "- **{}{}** ({})\n  {}\n\n",
                    participant.name, marker, participant.role, participant.description
                ));
            }
        }

        // Recent history section (if available)
        if !input.context.is_empty() {
            output.push_str("# Recent History\n\n");
            for ctx in &input.context {
                let marker = if ctx.speaker_name == input.current_participant {
                    " (YOU)"
                } else {
                    ""
                };
                let turn_info = if ctx.turn > 0 {
                    format!("[Turn {}] ", ctx.turn)
                } else {
                    String::new()
                };
                output.push_str(&format!(
                    "{}{}{} ({}):\n{}\n\n",
                    turn_info, ctx.speaker_name, marker, ctx.speaker_role, ctx.content
                ));
            }
        }

        // Current task (only add header if we have context sections)
        let has_context_sections = !input.participants.is_empty() || !input.context.is_empty();
        if has_context_sections {
            output.push_str("# Current Task\n");
        }
        output.push_str(&input.user_prompt);

        output
    }
}

/// Multipart-style format for long discussions.
pub struct MultipartContextFormatter;

impl ContextFormatter for MultipartContextFormatter {
    fn format(&self, input: &TurnInput) -> String {
        let mut output = String::new();

        // Participants section (if available)
        if !input.participants.is_empty() {
            output.push_str(
                "=================================================================================\n\
                 PARTICIPANTS\n\
                 =================================================================================\n\n",
            );
            for participant in &input.participants {
                let marker = if participant.name == input.current_participant {
                    " (YOU)"
                } else {
                    ""
                };
                output.push_str(&format!(
                    "{}{} ({})\n{}\n\n",
                    participant.name, marker, participant.role, participant.description
                ));
            }
        }

        // Recent history section (if available)
        if !input.context.is_empty() {
            output.push_str(
                "=================================================================================\n\
                 RECENT HISTORY\n\
                 =================================================================================\n\n",
            );

            for ctx in &input.context {
                let marker = if ctx.speaker_name == input.current_participant {
                    " (YOU)"
                } else {
                    ""
                };
                let turn_info = if ctx.turn > 0 {
                    format!("[Turn {}] ", ctx.turn)
                } else {
                    String::new()
                };
                output.push_str(&format!(
                    "───────────────────────────────────────────────────────────────────────────────\n\
                     {}{}{} ({})\n\
                     ───────────────────────────────────────────────────────────────────────────────\n\
                     {}\n\n",
                    turn_info, ctx.speaker_name, marker, ctx.speaker_role, ctx.content
                ));
            }
        }

        // Current task (only add header if we have context sections)
        let has_context_sections = !input.participants.is_empty() || !input.context.is_empty();
        if has_context_sections {
            output.push_str(
                "=================================================================================\n\
                 CURRENT TASK\n\
                 =================================================================================\n",
            );
        }
        output.push_str(&input.user_prompt);

        output
    }
}

/// Adaptive formatter that selects format based on content length.
pub struct AdaptiveContextFormatter {
    /// Threshold in characters for switching to multipart
    pub multipart_threshold: usize,
}

impl Default for AdaptiveContextFormatter {
    fn default() -> Self {
        Self {
            multipart_threshold: 1000,
        }
    }
}

impl ContextFormatter for AdaptiveContextFormatter {
    fn format(&self, input: &TurnInput) -> String {
        let total_context_chars: usize = input.context.iter().map(|c| c.content.len()).sum();

        if total_context_chars > self.multipart_threshold {
            MultipartContextFormatter.format(input)
        } else {
            SimpleContextFormatter.format(input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_input_no_context() {
        let input = TurnInput::new("Test prompt");

        assert_eq!(input.user_prompt, "Test prompt");
        assert!(input.context.is_empty());
        assert_eq!(input.to_prompt(), "Test prompt");
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

        let formatted = input.to_prompt();
        assert!(formatted.contains("# Recent History"));
        assert!(formatted.contains("Alice (Engineer)"));
        assert!(formatted.contains("Bob (Designer)"));
        assert!(formatted.contains("# Current Task"));
        assert!(formatted.contains("Discuss implementation"));
    }

    #[test]
    fn test_multipart_formatter() {
        let context = vec![ContextMessage::new("Alice", "Engineer", "Long message...")];

        let input = TurnInput::with_context("Test", context);
        let formatted = input.to_prompt_with_formatter(&MultipartContextFormatter);

        // Check for multipart formatting elements
        assert!(
            formatted.contains("======="),
            "Should contain header separator"
        );
        assert!(
            formatted.contains("──────"),
            "Should contain message separator"
        );
        assert!(formatted.contains("RECENT HISTORY"));
        assert!(formatted.contains("CURRENT TASK"));
    }

    #[test]
    fn test_adaptive_formatter_simple() {
        let context = vec![ContextMessage::new("Alice", "Engineer", "Short")];

        let input = TurnInput::with_context("Test", context);
        let formatter = AdaptiveContextFormatter::default();
        let formatted = input.to_prompt_with_formatter(&formatter);

        // Should use simple format (no fancy borders)
        assert!(!formatted.contains("═══"));
        assert!(formatted.contains("# Recent History"));
    }

    #[test]
    fn test_adaptive_formatter_multipart() {
        let long_content = "a".repeat(1500);
        let context = vec![ContextMessage::new("Alice", "Engineer", &long_content)];

        let input = TurnInput::with_context("Test", context);
        let formatter = AdaptiveContextFormatter::default();
        let formatted = input.to_prompt_with_formatter(&formatter);

        // Should use multipart format
        assert!(
            formatted.contains("======="),
            "Should use multipart format for long content"
        );
    }

    #[test]
    fn test_full_dialogue_context_simple_format() {
        let participants = vec![
            ParticipantInfo::new("Alice", "Engineer", "Expert in system architecture"),
            ParticipantInfo::new("Bob", "Designer", "Focuses on user experience"),
            ParticipantInfo::new("Carol", "Marketer", "Marketing strategy specialist"),
        ];

        let context = vec![
            ContextMessage::with_metadata(
                "Alice",
                "Engineer",
                "I think we should use microservices architecture",
                1,
                1699000000,
            ),
            ContextMessage::with_metadata(
                "Bob",
                "Designer",
                "The UI needs to be responsive and intuitive",
                1,
                1699000010,
            ),
        ];

        let turn_input = TurnInput::with_dialogue_context(
            "Discuss the implementation plan for the new feature",
            context,
            participants,
            "Bob",
        );

        let formatted = turn_input.to_prompt_with_formatter(&SimpleContextFormatter);

        println!("\n=== Simple Formatter Output ===\n{}\n", formatted);

        // Verify structure
        assert!(formatted.contains("# Participants"));
        assert!(formatted.contains("**Alice** (Engineer)"));
        assert!(formatted.contains("**Bob (YOU)** (Designer)"));
        assert!(formatted.contains("**Carol** (Marketer)"));
        assert!(formatted.contains("# Recent History"));
        assert!(formatted.contains("[Turn 1] Alice (Engineer)"));
        assert!(formatted.contains("[Turn 1] Bob (YOU) (Designer)"));
        assert!(formatted.contains("# Current Task"));
        assert!(formatted.contains("Discuss the implementation plan"));
    }

    #[test]
    fn test_full_dialogue_context_multipart_format() {
        let participants = vec![
            ParticipantInfo::new("Alice", "Engineer", "Expert in system architecture"),
            ParticipantInfo::new("Bob", "Designer", "Focuses on user experience"),
        ];

        let context = vec![ContextMessage::with_metadata(
            "Alice",
            "Engineer",
            "I think we should use microservices architecture for better scalability",
            1,
            1699000000,
        )];

        let turn_input = TurnInput::with_dialogue_context(
            "Discuss the implementation plan",
            context,
            participants,
            "Bob",
        );

        let formatted = turn_input.to_prompt_with_formatter(&MultipartContextFormatter);

        println!("\n=== Multipart Formatter Output ===\n{}\n", formatted);

        // Verify structure
        assert!(formatted.contains("PARTICIPANTS"));
        assert!(formatted.contains("Bob (YOU)"));
        assert!(formatted.contains("RECENT HISTORY"));
        assert!(formatted.contains("CURRENT TASK"));
        assert!(formatted.contains("======="));
        assert!(formatted.contains("───────"));
    }
}
