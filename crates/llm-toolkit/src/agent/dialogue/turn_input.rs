//! DTOs for turn-based input to dialogue participants.
//!
//! This module defines the data structures for distributing messages
//! to agents, including context from other participants.

use serde::{Deserialize, Serialize};

/// A single turn's input to an agent.
///
/// # Design
///
/// - `user_prompt`: The current system/user prompt for this turn
/// - `context`: Messages from other participants (in Broadcast mode)
///
/// This DTO separates the agent's own conversation history
/// (managed by HistoryAwareAgent) from the dialogue context
/// (managed by Dialogue).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnInput {
    /// The current user/system prompt for this turn
    pub user_prompt: String,

    /// Context messages from other participants
    #[serde(default)]
    pub context: Vec<ContextMessage>,
}

impl TurnInput {
    /// Creates a new turn input with just a user prompt (no context).
    pub fn new(user_prompt: impl Into<String>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            context: Vec::new(),
        }
    }

    /// Creates a turn input with context from other participants.
    pub fn with_context(user_prompt: impl Into<String>, context: Vec<ContextMessage>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            context,
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
/// This represents what other agents said in the previous turn,
/// which is distributed to all agents as context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMessage {
    /// Name of the speaker
    pub speaker_name: String,

    /// Role/title of the speaker
    pub speaker_role: String,

    /// Message content
    pub content: String,
}

impl ContextMessage {
    /// Creates a new context message.
    pub fn new(
        speaker_name: impl Into<String>,
        speaker_role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            speaker_name: speaker_name.into(),
            speaker_role: speaker_role.into(),
            content: content.into(),
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
        if input.context.is_empty() {
            return input.user_prompt.clone();
        }

        let mut output = String::new();

        // Context section
        output.push_str("# Context from other participants\n\n");
        for ctx in &input.context {
            output.push_str(&format!(
                "## {} ({})\n{}\n\n",
                ctx.speaker_name, ctx.speaker_role, ctx.content
            ));
        }

        // Current task
        output.push_str("# Current task\n");
        output.push_str(&input.user_prompt);

        output
    }
}

/// Multipart-style format for long discussions.
pub struct MultipartContextFormatter;

impl ContextFormatter for MultipartContextFormatter {
    fn format(&self, input: &TurnInput) -> String {
        if input.context.is_empty() {
            return input.user_prompt.clone();
        }

        let mut output = String::new();

        output.push_str(
            "=================================================================================\n\
             CONTEXT FROM OTHER PARTICIPANTS\n\
             =================================================================================\n\n",
        );

        for ctx in &input.context {
            output.push_str(&format!(
                "───────────────────────────────────────────────────────────────────────────────\n\
                 {} ({})\n\
                 ───────────────────────────────────────────────────────────────────────────────\n\
                 {}\n\n",
                ctx.speaker_name, ctx.speaker_role, ctx.content
            ));
        }

        output.push_str(
            "=================================================================================\n\
             CURRENT TASK\n\
             =================================================================================\n",
        );
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
        assert!(formatted.contains("Context from other participants"));
        assert!(formatted.contains("Alice (Engineer)"));
        assert!(formatted.contains("Bob (Designer)"));
        assert!(formatted.contains("Current task"));
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
        assert!(formatted.contains("CONTEXT FROM OTHER PARTICIPANTS"));
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
        assert!(formatted.contains("# Context from other participants"));
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
}
