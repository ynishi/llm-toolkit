//! Joining strategy for dialogue participants.
//!
//! This module defines how much conversation history each participant receives
//! when joining a dialogue. Different strategies enable different use cases:
//!
//! - **Recent**: Regular participants seeing only recent messages
//! - **Fresh**: External consultants with no history bias
//! - **Full**: New joiners needing complete context
//! - **Range**: Advanced scenarios with precise control

use super::message::DialogueMessage;
use serde::{Deserialize, Serialize};

/// Determines how much conversation history a participant receives.
///
/// Different participants may need different amounts of context:
/// - Core team members might only need recent messages (Recent)
/// - External consultants might work better with no history (Fresh)
/// - New joiners might need full context (Full)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoiningStrategy {
    /// Only recent N turns (default: 5 turns).
    ///
    /// # Use Case
    /// - Regular conversation participants
    /// - Keeps context focused and token usage low
    /// - Prevents information overload
    ///
    /// # Example
    /// In a 20-turn conversation, participant only sees turns 16-20
    Recent {
        /// Number of recent turns to include (default: 5)
        #[serde(default = "default_recent_turns")]
        turns: usize,
    },

    /// No history - only the current prompt.
    ///
    /// # Use Case
    /// - External consultant brought in for specific expertise
    /// - Fresh perspective without bias from prior discussion
    /// - One-shot analysis or evaluation
    ///
    /// # Example
    /// "Given this code snippet, suggest improvements" (no prior context)
    Fresh,

    /// Complete history from the beginning.
    ///
    /// # Use Case
    /// - New participant joining an ongoing thread
    /// - Needs full context to understand the discussion
    /// - Deep analysis requiring complete information
    ///
    /// # Example
    /// In a 20-turn conversation, participant sees all turns 1-20
    Full,

    /// Range: only turns between `start` and `end` (inclusive).
    ///
    /// # Use Case
    /// - Advanced use cases requiring precise control
    /// - Testing or debugging specific conversation segments
    ///
    /// # Example
    /// `Range { start: 5, end: 10 }` shows only turns 5-10
    Range {
        /// Start turn number (inclusive)
        start: usize,
        /// End turn number (inclusive, None means current turn)
        end: Option<usize>,
    },
}

fn default_recent_turns() -> usize {
    5
}

impl Default for JoiningStrategy {
    fn default() -> Self {
        Self::Recent {
            turns: default_recent_turns(),
        }
    }
}

impl JoiningStrategy {
    /// Creates a Recent strategy with default turn count (5).
    pub fn recent() -> Self {
        Self::default()
    }

    /// Creates a Recent strategy with custom turn count.
    pub fn recent_with_turns(turns: usize) -> Self {
        Self::Recent { turns }
    }

    /// Creates a Fresh strategy (no history).
    pub fn fresh() -> Self {
        Self::Fresh
    }

    /// Creates a Full strategy (all history).
    pub fn full() -> Self {
        Self::Full
    }

    /// Creates a Range strategy with specific turn range.
    pub fn range(start: usize, end: Option<usize>) -> Self {
        Self::Range { start, end }
    }

    /// Filters messages based on the strategy and current turn number.
    ///
    /// # Arguments
    /// * `all_messages` - All available messages in chronological order
    /// * `current_turn` - The current turn number
    ///
    /// # Returns
    /// Filtered messages according to the strategy
    ///
    /// # Notes
    /// - Messages are returned in chronological order
    /// - Current turn messages are excluded (only past history)
    pub fn filter_messages<'a>(
        &self,
        all_messages: &[&'a DialogueMessage],
        current_turn: usize,
    ) -> Vec<&'a DialogueMessage> {
        match self {
            Self::Recent { turns } => {
                let start_turn = current_turn.saturating_sub(*turns);
                all_messages
                    .iter()
                    .copied()
                    .filter(|msg| msg.turn >= start_turn && msg.turn < current_turn)
                    .collect()
            }
            Self::Fresh => {
                // No history - empty vec
                Vec::new()
            }
            Self::Full => {
                // All messages up to current turn (exclusive)
                all_messages
                    .iter()
                    .copied()
                    .filter(|msg| msg.turn < current_turn)
                    .collect()
            }
            Self::Range { start, end } => {
                let end_turn = end.unwrap_or(current_turn);
                all_messages
                    .iter()
                    .copied()
                    .filter(|msg| msg.turn >= *start && msg.turn < end_turn)
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::dialogue::message::Speaker;

    fn create_test_messages(turn_count: usize) -> Vec<DialogueMessage> {
        (1..=turn_count)
            .map(|turn| {
                DialogueMessage::new(
                    turn,
                    Speaker::agent("Agent", "Role"),
                    format!("Turn {}", turn),
                )
            })
            .collect()
    }

    #[test]
    fn test_recent_strategy_default_is_5_turns() {
        let strategy = JoiningStrategy::recent();
        assert_eq!(strategy, JoiningStrategy::Recent { turns: 5 });
    }

    #[test]
    fn test_recent_strategy_filters_last_n_turns() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::recent_with_turns(3);
        let filtered = strategy.filter_messages(&msg_refs, 10);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].turn, 7);
        assert_eq!(filtered[1].turn, 8);
        assert_eq!(filtered[2].turn, 9);
    }

    #[test]
    fn test_recent_strategy_handles_fewer_messages_than_window() {
        let messages = create_test_messages(3);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::recent_with_turns(5);
        let filtered = strategy.filter_messages(&msg_refs, 4);

        // Should return all 3 messages (turn 1, 2, 3)
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].turn, 1);
        assert_eq!(filtered[1].turn, 2);
        assert_eq!(filtered[2].turn, 3);
    }

    #[test]
    fn test_recent_strategy_excludes_current_turn() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::recent_with_turns(5);
        let filtered = strategy.filter_messages(&msg_refs, 8);

        // Should return turns 3-7 (not turn 8)
        assert_eq!(filtered.len(), 5);
        assert_eq!(filtered[0].turn, 3);
        assert_eq!(filtered[4].turn, 7);
        assert!(filtered.iter().all(|msg| msg.turn < 8));
    }

    #[test]
    fn test_fresh_strategy_returns_empty() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::fresh();
        let filtered = strategy.filter_messages(&msg_refs, 10);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_fresh_strategy_always_empty_regardless_of_turn() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::fresh();

        // Check at different turn numbers
        assert_eq!(strategy.filter_messages(&msg_refs, 1).len(), 0);
        assert_eq!(strategy.filter_messages(&msg_refs, 5).len(), 0);
        assert_eq!(strategy.filter_messages(&msg_refs, 10).len(), 0);
    }

    #[test]
    fn test_full_strategy_returns_all_messages_before_current_turn() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::full();
        let filtered = strategy.filter_messages(&msg_refs, 11);

        // Should return all 10 messages (turns 1-10) when current_turn is 11
        assert_eq!(filtered.len(), 10);
        assert_eq!(filtered[0].turn, 1);
        assert_eq!(filtered[9].turn, 10);
    }

    #[test]
    fn test_full_strategy_excludes_current_turn() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::full();
        let filtered = strategy.filter_messages(&msg_refs, 5);

        // Should return turns 1-4 (not turn 5)
        assert_eq!(filtered.len(), 4);
        assert_eq!(filtered[0].turn, 1);
        assert_eq!(filtered[3].turn, 4);
        assert!(filtered.iter().all(|msg| msg.turn < 5));
    }

    #[test]
    fn test_full_strategy_empty_on_first_turn() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::full();
        let filtered = strategy.filter_messages(&msg_refs, 1);

        // No history on first turn
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_custom_strategy_with_explicit_range() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::range(3, Some(7));
        let filtered = strategy.filter_messages(&msg_refs, 10);

        // Should return turns 3, 4, 5, 6 (7 is exclusive)
        assert_eq!(filtered.len(), 4);
        assert_eq!(filtered[0].turn, 3);
        assert_eq!(filtered[3].turn, 6);
    }

    #[test]
    fn test_custom_strategy_with_none_end_uses_current_turn() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::range(5, None);
        let filtered = strategy.filter_messages(&msg_refs, 8);

        // Should return turns 5, 6, 7 (8 is exclusive)
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].turn, 5);
        assert_eq!(filtered[2].turn, 7);
    }

    #[test]
    fn test_custom_strategy_empty_when_start_equals_end() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::range(5, Some(5));
        let filtered = strategy.filter_messages(&msg_refs, 10);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_custom_strategy_empty_when_start_after_current() {
        let messages = create_test_messages(10);
        let msg_refs: Vec<&DialogueMessage> = messages.iter().collect();

        let strategy = JoiningStrategy::range(15, None);
        let filtered = strategy.filter_messages(&msg_refs, 10);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_default_strategy_is_recent_5() {
        let strategy = JoiningStrategy::default();
        assert_eq!(strategy, JoiningStrategy::Recent { turns: 5 });
    }

    #[test]
    fn test_helper_constructors() {
        assert_eq!(
            JoiningStrategy::recent(),
            JoiningStrategy::Recent { turns: 5 }
        );
        assert_eq!(
            JoiningStrategy::recent_with_turns(10),
            JoiningStrategy::Recent { turns: 10 }
        );
        assert_eq!(JoiningStrategy::fresh(), JoiningStrategy::Fresh);
        assert_eq!(JoiningStrategy::full(), JoiningStrategy::Full);
        assert_eq!(
            JoiningStrategy::range(3, Some(7)),
            JoiningStrategy::Range {
                start: 3,
                end: Some(7)
            }
        );
    }

    #[test]
    fn test_serde_serialization() {
        let strategies = vec![
            JoiningStrategy::recent(),
            JoiningStrategy::fresh(),
            JoiningStrategy::full(),
            JoiningStrategy::range(3, Some(7)),
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: JoiningStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }
}
