//! Message storage and repository for dialogue messages.
//!
//! This module provides the central repository for managing dialogue messages
//! with efficient lookup by ID and chronological ordering.

use super::message::{DialogueMessage, MessageId, Speaker};
use std::collections::HashMap;

/// Central message repository within a Dialogue.
///
/// # Responsibility
///
/// - Store all dialogue messages with identity
/// - Provide efficient lookup by ID
/// - Maintain chronological order
/// - Support queries by turn, speaker, etc.
///
/// # Design Notes
///
/// - Messages are immutable once added
/// - Provides O(1) lookup by MessageId
/// - Maintains insertion order for chronological access
#[derive(Debug, Clone)]
pub struct MessageStore {
    /// All messages by ID (O(1) lookup)
    messages_by_id: HashMap<MessageId, DialogueMessage>,

    /// Ordered message IDs (chronological)
    message_order: Vec<MessageId>,
}

impl MessageStore {
    /// Creates a new empty message store.
    pub fn new() -> Self {
        Self {
            messages_by_id: HashMap::new(),
            message_order: Vec::new(),
        }
    }

    /// Adds a new message to the store.
    ///
    /// The message will be appended to the chronological order.
    pub fn push(&mut self, message: DialogueMessage) {
        let id = message.id;
        self.messages_by_id.insert(id, message);
        self.message_order.push(id);
    }

    /// Gets a message by its ID.
    pub fn get(&self, id: MessageId) -> Option<&DialogueMessage> {
        self.messages_by_id.get(&id)
    }

    /// Returns all messages in chronological order.
    pub fn all_messages(&self) -> Vec<&DialogueMessage> {
        self.message_order
            .iter()
            .filter_map(|id| self.messages_by_id.get(id))
            .collect()
    }

    /// Returns messages for a specific turn.
    pub fn messages_for_turn(&self, turn: usize) -> Vec<&DialogueMessage> {
        self.all_messages()
            .into_iter()
            .filter(|msg| msg.turn == turn)
            .collect()
    }

    /// Returns the current turn number.
    ///
    /// This counts the number of System messages (prompts) that have been sent.
    pub fn current_turn(&self) -> usize {
        self.all_messages()
            .iter()
            .filter(|msg| matches!(msg.speaker, Speaker::System))
            .count()
    }

    /// Returns the total number of messages.
    pub fn len(&self) -> usize {
        self.message_order.len()
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.message_order.is_empty()
    }

    /// Clears all messages from the store.
    pub fn clear(&mut self) {
        self.messages_by_id.clear();
        self.message_order.clear();
    }

    /// Returns messages that have not been sent to agents as context yet.
    ///
    /// This is used to get messages from previous turns that need to be
    /// distributed as context to agents in the next turn.
    ///
    /// Returns both System and Agent messages (excludes User messages).
    /// System messages are included to support cases like:
    /// - Sequential execution where the dialogue issues system prompts
    /// - History injection via system messages
    /// - Any other system-level context that agents should receive
    pub fn unsent_messages(&self) -> Vec<&DialogueMessage> {
        self.all_messages()
            .into_iter()
            .filter(|msg| {
                !msg.sent_to_agents
                    && (matches!(msg.speaker, Speaker::Agent { .. })
                        || matches!(msg.speaker, Speaker::System))
            })
            .collect()
    }

    /// Marks a message as sent to agents.
    ///
    /// This should be called after a message has been included in the context
    /// passed to agents in a subsequent turn.
    pub fn mark_as_sent(&mut self, id: MessageId) {
        if let Some(msg) = self.messages_by_id.get_mut(&id) {
            msg.sent_to_agents = true;
        }
    }

    /// Marks multiple messages as sent to agents.
    pub fn mark_all_as_sent(&mut self, ids: &[MessageId]) {
        for id in ids {
            self.mark_as_sent(*id);
        }
    }
}

impl Default for MessageStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::dialogue::message::Speaker;

    #[test]
    fn test_message_store_basic_operations() {
        let mut store = MessageStore::new();

        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        let msg1 = DialogueMessage::new(1, Speaker::System, "Hello".to_string());
        let msg1_id = msg1.id;
        store.push(msg1);

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let retrieved = store.get(msg1_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Hello");
    }

    #[test]
    fn test_chronological_order() {
        let mut store = MessageStore::new();

        let msg1 = DialogueMessage::new(1, Speaker::System, "First".to_string());
        let msg2 = DialogueMessage::new(1, Speaker::agent("A", "Role"), "Second".to_string());
        let msg3 = DialogueMessage::new(2, Speaker::System, "Third".to_string());

        store.push(msg1);
        store.push(msg2);
        store.push(msg3);

        let all = store.all_messages();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].content, "First");
        assert_eq!(all[1].content, "Second");
        assert_eq!(all[2].content, "Third");
    }

    #[test]
    fn test_messages_for_turn() {
        let mut store = MessageStore::new();

        let msg1 = DialogueMessage::new(1, Speaker::System, "Turn 1".to_string());
        let msg2 = DialogueMessage::new(1, Speaker::agent("A", "Role"), "Response 1".to_string());
        let msg3 = DialogueMessage::new(2, Speaker::System, "Turn 2".to_string());
        let msg4 = DialogueMessage::new(2, Speaker::agent("B", "Role"), "Response 2".to_string());

        store.push(msg1);
        store.push(msg2);
        store.push(msg3);
        store.push(msg4);

        let turn1 = store.messages_for_turn(1);
        assert_eq!(turn1.len(), 2);
        assert_eq!(turn1[0].content, "Turn 1");
        assert_eq!(turn1[1].content, "Response 1");

        let turn2 = store.messages_for_turn(2);
        assert_eq!(turn2.len(), 2);
        assert_eq!(turn2[0].content, "Turn 2");
        assert_eq!(turn2[1].content, "Response 2");
    }

    #[test]
    fn test_current_turn() {
        let mut store = MessageStore::new();

        assert_eq!(store.current_turn(), 0);

        store.push(DialogueMessage::new(
            1,
            Speaker::System,
            "Prompt 1".to_string(),
        ));
        assert_eq!(store.current_turn(), 1);

        store.push(DialogueMessage::new(
            1,
            Speaker::agent("A", "Role"),
            "Response".to_string(),
        ));
        assert_eq!(store.current_turn(), 1); // Still turn 1

        store.push(DialogueMessage::new(
            2,
            Speaker::System,
            "Prompt 2".to_string(),
        ));
        assert_eq!(store.current_turn(), 2);
    }

    #[test]
    fn test_clear() {
        let mut store = MessageStore::new();

        store.push(DialogueMessage::new(1, Speaker::System, "Test".to_string()));
        assert_eq!(store.len(), 1);

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_unsent_messages_returns_agent_and_system_messages() {
        let mut store = MessageStore::new();

        // Add various message types
        let msg1 = DialogueMessage::new(1, Speaker::System, "System prompt".to_string());
        let msg2 =
            DialogueMessage::new(1, Speaker::user("User", "Human"), "User input".to_string());
        let msg3 = DialogueMessage::new(
            1,
            Speaker::agent("Alice", "Engineer"),
            "Alice response".to_string(),
        );
        let msg4 = DialogueMessage::new(
            1,
            Speaker::agent("Bob", "Designer"),
            "Bob response".to_string(),
        );

        store.push(msg1);
        store.push(msg2);
        store.push(msg3);
        store.push(msg4);

        // unsent_messages should return System and Agent messages (excludes User)
        let unsent = store.unsent_messages();
        assert_eq!(unsent.len(), 3);
        assert_eq!(unsent[0].content, "System prompt");
        assert_eq!(unsent[1].content, "Alice response");
        assert_eq!(unsent[2].content, "Bob response");
    }

    #[test]
    fn test_unsent_messages_respects_sent_flag() {
        let mut store = MessageStore::new();

        let msg1 = DialogueMessage::new(
            1,
            Speaker::agent("Alice", "Engineer"),
            "Alice response".to_string(),
        );
        let msg1_id = msg1.id;
        let msg2 = DialogueMessage::new(
            1,
            Speaker::agent("Bob", "Designer"),
            "Bob response".to_string(),
        );
        let msg2_id = msg2.id;

        store.push(msg1);
        store.push(msg2);

        // Initially, both are unsent
        assert_eq!(store.unsent_messages().len(), 2);

        // Mark Alice's message as sent
        store.mark_as_sent(msg1_id);

        // Now only Bob's message should be unsent
        let unsent = store.unsent_messages();
        assert_eq!(unsent.len(), 1);
        assert_eq!(unsent[0].content, "Bob response");

        // Mark Bob's message as sent
        store.mark_as_sent(msg2_id);

        // Now no messages are unsent
        assert_eq!(store.unsent_messages().len(), 0);
    }

    #[test]
    fn test_mark_all_as_sent() {
        let mut store = MessageStore::new();

        let msg1 = DialogueMessage::new(
            1,
            Speaker::agent("Alice", "Engineer"),
            "Alice response".to_string(),
        );
        let msg1_id = msg1.id;
        let msg2 = DialogueMessage::new(
            1,
            Speaker::agent("Bob", "Designer"),
            "Bob response".to_string(),
        );
        let msg2_id = msg2.id;
        let msg3 = DialogueMessage::new(
            1,
            Speaker::agent("Charlie", "Manager"),
            "Charlie response".to_string(),
        );
        let msg3_id = msg3.id;

        store.push(msg1);
        store.push(msg2);
        store.push(msg3);

        // Initially all are unsent
        assert_eq!(store.unsent_messages().len(), 3);

        // Mark Alice and Bob as sent
        store.mark_all_as_sent(&[msg1_id, msg2_id]);

        // Only Charlie should be unsent
        let unsent = store.unsent_messages();
        assert_eq!(unsent.len(), 1);
        assert_eq!(unsent[0].content, "Charlie response");

        // Verify Alice and Bob are marked as sent
        assert!(store.get(msg1_id).unwrap().sent_to_agents);
        assert!(store.get(msg2_id).unwrap().sent_to_agents);
        assert!(!store.get(msg3_id).unwrap().sent_to_agents);
    }

    #[test]
    fn test_unsent_messages_excludes_user_only() {
        let mut store = MessageStore::new();

        // Add messages in Turn 1
        store.push(DialogueMessage::new(
            1,
            Speaker::System,
            "Turn 1 prompt".to_string(),
        ));
        store.push(DialogueMessage::new(
            1,
            Speaker::agent("Alice", "Engineer"),
            "Turn 1 Alice".to_string(),
        ));

        // Add messages in Turn 2 (including User message)
        store.push(DialogueMessage::new(
            2,
            Speaker::user("User", "Human"),
            "Turn 2 user input".to_string(),
        ));
        store.push(DialogueMessage::new(
            2,
            Speaker::agent("Bob", "Designer"),
            "Turn 2 Bob".to_string(),
        ));

        // unsent_messages should return System and Agent messages (excludes User)
        let unsent = store.unsent_messages();
        assert_eq!(unsent.len(), 3);

        // Verify it includes System and Agent messages, but not User
        assert_eq!(unsent[0].content, "Turn 1 prompt");
        assert!(matches!(unsent[0].speaker, Speaker::System));
        assert_eq!(unsent[1].content, "Turn 1 Alice");
        assert!(matches!(unsent[1].speaker, Speaker::Agent { .. }));
        assert_eq!(unsent[2].content, "Turn 2 Bob");
        assert!(matches!(unsent[2].speaker, Speaker::Agent { .. }));
    }

    #[test]
    fn test_mark_as_sent_nonexistent_id() {
        let mut store = MessageStore::new();

        let msg = DialogueMessage::new(
            1,
            Speaker::agent("Alice", "Engineer"),
            "Response".to_string(),
        );
        let msg_id = msg.id;
        store.push(msg);

        // Create a fake ID that doesn't exist
        let fake_id = MessageId::new();

        // Marking a non-existent ID should not panic
        store.mark_as_sent(fake_id);

        // The real message should still be unsent
        assert_eq!(store.unsent_messages().len(), 1);
        assert!(!store.get(msg_id).unwrap().sent_to_agents);
    }
}
