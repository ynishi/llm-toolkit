//! Domain model for dialogue messages.
//!
//! This module defines the core entities and value objects for managing
//! dialogue messages with identity and lifecycle tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::PayloadMessage;
use crate::attachment::Attachment;

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

/// A single message in a dialogue (Internal Entity).
///
/// This represents the canonical message entity stored in [`MessageStore`](super::store::MessageStore).
/// It contains complete metadata for persistence, querying, and history reconstruction.
///
/// # Relationship to DialogueTurn
///
/// `DialogueMessage` is the **internal representation** of a dialogue message,
/// with full metadata for storage and tracking.
///
/// The public API uses [`DialogueTurn`](super::DialogueTurn), which is a **lightweight DTO**
/// containing only speaker and content. This separation provides:
/// - **Internal flexibility**: Can add/modify metadata without breaking public API
/// - **Efficient storage**: Full tracking with ID, turn number, timestamp
/// - **Query support**: Can filter/search by turn, speaker, timestamp
///
/// # Conversion
///
/// - **To DialogueTurn**: Use [`Dialogue::history()`](super::Dialogue::history) - strips metadata
/// - **From DialogueTurn**: Use [`Dialogue::with_history()`](super::Dialogue::with_history) - generates new metadata
///
/// # Design Notes
///
/// - **Entity**: Messages have identity via `MessageId`
/// - **Immutable**: Once created, messages should not be modified (event sourcing pattern)
/// - **Turn-based**: Messages are organized by turn number for context retrieval
/// - **Timestamped**: Each message records when it was created for audit/debugging
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::dialogue::message::DialogueMessage;
/// use llm_toolkit::agent::dialogue::Speaker;
///
/// // Create a new message
/// let message = DialogueMessage::new(
///     1, // turn number
///     Speaker::agent("Alice", "Engineer"),
///     "Let's use Rust for this project".to_string(),
/// );
///
/// // Access metadata
/// println!("Message ID: {}", message.id.as_u64());
/// println!("Turn: {}", message.turn);
/// println!("Timestamp: {}", message.timestamp);
/// ```
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

    /// Tracks which agents have received this message as context.
    ///
    /// This field prevents duplicate context delivery - each agent receives
    /// other agents' responses exactly once as context in subsequent turns.
    ///
    /// # Variants
    ///
    /// - `Agents(Vec<Speaker>)`: List of specific agents that received this message
    /// - `All`: Message has been broadcast to all agents
    ///
    /// # Usage
    ///
    /// When a message is included in a payload sent to an agent, that agent's
    /// Speaker is added to this list. The MessageStore uses this to filter
    /// unsent messages when building context for the next turn.
    #[serde(default)]
    pub sent_agents: SentAgents,
}

/// Tracks which agents have received a message as context.
///
/// This enum prevents duplicate context delivery by recording which agents
/// have already seen this message.
///
/// # Design
///
/// - **Agents(Vec<Speaker>)**: Tracks individual agents that received the message
/// - **All**: Optimization for broadcast scenarios where all agents received the message
///
/// # State Transitions
///
/// ```text
/// Agents([])                  // Initial state (no one received)
///   -> Agents([Alice])        // Alice received
///   -> Agents([Alice, Bob])   // Alice and Bob received
///   -> All                    // Broadcast to all (optional optimization)
/// ```
///
/// # Serialization
///
/// Uses internally tagged representation for better JSON structure:
/// - `Agents([...])` → `{"type": "agents", "agents": [...]}`
/// - `All` → `{"type": "all"}`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SentAgents {
    /// Specific agents that received this message.
    #[serde(rename = "agents")]
    Agents {
        /// List of agents that received this message.
        agents: Vec<Speaker>,
    },

    /// All agents have received this message (broadcast).
    All,
}

impl Default for SentAgents {
    fn default() -> Self {
        Self::Agents { agents: vec![] }
    }
}

impl SentAgents {
    /// Records that this message was sent to the given agent.
    ///
    /// # Arguments
    ///
    /// * `speaker` - The agent that received this message as context
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut sent = SentAgents::default();
    /// sent.sent(Speaker::agent("Alice", "Engineer"));
    /// sent.sent(Speaker::agent("Bob", "Designer"));
    /// ```
    pub fn sent(&mut self, speaker: Speaker) {
        match self {
            Self::Agents { agents } => agents.push(speaker),
            Self::All => {} // Already All
        }
    }

    /// Returns true if no agents have received this message yet.
    ///
    /// # Returns
    ///
    /// - `true`: No agents received this message (Agents { agents: [] })
    /// - `false`: At least one agent received, or All
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Agents { agents } => agents.is_empty(),
            Self::All => false,
        }
    }
}

impl From<DialogueMessage> for PayloadMessage {
    fn from(msg: DialogueMessage) -> PayloadMessage {
        PayloadMessage {
            speaker: msg.speaker,
            content: msg.content,
            metadata: msg.metadata,
        }
    }
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
            sent_agents: SentAgents::default(),
        }
    }

    pub fn with_metadata(&mut self, metadata: &MessageMetadata) -> Self {
        self.metadata = metadata.clone();
        self.clone()
    }

    /// Returns the speaker's name.
    pub fn speaker_name(&self) -> &str {
        self.speaker.name()
    }

    /// Returns the speaker's role (if participant).
    pub fn speaker_role(&self) -> Option<&str> {
        self.speaker.role()
    }

    /// Returns true if this message has been sent to at least one agent as context.
    ///
    /// This is used to filter unsent messages when building context for subsequent turns.
    ///
    /// # Returns
    ///
    /// - `true`: Message has been delivered to at least one agent
    /// - `false`: Message has not been sent to any agent yet
    pub fn sent_to_agents(&self) -> bool {
        !self.sent_agents.is_empty()
    }

    /// Records that this message was sent to the given agent as context.
    ///
    /// This should be called after including this message in a payload sent to an agent.
    ///
    /// # Arguments
    ///
    /// * `speaker` - The agent that received this message as context
    pub fn sent(&mut self, speaker: Speaker) {
        self.sent_agents.sent(speaker);
    }
}

/// Represents who spoke in a dialogue message.
///
/// # Design Notes
///
/// - System: System-generated prompts/instructions
/// - User: Human user input (with name and role)
/// - Agent: AI agent with persona (name + role + optional icon)
///
/// # Visual Identity
///
/// Agents can optionally include an icon for visual anchoring, which improves
/// recognition in conversation logs and strengthens LLM role adherence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Speaker {
    /// System-generated prompt or instruction
    System,

    /// Human user
    User {
        /// Name of the user
        name: String,

        /// Role/title of the user (e.g., "Customer", "Admin", "Product Manager")
        role: String,
    },

    /// AI agent with persona
    Agent {
        /// Name of the agent
        name: String,

        /// Role/title of the agent
        role: String,

        /// Optional visual icon/emoji (e.g., "🎨", "🔧", "📊")
        #[serde(skip_serializing_if = "Option::is_none")]
        icon: Option<String>,
    },
}

impl Speaker {
    /// Returns the speaker's name.
    pub fn name(&self) -> &str {
        match self {
            Speaker::System => "System",
            Speaker::User { name, .. } => name,
            Speaker::Agent { name, .. } => name,
        }
    }

    /// Returns the speaker's role (if user or agent).
    pub fn role(&self) -> Option<&str> {
        match self {
            Speaker::System => None,
            Speaker::User { role, .. } => Some(role),
            Speaker::Agent { role, .. } => Some(role),
        }
    }

    /// Returns the speaker's icon (if agent with visual identity).
    pub fn icon(&self) -> Option<&str> {
        match self {
            Speaker::Agent { icon, .. } => icon.as_deref(),
            _ => None,
        }
    }

    /// Returns a display name with icon if available.
    ///
    /// Returns formats like:
    /// - "🎨 Alice" (agent with icon)
    /// - "Alice" (agent without icon, or user)
    /// - "System" (system)
    pub fn display_name(&self) -> String {
        match self {
            Speaker::System => "System".to_string(),
            Speaker::User { name, .. } => name.clone(),
            Speaker::Agent { name, icon, .. } => match icon {
                Some(icon) => format!("{} {}", icon, name),
                None => name.clone(),
            },
        }
    }

    /// Creates a new user speaker.
    pub fn user(name: impl Into<String>, role: impl Into<String>) -> Self {
        Self::User {
            name: name.into(),
            role: role.into(),
        }
    }

    /// Creates a new agent speaker.
    pub fn agent(name: impl Into<String>, role: impl Into<String>) -> Self {
        Self::Agent {
            name: name.into(),
            role: role.into(),
            icon: None,
        }
    }

    /// Creates a new agent speaker with icon.
    pub fn agent_with_icon(
        name: impl Into<String>,
        role: impl Into<String>,
        icon: impl Into<String>,
    ) -> Self {
        Self::Agent {
            name: name.into(),
            role: role.into(),
            icon: Some(icon.into()),
        }
    }

    /// Creates a new participant speaker (backward compatibility).
    #[deprecated(note = "Use `agent()` instead")]
    pub fn participant(name: impl Into<String>, role: impl Into<String>) -> Self {
        Self::agent(name, role)
    }
}

/// Type of message for controlling dialogue reaction behavior.
///
/// This allows fine-grained control over when agents should react to messages.
///
/// # Message Types
///
/// - **Conversational**: Normal back-and-forth dialogue messages (default)
/// - **Notification**: Status updates, progress reports (may or may not trigger reactions)
/// - **System**: Explicit system commands or instructions (typically triggers reactions)
/// - **ContextInfo**: Background information, command results (does not trigger reactions)
/// - **Custom**: Application-specific message types
///
/// # Examples
///
/// ```rust,ignore
/// // Conversational message (triggers reaction)
/// MessageMetadata::new().with_type(MessageType::Conversational)
///
/// // Command result as context (no reaction)
/// MessageMetadata::new().with_type(MessageType::ContextInfo)
///
/// // System notification
/// MessageMetadata::new().with_type(MessageType::Notification)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOrigin {
    /// Message that originated from an incoming Payload (user/system input).
    IncomingPayload,
    /// Message generated by an agent as part of its response.
    AgentGenerated,
}

impl Default for MessageOrigin {
    fn default() -> Self {
        Self::IncomingPayload
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// Conversational message in dialogue (default)
    Conversational,

    /// Notification or status update
    Notification,

    /// System command or instruction
    System,

    /// Context information only (e.g., command results, background info)
    ///
    /// Messages with this type are stored as context but do not trigger
    /// agent reactions by default.
    ContextInfo,

    /// Custom message type
    Custom(String),
}

impl Default for MessageType {
    fn default() -> Self {
        Self::Conversational
    }
}

/// Metadata associated with a dialogue message.
///
/// This can be extended with custom fields as needed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Estimated token count (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,

    /// Whether this message has attachments
    #[serde(default)]
    pub has_attachments: bool,

    /// Type of this message (affects reaction behavior)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_type: Option<MessageType>,

    /// Attachments associated with this message/payload.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,

    /// Where this message originated from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<MessageOrigin>,

    /// Custom application data
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

impl MessageMetadata {
    /// Creates a new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the message type.
    pub fn with_type(mut self, message_type: MessageType) -> Self {
        self.message_type = Some(message_type);
        self
    }

    /// Adds a custom key-value pair.
    pub fn with_custom(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Checks if this metadata has the specified message type.
    pub fn is_type(&self, message_type: &MessageType) -> bool {
        self.message_type.as_ref() == Some(message_type)
    }

    /// Returns true if this message should not trigger agent reactions.
    ///
    /// Currently, only `ContextInfo` messages are considered context-only.
    pub fn is_context_only(&self) -> bool {
        matches!(self.message_type, Some(MessageType::ContextInfo))
    }

    /// Attaches binary resources to this metadata.
    pub fn with_attachments(mut self, attachments: Vec<Attachment>) -> Self {
        if !attachments.is_empty() {
            self.attachments.extend(attachments);
            self.has_attachments = true;
        }
        self
    }

    /// Returns attachments associated with this metadata.
    pub fn attachments(&self) -> &[Attachment] {
        &self.attachments
    }

    /// Sets the message origin.
    pub fn with_origin(mut self, origin: MessageOrigin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Ensures an origin is set, leaving existing values untouched.
    pub fn ensure_origin(mut self, origin: MessageOrigin) -> Self {
        if self.origin.is_none() {
            self.origin = Some(origin);
        }
        self
    }

    /// Returns the origin of this message, if known.
    pub fn origin(&self) -> Option<MessageOrigin> {
        self.origin
    }
}

/// Formats a list of messages as a prompt string.
///
/// Uses adaptive formatting based on total content length:
/// - **Simple format** (< 1000 chars): Markdown-style with `#` headers
/// - **Multipart format** (≥ 1000 chars): Explicit delimiters (`===`, `───`)
///
/// # Arguments
///
/// * `messages` - List of (Speaker, content) tuples
///
/// # Example
///
/// ```ignore
/// use llm_toolkit::agent::dialogue::{Speaker, format_messages_to_prompt};
///
/// let messages = vec![
///     (Speaker::System, "Task: Discuss architecture".to_string()),
///     (Speaker::agent("Alice", "Engineer"), "I suggest microservices".to_string()),
/// ];
///
/// let prompt = format_messages_to_prompt(&messages);
/// ```
pub fn format_messages_to_prompt(messages: &[(Speaker, String)]) -> String {
    const MULTIPART_THRESHOLD: usize = 1000;

    // Calculate total content length
    let total_chars: usize = messages.iter().map(|(_, content)| content.len()).sum();

    if total_chars >= MULTIPART_THRESHOLD {
        format_messages_multipart(messages)
    } else {
        format_messages_simple(messages)
    }
}

/// Simple markdown format for messages.
fn format_messages_simple(messages: &[(Speaker, String)]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut output = String::from("# Messages\n\n");

    for (speaker, content) in messages {
        let speaker_info = match speaker.role() {
            Some(role) => format!("{} ({})", speaker.name(), role),
            None => speaker.name().to_string(),
        };

        output.push_str(&format!("## {}\n{}\n\n", speaker_info, content));
    }

    output
}

/// Multipart format with explicit delimiters for long messages.
fn format_messages_multipart(messages: &[(Speaker, String)]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut output = String::from(
        "=================================================================================\n\
         MESSAGES\n\
         =================================================================================\n\n",
    );

    for (speaker, content) in messages {
        let speaker_info = match speaker.role() {
            Some(role) => format!("{} ({})", speaker.name(), role),
            None => speaker.name().to_string(),
        };

        output.push_str(&format!(
            "───────────────────────────────────────────────────────────────────────────────\n\
             {}\n\
             ───────────────────────────────────────────────────────────────────────────────\n\
             {}\n\n",
            speaker_info, content
        ));
    }

    output
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
        let speaker = Speaker::agent("Alice", "Engineer");

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
        let msg = DialogueMessage::new(1, Speaker::agent("Bob", "Designer"), "Hello".to_string());

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: DialogueMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, msg.id);
        assert_eq!(deserialized.turn, msg.turn);
        assert_eq!(deserialized.content, msg.content);
    }

    #[test]
    fn test_format_messages_simple() {
        let messages = vec![
            (Speaker::System, "Task: Discuss architecture".to_string()),
            (
                Speaker::agent("Alice", "Engineer"),
                "I suggest microservices".to_string(),
            ),
        ];

        let prompt = format_messages_to_prompt(&messages);

        // Should use simple format (short messages)
        assert!(prompt.contains("# Messages"));
        assert!(prompt.contains("## System"));
        assert!(prompt.contains("## Alice (Engineer)"));
        assert!(prompt.contains("Task: Discuss architecture"));
        assert!(prompt.contains("I suggest microservices"));

        // Should NOT contain multipart delimiters
        assert!(!prompt.contains("==="));
        assert!(!prompt.contains("───"));
    }

    #[test]
    fn test_format_messages_multipart() {
        let long_content = "a".repeat(1500);
        let messages = vec![
            (Speaker::System, "Short task".to_string()),
            (Speaker::agent("Alice", "Engineer"), long_content.clone()),
        ];

        let prompt = format_messages_to_prompt(&messages);

        // Should use multipart format (long messages)
        assert!(prompt.contains("MESSAGES"));
        assert!(prompt.contains("==="));
        assert!(prompt.contains("───"));
        assert!(prompt.contains("System"));
        assert!(prompt.contains("Alice (Engineer)"));

        // Should NOT contain markdown headers
        assert!(!prompt.contains("# Messages"));
        assert!(!prompt.contains("## System"));
    }

    #[test]
    fn test_format_messages_empty() {
        let messages: Vec<(Speaker, String)> = vec![];
        let prompt = format_messages_to_prompt(&messages);

        assert_eq!(prompt, "");
    }

    #[test]
    fn test_format_messages_threshold() {
        // Test exactly at threshold (1000 chars)
        let content_999 = "a".repeat(999);
        let messages_under = vec![(Speaker::System, content_999)];
        let prompt_under = format_messages_to_prompt(&messages_under);
        assert!(
            prompt_under.contains("# Messages"),
            "Should use simple format for 999 chars"
        );

        let content_1000 = "a".repeat(1000);
        let messages_at = vec![(Speaker::System, content_1000)];
        let prompt_at = format_messages_to_prompt(&messages_at);
        assert!(
            prompt_at.contains("==="),
            "Should use multipart format for 1000 chars"
        );
    }

    // === SentAgents Tests ===

    #[test]
    fn test_sent_agents_default() {
        let sent = SentAgents::default();
        assert!(sent.is_empty());
        assert_eq!(sent, SentAgents::Agents { agents: vec![] });
    }

    #[test]
    fn test_sent_agents_add_single_agent() {
        let mut sent = SentAgents::default();
        let alice = Speaker::agent("Alice", "Engineer");

        sent.sent(alice.clone());

        assert!(!sent.is_empty());
        match sent {
            SentAgents::Agents { agents } => {
                assert_eq!(agents.len(), 1);
                assert_eq!(agents[0].name(), "Alice");
            }
            SentAgents::All => panic!("Expected Agents variant"),
        }
    }

    #[test]
    fn test_sent_agents_add_multiple_agents() {
        let mut sent = SentAgents::default();
        let alice = Speaker::agent("Alice", "Engineer");
        let bob = Speaker::agent("Bob", "Designer");

        sent.sent(alice.clone());
        sent.sent(bob.clone());

        assert!(!sent.is_empty());
        match sent {
            SentAgents::Agents { agents } => {
                assert_eq!(agents.len(), 2);
                assert_eq!(agents[0].name(), "Alice");
                assert_eq!(agents[1].name(), "Bob");
            }
            SentAgents::All => panic!("Expected Agents variant"),
        }
    }

    #[test]
    fn test_sent_agents_all_variant() {
        let sent = SentAgents::All;
        assert!(!sent.is_empty());
    }

    #[test]
    fn test_sent_agents_all_ignores_additional_agents() {
        let mut sent = SentAgents::All;
        let alice = Speaker::agent("Alice", "Engineer");

        sent.sent(alice);

        // Should remain All
        assert_eq!(sent, SentAgents::All);
        assert!(!sent.is_empty());
    }

    #[test]
    fn test_sent_agents_serialize_empty() {
        let sent = SentAgents::default();
        let json = serde_json::to_string(&sent).unwrap();

        // Should serialize as internally tagged
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "agents");
        assert!(value["agents"].is_array());
        assert_eq!(value["agents"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_sent_agents_serialize_with_agents() {
        let mut sent = SentAgents::default();
        sent.sent(Speaker::agent("Alice", "Engineer"));
        sent.sent(Speaker::agent("Bob", "Designer"));

        let json = serde_json::to_string(&sent).unwrap();

        // Should serialize as internally tagged
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "agents");
        assert!(value["agents"].is_array());
        assert_eq!(value["agents"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_sent_agents_serialize_all() {
        let sent = SentAgents::All;
        let json = serde_json::to_string(&sent).unwrap();

        // Should serialize as internally tagged
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "all");
        // All variant should not have any other fields
        assert!(value.get("agents").is_none());
    }

    #[test]
    fn test_sent_agents_deserialize_empty() {
        let json = r#"{"type":"agents","agents":[]}"#;
        let sent: SentAgents = serde_json::from_str(json).unwrap();

        assert!(sent.is_empty());
        assert_eq!(sent, SentAgents::Agents { agents: vec![] });
    }

    #[test]
    fn test_sent_agents_deserialize_with_agents() {
        let json = r#"{"type":"agents","agents":[
            {"type":"agent","name":"Alice","role":"Engineer"},
            {"type":"agent","name":"Bob","role":"Designer"}
        ]}"#;
        let sent: SentAgents = serde_json::from_str(json).unwrap();

        assert!(!sent.is_empty());
        match sent {
            SentAgents::Agents { agents } => {
                assert_eq!(agents.len(), 2);
                assert_eq!(agents[0].name(), "Alice");
                assert_eq!(agents[1].name(), "Bob");
            }
            SentAgents::All => panic!("Expected Agents variant"),
        }
    }

    #[test]
    fn test_sent_agents_deserialize_all() {
        let json = r#"{"type":"all"}"#;
        let sent: SentAgents = serde_json::from_str(json).unwrap();

        assert!(!sent.is_empty());
        assert_eq!(sent, SentAgents::All);
    }

    #[test]
    fn test_sent_agents_round_trip() {
        let mut original = SentAgents::default();
        original.sent(Speaker::agent("Alice", "Engineer"));
        original.sent(Speaker::agent("Bob", "Designer"));

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: SentAgents = serde_json::from_str(&json).unwrap();

        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_sent_agents_all_round_trip() {
        let original = SentAgents::All;

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: SentAgents = serde_json::from_str(&json).unwrap();

        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_dialogue_message_sent_to_agents() {
        let mut msg = DialogueMessage::new(1, Speaker::System, "Test".to_string());

        assert!(!msg.sent_to_agents());

        msg.sent(Speaker::agent("Alice", "Engineer"));
        assert!(msg.sent_to_agents());
    }

    #[test]
    fn test_dialogue_message_serialization_with_sent_agents() {
        let mut msg =
            DialogueMessage::new(1, Speaker::agent("Alice", "Engineer"), "Hello".to_string());
        msg.sent(Speaker::agent("Bob", "Designer"));

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: DialogueMessage = serde_json::from_str(&json).unwrap();

        assert!(deserialized.sent_to_agents());
    }
}
