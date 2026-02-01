//! Dialogue component for multi-agent conversational interactions.
//!
//! This module provides abstractions for managing turn-based dialogues between
//! multiple agents, with configurable turn-taking strategies.
//!

#![allow(clippy::result_large_err)]
//! # Architecture
//!
//! The dialogue system is built on a message-based architecture with clear separation of concerns:
//!
//! - **Message as Entity**: Each message has a unique ID and lifecycle
//! - **Context Distribution**: Agents receive context from other participants
//! - **History Management**: Each agent manages its own conversation history
//! - **Flexible Formatting**: Adaptive formatting based on content length
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
//! # Multimodal Input Support
//!
//! All dialogue methods accept `impl Into<Payload>`, allowing both text and
//! multimodal input (text + attachments) while maintaining backward compatibility.
//!
//! ## Examples
//!
//! ### Text-only input (backward compatible)
//!
//! ```rust,ignore
//! let mut dialogue = Dialogue::broadcast()
//!     .add_participant(persona, agent);
//!
//! // String literal
//! let turns = dialogue.run("Discuss AI ethics").await?;
//!
//! // String variable
//! let prompt = "Analyze this".to_string();
//! let turns = dialogue.run(prompt).await?;
//! ```
//!
//! ### Multimodal input with attachments
//!
//! ```rust,ignore
//! use llm_toolkit::attachment::Attachment;
//! use llm_toolkit::agent::Payload;
//!
//! let mut dialogue = Dialogue::broadcast()
//!     .add_participant(persona, agent);
//!
//! // Single attachment
//! let payload = Payload::text("What's in this image?")
//!     .with_attachment(Attachment::local("image.png"));
//! let turns = dialogue.run(payload).await?;
//!
//! // Multiple attachments
//! let payload = Payload::text("Analyze these files")
//!     .with_attachment(Attachment::local("data.csv"))
//!     .with_attachment(Attachment::local("metadata.json"));
//! let turns = dialogue.run(payload).await?;
//! ```
//!
//! ### Multi-turn conversations with context
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

pub mod constructor;
pub mod context;
pub mod joining_strategy;
pub mod message;
pub mod session;
pub mod state;
pub mod store;
pub mod turn_input;

use crate::ToPrompt;
use crate::agent::chat::Chat;
use crate::agent::dialogue::joining_strategy::JoiningStrategy;
use crate::agent::persona::Persona;
use crate::agent::{Agent, AgentError, Payload, PayloadMessage};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::task::JoinSet;
use tracing::{debug, error, trace};

// Re-export key types
pub use context::{DialogueContext, TalkStyle, TalkStyleTemplate};
pub use message::{
    DialogueMessage, MessageId, MessageMetadata, MessageOrigin, Speaker, format_messages_to_prompt,
};
pub use session::DialogueSession;
pub use store::MessageStore;
pub use turn_input::{ContextMessage, ParticipantInfo, TurnInput};

// Internal modules (not re-exported)
use state::{BroadcastState, SessionState};

/// Formats dialogue history as a human-readable conversation log.
///
/// Converts a vector of `DialogueTurn` into a formatted text representation
/// suitable for injecting as context. The format groups messages by speaker
/// and provides clear separation for readability.
///
/// # Arguments
///
/// * `history` - The conversation history to format
///
/// # Returns
///
/// A formatted string representation of the conversation
///
/// # Examples
///
/// ```rust,ignore
/// let history = vec![
///     DialogueTurn { speaker: Speaker::user("User", "User"), content: "Hello".to_string() },
///     DialogueTurn { speaker: Speaker::agent("Alice", "PM"), content: "Hi there!".to_string() },
/// ];
/// let formatted = format_dialogue_history_as_text(&history);
/// // Returns:
/// // # Previous Conversation History
/// //
/// // [User]
/// // Hello
/// //
/// // [Alice (PM)]
/// // Hi there!
/// ```
fn format_dialogue_history_as_text(history: &[DialogueTurn]) -> String {
    let mut output = String::from("# Previous Conversation History\n\n");
    output.push_str("The following is the conversation history from previous sessions. ");
    output.push_str("Please use this context to maintain continuity in the discussion.\n\n");

    for (idx, turn) in history.iter().enumerate() {
        // Add speaker label with appropriate formatting (includes icon if present)
        let speaker_label = match &turn.speaker {
            Speaker::System => "[System]".to_string(),
            Speaker::User { name, .. } => format!("[{}]", name),
            Speaker::Agent { name, role, icon } => match icon {
                Some(icon) => format!("[{} {} ({})]", icon, name, role),
                None => format!("[{} ({})]", name, role),
            },
        };

        output.push_str(&format!("{}. {}\n", idx + 1, speaker_label));
        output.push_str(&turn.content);
        output.push_str("\n\n");
    }

    output.push_str("---\n");
    output.push_str("End of previous conversation. Continue from here.\n");

    output
}

/// Extracts @mentions from a text string.
///
/// Finds all occurrences of `@name` pattern (where name is alphanumeric + underscores).
/// Performs exact matching against provided participant names.
///
/// # Arguments
///
/// * `text` - The text to search for mentions
/// * `participant_names` - List of valid participant names for matching
///
/// # Returns
///
/// A vector of matched participant names (deduplicated).
///
/// # Examples
///
/// Extracts @mentions from text using the specified matching strategy.
///
/// # Arguments
/// * `text` - The text to search for mentions
/// * `participant_names` - List of valid participant names
/// * `strategy` - The matching strategy to use
///
/// # Returns
/// A deduplicated vector of participant names that were mentioned in the text.
///
/// # Examples
/// ```rust,ignore
/// let text = "@Alice @Bob what do you think?";
/// let participants = vec!["Alice", "Bob", "Charlie"];
/// let mentions = extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::ExactWord);
/// assert_eq!(mentions, vec!["Alice", "Bob"]);
/// ```
fn extract_mentions_with_strategy<'a>(
    text: &str,
    participant_names: &'a [&'a str],
    strategy: MentionMatchStrategy,
) -> Vec<&'a str> {
    use std::collections::HashSet;

    let mut mentioned = HashSet::new();

    match strategy {
        MentionMatchStrategy::ExactWord => {
            // Match @word pattern (no spaces)
            // Matches until space or common ASCII delimiters
            // Multibyte delimiters typically have space before them, so space-based splitting works
            let mention_regex =
                Regex::new(r#"@([^\s@,.!?;:()\[\]{}<>"'`/\\|]+)"#).expect("Invalid regex pattern");
            for cap in mention_regex.captures_iter(text) {
                if let Some(mention) = cap.get(1) {
                    let mention_str = mention.as_str();
                    // Exact match against participant names
                    if let Some(&matched_name) =
                        participant_names.iter().find(|&&name| name == mention_str)
                    {
                        mentioned.insert(matched_name);
                    }
                }
            }
        }
        MentionMatchStrategy::Name => {
            // Match full names including spaces
            // Requires explicit delimiter (space or punctuation) after the name
            for &name in participant_names {
                // Match name followed by whitespace, common punctuation, or end of string
                // Note: Japanese honorifics like "さん" without space won't match
                // User should write "@あやか なかむら さん" with spaces
                let pattern = format!("@{}(?:\\s|[,.!?;:]|$)", regex::escape(name));
                if let Ok(name_regex) = Regex::new(&pattern)
                    && name_regex.is_match(text)
                {
                    mentioned.insert(name);
                }
            }

            // Remove names that are prefixes of other matched names
            // e.g., if both "Ayaka" and "Ayaka Nakamura" matched, keep only "Ayaka Nakamura"
            let mentioned_copy: Vec<&str> = mentioned.iter().copied().collect();
            mentioned.retain(|&name| {
                !mentioned_copy
                    .iter()
                    .any(|&other| other != name && other.starts_with(name))
            });
        }
        MentionMatchStrategy::Partial => {
            // Match by prefix, selecting longest matching candidate
            // Matches until space or common ASCII delimiters
            let mention_regex =
                Regex::new(r#"@([^\s@,.!?;:()\[\]{}<>"'`/\\|]+)"#).expect("Invalid regex pattern");

            for cap in mention_regex.captures_iter(text) {
                if let Some(mention) = cap.get(1) {
                    let mention_str = mention.as_str();

                    // Find all participants that start with this prefix
                    let mut matches: Vec<&str> = participant_names
                        .iter()
                        .filter(|&&name| name.starts_with(mention_str))
                        .copied()
                        .collect();

                    // Sort by length (descending) and take the longest match
                    matches.sort_by_key(|b| std::cmp::Reverse(b.len()));
                    if let Some(&longest_match) = matches.first() {
                        mentioned.insert(longest_match);
                    }
                }
            }
        }
    }

    mentioned.into_iter().collect()
}

/// Legacy function for backward compatibility (test-only).
/// Uses ExactWord strategy by default.
#[cfg(test)]
#[deprecated(since = "0.53.0", note = "Use extract_mentions_with_strategy instead")]
fn extract_mentions<'a>(text: &str, participant_names: &'a [&'a str]) -> Vec<&'a str> {
    extract_mentions_with_strategy(text, participant_names, MentionMatchStrategy::ExactWord)
}

/// Blueprint for creating a Dialogue.
///
/// Provides a high-level description of the dialogue setup, including
/// the agenda, context, optional participants, and execution strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueBlueprint {
    /// The agenda or topic for the dialogue
    pub agenda: String,

    /// Contextual information about the dialogue
    pub context: String,

    /// Optional pre-defined participants
    pub participants: Option<Vec<Persona>>,

    /// Optional execution strategy
    pub execution_strategy: Option<ExecutionModel>,
}

/// Represents a single turn in the dialogue (public API).
///
/// This is a lightweight Data Transfer Object (DTO) used for:
/// - **Streaming results**: Returned by `next_turn()` as each agent completes
/// - **Batch results**: Returned by `run()` as a collection of all turns
/// - **History snapshot**: Returned by `history()` for backward compatibility
/// - **Serialization**: Can be saved/loaded as JSON for dialogue persistence
///
/// # Relationship to DialogueMessage
///
/// `DialogueTurn` is the **public-facing** representation of a dialogue turn,
/// containing only the essential information (speaker and content).
///
/// Internally, the dialogue system uses [`DialogueMessage`](message::DialogueMessage),
/// which is an **internal entity** stored in [`MessageStore`](store::MessageStore).
/// `DialogueMessage` includes additional metadata like:
/// - Unique message ID
/// - Turn number
/// - Creation timestamp
/// - Custom metadata
///
/// This separation follows the DTO pattern and provides:
/// - **Lightweight streaming**: No overhead from IDs/timestamps during real-time emission
/// - **Clean public API**: Users don't need to manage internal details
/// - **Flexibility**: Internal storage format can evolve without breaking changes
///
/// # Conversion
///
/// - **From DialogueMessage**: Use [`Dialogue::history()`] to convert internal messages to public turns
/// - **To DialogueMessage**: Use [`Dialogue::with_history()`] to load turns into MessageStore
///
/// # Examples
///
/// ```rust,ignore
/// // Streaming turns as they complete
/// let mut session = dialogue.partial_session("Discuss the topic");
/// while let Some(result) = session.next_turn().await {
///     let turn = result?;
///     println!("{}: {}", turn.speaker.name(), turn.content);
/// }
///
/// // Batch execution
/// let turns = dialogue.run("Discuss the topic").await?;
/// for turn in turns {
///     println!("{}: {}", turn.speaker.name(), turn.content);
/// }
///
/// // Save/load dialogue history
/// let history = dialogue.history();
/// std::fs::write("dialogue.json", serde_json::to_string(&history)?)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    /// Who spoke in this turn
    pub speaker: Speaker,

    /// What was said
    pub content: String,
}

/// Strategy for matching @mentions in dialogue messages.
///
/// Different strategies handle various naming conventions and mention patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionMatchStrategy {
    /// Match `@word` pattern (no spaces).
    ///
    /// Matches any non-whitespace characters after `@`, supporting multibyte characters.
    /// Examples: `@Alice`, `@Bob123`, `@太郎`, `@あやか`
    ExactWord,

    /// Match full names including spaces with `@` prefix.
    ///
    /// Supports mentions like `@Ayaka Nakamura` for participant "Ayaka Nakamura".
    /// The mention must be an exact match of the participant's full name.
    Name,

    /// Match by prefix, selecting the longest matching candidate.
    ///
    /// Examples: `@Ayaka` matches "Ayaka Nakamura" but not "Alice".
    /// If multiple participants have the same prefix, the longest name is selected.
    Partial,
}

impl Default for MentionMatchStrategy {
    fn default() -> Self {
        Self::ExactWord
    }
}

/// Represents the execution model for dialogue strategies.
///
/// This enum unifies execution mode (Sequential/Broadcast/Mentioned) with
/// ordering strategy, eliminating the need for separate order fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionModel {
    /// Sequential execution.
    Sequential,

    /// Sequential execution with ordering control.
    ///
    /// Participants execute one by one, with output chained as input.
    /// Order can be specified explicitly or follow the addition order.
    OrderedSequential(SequentialOrder),

    /// Broadcast execution.
    Broadcast,

    /// Broadcast execution with ordering control.
    ///
    /// All participants respond in parallel to the same input.
    /// Order controls how results are yielded (completion order vs participant order).
    OrderedBroadcast(BroadcastOrder),

    /// Only @mentioned participants respond (falls back to Broadcast if no mentions).
    ///
    /// Supports multiple mentions like "@Alice @Bob what do you think?"
    /// If no mentions are found in the message, behaves like Broadcast mode.
    Mentioned {
        /// Strategy for matching mentions in messages.
        #[serde(default)]
        strategy: MentionMatchStrategy,
    },

    /// Moderator dynamically determines execution model.
    ///
    /// A moderator agent evaluates the current context and decides
    /// which execution model to use (Sequential with specific order,
    /// Broadcast, etc.) for each turn.
    Moderator,
}

/// Determines when agents should react to messages in a dialogue.
///
/// This strategy controls whether a message triggers agent responses or is
/// stored as context-only information. Useful for scenarios like:
/// - Slash command results that should be available as context but not trigger reactions
/// - System notifications that provide information without requiring responses
/// - Manual control over when agents should engage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReactionStrategy {
    /// Always react to all messages (default, backward compatible).
    Always,

    /// Only react to User messages.
    UserOnly,

    /// Only react to Agent messages.
    AgentOnly,

    /// React to all messages except System messages.
    ExceptSystem,

    /// React to conversational messages (User or Agent), excluding System messages.
    ///
    /// This is useful for dialogues where agents should engage in conversation
    /// but ignore system-level metadata or notifications.
    Conversational,

    /// React to all messages except ContextInfo type messages.
    ///
    /// ContextInfo messages are background information that should be available
    /// in history but not trigger agent responses. This strategy allows reacting
    /// to all other message types including System messages.
    ExceptContextInfo,
}

impl Default for ReactionStrategy {
    fn default() -> Self {
        Self::Always
    }
}

/// Represents a participant waiting to join the dialogue (pending state).
///
/// When a participant joins mid-dialogue via `join_in_progress()`, they are
/// first placed in this pending state ("waiting at the entrance"). On their
/// first turn, they receive a specially constructed payload based on their
/// JoiningStrategy, after which they transition to regular participant status.
#[derive(Debug, Clone)]
pub(super) struct PendingParticipant {
    /// JoiningStrategy determining how much history they receive
    pub joining_strategy: JoiningStrategy,
}

/// Internal representation of a dialogue participant.
///
/// Wraps a persona and its associated agent implementation.
pub(super) struct Participant {
    pub(super) persona: Persona,
    pub(super) agent: Arc<crate::agent::AnyAgent<String>>,
    /// Optional joining strategy for mid-dialogue participation.
    /// When set, controls how much conversation history this participant receives.
    pub(super) joining_strategy: Option<JoiningStrategy>,
    /// Tracks whether this participant has sent at least one message.
    /// Used to apply joining strategy only on the first interaction.
    pub(super) has_sent_once: bool,
}

impl Clone for Participant {
    fn clone(&self) -> Self {
        Self {
            persona: self.persona.clone(),
            agent: Arc::clone(&self.agent),
            joining_strategy: self.joining_strategy,
            has_sent_once: self.has_sent_once,
        }
    }
}

impl Participant {
    /// Returns the name of the participant from their persona.
    pub(super) fn name(&self) -> &str {
        &self.persona.name
    }

    /// Creates a Speaker from this participant's persona.
    ///
    /// Includes visual identity (icon) if present in the persona.
    pub(super) fn to_speaker(&self) -> Speaker {
        match &self.persona.visual_identity {
            Some(identity) => Speaker::agent_with_icon(
                self.persona.name.clone(),
                self.persona.role.clone(),
                identity.icon.clone(),
            ),
            None => Speaker::agent(self.persona.name.clone(), self.persona.role.clone()),
        }
    }
}

/// Controls the order in which broadcast responses are yielded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BroadcastOrder {
    /// Yields turns as soon as each participant finishes (default).
    Completion,
    /// Buffers responses and yields them in the original participant order.
    ParticipantOrder,
    /// Buffers responses and yields them in a custom specified order by participant name.
    Explicit(Vec<String>),
}

/// Controls the execution order for sequential dialogues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequentialOrder {
    /// Execute participants in the order they were added (default behavior).
    AsAdded,
    /// Execute participants by persona name. Any participants not listed are appended
    /// afterward in their original order.
    Explicit(Vec<String>),
}

/// A dialogue manager for multi-agent conversations.
///
/// The dialogue maintains a list of participants and a conversation history,
/// using a turn-taking strategy to determine execution behavior.
///
/// # Architecture (New)
///
/// - **MessageStore**: Central repository for all dialogue messages
/// - **Context Distribution**: Agents receive context from other participants
/// - **History Management**: Each agent manages own history via HistoryAwareAgent
/// - **Adaptive Formatting**: Automatic format selection based on content length
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::dialogue::Dialogue;
/// use llm_toolkit::agent::impls::ClaudeCodeAgent;
///
/// // Broadcast mode: all agents respond in parallel
/// let mut dialogue = Dialogue::broadcast()
///     .add_participant(persona1, ClaudeCodeAgent::new())
///     .add_participant(persona2, ClaudeCodeAgent::new());
///
/// let responses = dialogue.run("Discuss AI ethics".to_string()).await?;
///
/// // Sequential mode: output chains from one agent to the next
/// let mut dialogue = Dialogue::sequential()
///     .add_participant(persona1, ClaudeCodeAgent::new())
///     .add_participant(persona2, ClaudeCodeAgent::new());
///
/// let final_output = dialogue.run("Process this input".to_string()).await?;
/// ```
#[derive(Clone)]
pub struct Dialogue {
    pub(super) participants: Vec<Participant>,

    /// Message store for all dialogue messages
    pub(super) message_store: MessageStore,

    pub(super) execution_model: ExecutionModel,

    /// Optional dialogue context that shapes conversation tone and behavior
    pub(super) context: Option<DialogueContext>,

    /// Strategy for determining when agents should react to messages
    pub(super) reaction_strategy: ReactionStrategy,

    /// Optional moderator agent for dynamic execution model selection.
    ///
    /// When execution_model is Moderator, this agent is consulted to determine
    /// the execution strategy for each turn.
    pub(super) moderator: Option<Arc<crate::agent::AnyAgent<ExecutionModel>>>,

    /// Pool of participants waiting to join ("waiting at the entrance").
    ///
    /// Maps participant name to their pending state. When a participant joins
    /// mid-dialogue via `join_in_progress()`, they are placed here until their
    /// first turn completes, after which they transition to regular participant status.
    pub(super) pending_participants: HashMap<String, PendingParticipant>,
}

/// Prepared context for broadcast-based execution models.
///
/// Contains all common data needed to spawn tasks for participants:
/// - Participant info list
/// - Joining history contexts for pending participants
/// - Unsent messages from various sources
/// - Message IDs to mark as sent after task execution
///
/// This context is used by:
/// - `spawn_broadcast_tasks()`: Uses all participants
/// - `spawn_mentioned_tasks()`: Filters participants based on mentions
struct BroadcastContext {
    participants_info: Vec<ParticipantInfo>,
    joining_history_contexts: Vec<Option<Vec<PayloadMessage>>>,
    unsent_incoming: Vec<PayloadMessage>,
    unsent_from_agent: Vec<PayloadMessage>,
    message_ids_to_mark: Vec<MessageId>,
}

impl Dialogue {
    // Constructor is in constructor.rs

    /// Creates a single Participant from a Persona and LLM agent.
    ///
    /// This is a private helper that encapsulates the standard participant
    /// creation pattern used throughout the module.
    fn create_participant<T>(
        persona: Persona,
        llm_agent: T,
        joining_strategy: Option<JoiningStrategy>,
    ) -> Participant
    where
        T: Agent<Output = String> + 'static,
    {
        let chat_agent = Chat::new(llm_agent)
            .with_persona(persona.clone())
            .with_history(true)
            .with_joining_strategy(joining_strategy)
            .build();

        Participant {
            persona,
            agent: Arc::new(*chat_agent),
            joining_strategy,
            has_sent_once: false,
        }
    }

    /// Creates multiple Participants from a list of Personas.
    ///
    /// This helper converts a Vec<Persona> into Vec<Participant> by
    /// creating a Chat agent for each persona with the provided base agent.
    fn create_participants<T>(
        personas: Vec<Persona>,
        llm_agent: T,
        joining_strategy: Option<JoiningStrategy>,
    ) -> Vec<Participant>
    where
        T: Agent<Output = String> + Clone + 'static,
    {
        personas
            .into_iter()
            .map(|persona| Self::create_participant(persona, llm_agent.clone(), joining_strategy))
            .collect()
    }

    /// Returns participant information for all participants in the dialogue.
    ///
    /// This helper method extracts name, role, background, and capabilities from each
    /// participant's persona for use in context distribution.
    ///
    /// Capabilities are filtered by the dialogue's policy if one is set.
    fn get_participants_info(&self) -> Vec<ParticipantInfo> {
        self.participants
            .iter()
            .map(|p| {
                // Get capabilities from persona
                let mut capabilities = p.persona.capabilities.clone();

                // Apply policy filtering if context has a policy
                if let Some(ref context) = self.context
                    && let Some(ref policy) = context.policy
                    && let Some(allowed) = policy.get(&p.persona.name)
                {
                    // Filter: only keep capabilities that are in the allowed list
                    capabilities = capabilities.map(|caps| {
                        caps.into_iter()
                            .filter(|cap| allowed.contains(cap))
                            .collect()
                    });
                }

                ParticipantInfo::new(
                    p.persona.name.clone(),
                    p.persona.role.clone(),
                    p.persona.background.clone(),
                )
                .with_capabilities(capabilities.unwrap_or_default())
            })
            .collect()
    }

    /// Returns the indices for sequential execution respecting the configured order.
    ///
    /// This method extracts the SequentialOrder from the ExecutionModel and
    /// resolves it to participant indices.
    fn resolve_sequential_indices(
        &self,
        order: &SequentialOrder,
    ) -> Result<Vec<usize>, AgentError> {
        match order {
            SequentialOrder::AsAdded => Ok((0..self.participants.len()).collect()),
            SequentialOrder::Explicit(order) => {
                let mut indices = Vec::with_capacity(self.participants.len());
                let mut seen = HashSet::new();

                for name in order {
                    let idx = self
                        .participants
                        .iter()
                        .position(|p| p.name() == name)
                        .ok_or_else(|| {
                            AgentError::ExecutionFailed(format!(
                                "Sequential order references unknown participant '{}'",
                                name
                            ))
                        })?;

                    if seen.insert(idx) {
                        indices.push(idx);
                    }
                }

                for (idx, _) in self.participants.iter().enumerate() {
                    if seen.insert(idx) {
                        indices.push(idx);
                    }
                }

                Ok(indices)
            }
        }
    }

    pub fn add_participant<T>(&mut self, persona: Persona, llm_agent: T) -> &mut Self
    where
        T: Agent<Output = String> + 'static,
    {
        self.participants
            .push(Self::create_participant(persona, llm_agent, None));

        self
    }

    /// Adds a participant to an ongoing dialogue with custom joining strategy.
    ///
    /// This method is designed for mid-dialogue participation scenarios where
    /// a new agent joins an already-running conversation. Unlike [`add_participant`],
    /// which assumes initial setup before dialogue starts, this method requires
    /// explicit specification of how much conversation history the new participant
    /// should receive.
    ///
    /// # Use Cases
    ///
    /// - **Expert Consultation**: Bring in a specialist mid-conversation with fresh
    ///   perspective ([`JoiningStrategy::Fresh`])
    /// - **New Team Member**: Onboard someone who needs full context to contribute
    ///   meaningfully ([`JoiningStrategy::Full`])
    /// - **Focused Review**: Add a reviewer who only needs recent context
    ///   ([`JoiningStrategy::Recent`])
    ///
    /// # Arguments
    ///
    /// * `persona` - The identity and role of the joining participant
    /// * `llm_agent` - The underlying LLM agent implementation
    /// * `joining_strategy` - How much conversation history to provide
    ///
    /// # Returns
    ///
    /// Mutable reference to self for method chaining
    ///
    /// # Design Rationale
    ///
    /// This is a separate method from [`add_participant`] because:
    /// - Initial participants assume empty history (dialogue hasn't started)
    /// - Mid-dialogue joiners require explicit history handling decisions
    /// - Type system enforces that joining mid-dialogue requires a strategy
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Add expert with no historical bias
    /// dialogue.join_in_progress(
    ///     security_expert,
    ///     llm_agent.clone(),
    ///     JoiningStrategy::Fresh
    /// );
    ///
    /// // Add new team member with full context
    /// dialogue.join_in_progress(
    ///     new_developer,
    ///     llm_agent.clone(),
    ///     JoiningStrategy::Full
    /// );
    ///
    /// // Add reviewer needing only recent messages
    /// dialogue.join_in_progress(
    ///     code_reviewer,
    ///     llm_agent.clone(),
    ///     JoiningStrategy::recent_with_turns(10)
    /// );
    /// ```
    ///
    /// [`add_participant`]: Self::add_participant
    pub fn join_in_progress<T>(
        &mut self,
        persona: Persona,
        llm_agent: T,
        joining_strategy: JoiningStrategy,
    ) -> &mut Self
    where
        T: Agent<Output = String> + 'static,
    {
        let participant_name = persona.name.clone();

        // Add as regular participant (no joining_strategy set on Participant)
        self.participants.push(Self::create_participant(
            persona, llm_agent, None, // joining_strategy managed by PendingParticipant
        ));

        // Place in pending pool ("waiting at the entrance")
        self.pending_participants
            .insert(participant_name, PendingParticipant { joining_strategy });

        self
    }

    /// Adds a pre-configured agent to the dialogue as a participant.
    ///
    /// Use this when you need to configure the agent before adding it to the dialogue
    /// (e.g., PersonaAgent with custom ContextConfig). Unlike `add_participant`,
    /// this method accepts an already-configured agent instead of a base agent.
    ///
    /// # Arguments
    ///
    /// * `persona` - The persona information for this participant
    /// * `agent` - A pre-configured agent (e.g., PersonaAgent with settings)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use llm_toolkit::agent::persona::{PersonaAgent, ContextConfig};
    ///
    /// let config = ContextConfig {
    ///     participants_after_context: true,
    ///     include_trailing_prompt: true,
    ///     ..Default::default()
    /// };
    ///
    /// let persona_agent = PersonaAgent::new(base_agent, persona.clone())
    ///     .with_context_config(config);
    ///
    /// dialogue.add_agent(persona, persona_agent);
    /// ```
    pub fn add_agent<T>(&mut self, persona: Persona, agent: T) -> &mut Self
    where
        T: Agent<Output = String> + 'static,
    {
        let chat_agent = Chat::new(agent).with_history(true).build();

        self.participants.push(Participant {
            persona,
            agent: Arc::new(*chat_agent),
            joining_strategy: None,
            has_sent_once: false,
        });

        self
    }

    /// Returns the names of all current participants in the dialogue.
    ///
    /// This is useful for:
    /// - UI auto-completion of @mentions
    /// - Displaying current participants
    /// - Validating participant existence before operations
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::mentioned()
    ///     .add_participant(alice, agent1)
    ///     .add_participant(bob, agent2);
    ///
    /// let names = dialogue.participant_names();
    /// assert_eq!(names, vec!["Alice", "Bob"]);
    ///
    /// // Use for auto-completion in UI
    /// for name in dialogue.participant_names() {
    ///     println!("Available: @{}", name);
    /// }
    /// ```
    pub fn participant_names(&self) -> Vec<&str> {
        self.participants.iter().map(|p| p.name()).collect()
    }

    /// Removes a participant from the dialogue by name.
    ///
    /// This is useful for guest participants who are only needed for specific
    /// parts of the conversation.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the participant to remove
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or an error if the participant is not found.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::persona::Persona;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// let mut dialogue = Dialogue::from_persona_team(team, llm)?;
    ///
    /// // Add a guest customer success manager
    /// let cs_guest = Persona {
    ///     name: "CS Manager".to_string(),
    ///     /* ... */
    /// };
    /// dialogue.add_participant(cs_guest, ClaudeCodeAgent::new());
    /// assert_eq!(dialogue.participant_count(), 6); // 5 core + 1 guest
    ///
    /// // Get customer feedback
    /// let feedback = dialogue.run("Review feature UX from customer perspective").await?;
    ///
    /// // Remove guest after their input
    /// dialogue.remove_participant("CS Manager")?;
    /// assert_eq!(dialogue.participant_count(), 5); // Back to core team
    ///
    /// // Continue with core team
    /// let next_steps = dialogue.run("Define implementation plan").await?;
    /// ```
    pub fn remove_participant(&mut self, name: &str) -> Result<(), AgentError> {
        let position = self
            .participants
            .iter()
            .position(|p| p.name() == name)
            .ok_or_else(|| {
                AgentError::ExecutionFailed(format!(
                    "Cannot remove participant '{}': participant not found",
                    name
                ))
            })?;

        self.participants.remove(position);
        Ok(())
    }

    fn apply_metadata_attachments(mut payload: Payload, messages: &[PayloadMessage]) -> Payload {
        for msg in messages {
            for attachment in msg.metadata.attachments() {
                payload = payload.with_attachment(attachment.clone());
            }
        }
        payload
    }

    /// Converts a payload into DialogueMessages and stores them in the MessageStore.
    /// Returns the prompt text plus the IDs of stored messages.
    fn store_payload_messages(
        &mut self,
        payload: &Payload,
        turn: usize,
    ) -> (String, Vec<MessageId>) {
        let (messages, prompt_text) = self.extract_messages_from_payload(payload, turn);
        let mut stored_ids = Vec::new();
        for msg in messages {
            let id = msg.id;
            self.message_store.push(msg);
            stored_ids.push(id);
        }
        (prompt_text, stored_ids)
    }

    /// Determines whether agents should react to the given payload.
    ///
    /// This checks the reaction strategy to decide if the message should trigger
    /// agent responses or be stored as context-only information.
    ///
    /// # TODO: ReactionStrategy Design
    ///
    /// Current implementation only checks Speaker type (User/Agent/System).
    /// Future improvements needed:
    /// - Add test coverage for all strategies
    /// - Review overall ReactionStrategy design and semantics
    fn should_react(&self, payload: &Payload) -> bool {
        use crate::agent::dialogue::Speaker;
        use crate::agent::dialogue::message::MessageType;

        let messages = payload.to_messages();

        // If no messages (e.g., text-only payload), check based on strategy defaults
        if messages.is_empty() {
            // Text-only payloads should trigger reactions for most strategies
            return !matches!(self.reaction_strategy, ReactionStrategy::AgentOnly);
        }

        // Helper to check if message is ContextInfo
        let is_context_info = |msg: &crate::agent::PayloadMessage| {
            msg.metadata
                .message_type
                .as_ref()
                .map(|t| matches!(t, MessageType::ContextInfo))
                .unwrap_or(false)
        };

        // Never react if ALL messages are ContextInfo
        let all_context_info = messages.iter().all(is_context_info);
        if all_context_info {
            return false;
        }

        match &self.reaction_strategy {
            ReactionStrategy::Always => {
                // React to all messages except when all are ContextInfo (already checked)
                true
            }
            ReactionStrategy::UserOnly => {
                // React to User messages (excluding ContextInfo)
                messages
                    .iter()
                    .any(|msg| matches!(msg.speaker, Speaker::User { .. }) && !is_context_info(msg))
            }
            ReactionStrategy::AgentOnly => {
                // React to Agent messages (excluding ContextInfo)
                messages.iter().any(|msg| {
                    matches!(msg.speaker, Speaker::Agent { .. }) && !is_context_info(msg)
                })
            }
            ReactionStrategy::ExceptSystem => {
                // React to all non-System messages (excluding ContextInfo)
                messages
                    .iter()
                    .any(|msg| !matches!(msg.speaker, Speaker::System) && !is_context_info(msg))
            }
            ReactionStrategy::Conversational => {
                // React to User or Agent messages (excluding System and ContextInfo)
                messages.iter().any(|msg| {
                    (matches!(msg.speaker, Speaker::User { .. })
                        || matches!(msg.speaker, Speaker::Agent { .. }))
                        && !is_context_info(msg)
                })
            }
            ReactionStrategy::ExceptContextInfo => {
                // React to all messages except ContextInfo (already filtered above)
                true
            }
        }
    }

    fn next_turn(&self) -> usize {
        self.message_store.latest_turn() + 1
    }

    /// Join a pending participant's join-in-progress activation.
    ///
    /// This method handles the complete lifecycle of activating a pending participant:
    /// 1. Extracts filtered history according to the participant's JoiningStrategy
    /// 2. Removes the participant from the pending pool (now active)
    /// 3. Marks all historical messages as sent to this participant
    ///
    /// This logic is consistent across all execution modes (Broadcast, Sequential, Mentioned).
    ///
    /// # Arguments
    /// * `participant_name` - Name of the participant to activate
    /// * `participant` - Reference to the participant structure (for speaker info)
    /// * `current_turn` - The current turn number
    ///
    /// # Returns
    /// Filtered historical messages according to the participant's JoiningStrategy.
    /// Returns empty vec if the participant is not pending.
    fn join_pending_participant(
        &mut self,
        speaker: Speaker,
        current_turn: usize,
    ) -> Option<Vec<PayloadMessage>> {
        let name = speaker.name();
        if let Some(pending_info) = self.pending_participants.get(name) {
            // Extract filtered history according to JoiningStrategy
            let filtered_history: Vec<PayloadMessage> = {
                let all_messages = self.message_store.all_messages();
                let message_refs: Vec<&DialogueMessage> = all_messages.to_vec();
                let history_refs = pending_info
                    .joining_strategy
                    .historical_messages(&message_refs, current_turn);
                // Convert to PayloadMessage to release the borrow
                history_refs
                    .iter()
                    .map(|&msg| PayloadMessage::from(msg))
                    .collect()
            };

            // Remove from pending pool (participant is now active)
            self.pending_participants.remove(name);

            // Mark all historical messages as sent to this participant
            self.message_store.mark_as_sent_all_for(speaker.clone());

            trace!(
                target = "llm_toolkit::dialogue",
                participant = name,
                turn = current_turn,
                history_count = filtered_history.len(),
                "Activated pending participant with filtered history"
            );
            Some(filtered_history)
        } else {
            // Not a pending participant
            None
        }
    }

    /// Prepare common context for broadcast-based execution.
    ///
    /// This method collects all shared data needed for spawning agent tasks:
    /// 1. Participant info list (for all participants)
    /// 2. Joining history contexts (for pending participants using JoiningStrategy)
    /// 3. Unsent incoming messages (from external sources)
    /// 4. Unsent agent-generated messages (from other participants)
    /// 5. Message IDs to mark as sent after task completion
    ///
    /// # Usage
    /// - `spawn_broadcast_tasks()`: Uses all participants from the context
    /// - `spawn_mentioned_tasks()`: Filters participants based on @mentions, then uses this context
    ///
    /// # Arguments
    /// * `current_turn` - The current turn number
    ///
    /// # Returns
    /// A `BroadcastContext` containing all prepared data
    fn prepare_broadcast_context(&mut self, current_turn: usize) -> BroadcastContext {
        // Build participant list
        let participants_info = self.get_participants_info();

        // Prepare joining history contexts for all participants
        let mut joining_history_contexts = vec![];
        for idx in 0..self.participants.len() {
            let participant = &self.participants[idx];
            let speaker = participant.to_speaker();
            let joining_history_context = self.join_pending_participant(speaker, current_turn);
            joining_history_contexts.push(joining_history_context);
        }

        // Get unsent agent-generated messages from MessageStore
        let unsent_from_agent: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::AgentGenerated)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect agent message IDs to mark as sent later
        let mut message_ids_to_mark: Vec<_> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::AgentGenerated)
            .iter()
            .map(|msg| msg.id)
            .collect();

        // Get unsent incoming messages (from external sources)
        let unsent_incoming: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect incoming message IDs to mark as sent later
        message_ids_to_mark.extend(
            self.message_store
                .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
                .iter()
                .map(|msg| msg.id),
        );

        BroadcastContext {
            participants_info,
            joining_history_contexts,
            unsent_incoming,
            unsent_from_agent,
            message_ids_to_mark,
        }
    }

    /// Runs the dialogue with the configured execution model.
    ///
    /// The behavior depends on the execution model:
    /// - **Broadcast**: All participants respond in parallel to the full conversation history.
    ///   Returns a vector of dialogue turns from all participants.
    /// - **Sequential**: Participants execute in order, with each agent's output becoming the
    ///   next agent's input. Returns a vector containing only the final agent's turn.
    ///
    /// This method accepts any type that can be converted into a `Payload`, including:
    /// - `String` or `&str` for text-only input
    /// - `Payload` for multimodal input with attachments
    ///
    /// # Arguments
    ///
    /// * `initial_prompt` - The prompt to start the dialogue
    ///
    /// # Returns
    ///
    /// A vector of `DialogueTurn` containing participant names and their responses.
    /// The structure depends on the execution model:
    /// - Broadcast: Contains turns from all participants
    /// - Sequential: Contains only the final participant's turn
    ///
    /// # Errors
    ///
    /// Returns `AgentError` if any agent fails during execution.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// // Broadcast mode with text
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    /// let all_turns = dialogue.run("Discuss AI ethics").await?;
    ///
    /// // Sequential mode with multimodal input
    /// let mut dialogue = Dialogue::sequential()
    ///     .add_participant(persona1, summarizer)
    ///     .add_participant(persona2, translator);
    /// let payload = Payload::text("Process this image")
    ///     .with_attachment(Attachment::local("image.png"));
    /// let final_turn = dialogue.run(payload).await?;
    /// ```
    #[crate::tracing::instrument(
        name = "dialogue.run",
        skip(self, initial_prompt),
        fields(
            execution_model = ?self.execution_model,
            participants_count = self.participants.len(),
        )
    )]
    pub async fn run(
        &mut self,
        initial_prompt: impl Into<Payload>,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        let payload = initial_prompt.into();
        let current_turn = self.next_turn();
        // Store incoming payload for history/unsent tracking
        let (stored_prompt, _) = self.store_payload_messages(&payload, current_turn);

        // Check if agents should react to this message
        if !self.should_react(&payload) {
            crate::tracing::trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                stored_prompt = stored_prompt.len(),
                "Starting run passed no react"
            );
            return Ok(vec![]);
        }

        crate::tracing::trace!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            stored_prompt = stored_prompt.len(),
            execution_model = ?self.execution_model,
            participant_count = self.participants.len(),
            "Starting run"
        );

        // Use new implementation for both modes
        // Note: no_react_messages will be prepended by each execution mode
        match self.execution_model.clone() {
            ExecutionModel::Sequential => {
                // Sequential with default AsAdded order
                self.run_sequential(current_turn, &SequentialOrder::AsAdded)
                    .await
            }
            ExecutionModel::OrderedSequential(order) => {
                self.run_sequential(current_turn, &order).await
            }
            ExecutionModel::Broadcast => {
                // Broadcast with default Completion order
                self.run_broadcast(current_turn, BroadcastOrder::Completion)
                    .await
            }
            ExecutionModel::OrderedBroadcast(order) => {
                self.run_broadcast(current_turn, order).await
            }
            ExecutionModel::Mentioned { strategy } => {
                self.run_mentioned(current_turn, strategy).await
            }
            ExecutionModel::Moderator => {
                // Consult moderator for execution strategy
                self.run_with_moderator(current_turn, payload).await
            }
        }
    }

    /// Moderator-driven execution.
    ///
    /// Consults the moderator agent to determine the execution model for this turn,
    /// then delegates to the appropriate execution method.
    async fn run_with_moderator(
        &mut self,
        current_turn: usize,
        payload: Payload,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        debug!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            execution_model = "moderator",
            participant_count = self.participants.len(),
            "Consulting moderator for execution strategy"
        );

        // Get moderator agent
        let moderator = self.moderator.as_ref().ok_or_else(|| {
            AgentError::ExecutionFailed(
                "ExecutionModel::Moderator requires a moderator agent. Use with_moderator() to set one.".to_string(),
            )
        })?;

        // Build context for moderator decision
        let moderator_context = self.build_moderator_context(&payload, current_turn);

        // Consult moderator
        let decided_model = moderator.execute(moderator_context).await?;

        debug!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            decided_model = ?decided_model,
            "Moderator decided execution strategy"
        );

        // Execute with the decided model
        match decided_model {
            ExecutionModel::Sequential => {
                // Sequential with default AsAdded order
                self.run_sequential(current_turn, &SequentialOrder::AsAdded)
                    .await
            }
            ExecutionModel::OrderedSequential(order) => {
                self.run_sequential(current_turn, &order).await
            }
            ExecutionModel::Broadcast => {
                // Broadcast with default Completion order
                self.run_broadcast(current_turn, BroadcastOrder::Completion)
                    .await
            }
            ExecutionModel::OrderedBroadcast(order) => {
                self.run_broadcast(current_turn, order).await
            }
            ExecutionModel::Mentioned { strategy } => {
                self.run_mentioned(current_turn, strategy).await
            }
            ExecutionModel::Moderator => {
                // Prevent infinite recursion
                Err(AgentError::ExecutionFailed(
                    "Moderator cannot return Moderator execution model (infinite recursion)"
                        .to_string(),
                ))
            }
        }
    }

    /// Builds context for moderator decision-making.
    ///
    /// Includes conversation history, participant info, and current payload.
    fn build_moderator_context(&self, payload: &Payload, current_turn: usize) -> Payload {
        let mut context_messages = Vec::new();

        // Add conversation history summary
        if current_turn > 1 {
            let history = self.history();
            let history_text = format!(
                "Conversation history ({} previous turns):\n{}",
                history.len(),
                history
                    .iter()
                    .map(|turn| format!("[{}]: {}", turn.speaker.name(), turn.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            context_messages.push(PayloadMessage::new(Speaker::System, history_text));
        }

        // Add participant information
        let participants_info = self
            .participants
            .iter()
            .map(|p| {
                format!(
                    "- {} ({}): {}",
                    p.persona.name, p.persona.role, p.persona.background
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        context_messages.push(PayloadMessage::new(
            Speaker::System,
            format!("Available participants:\n{}", participants_info),
        ));

        // Add current payload
        for msg in payload.to_messages() {
            context_messages.push(msg);
        }

        Payload::from_messages(context_messages)
    }

    /// New broadcast implementation using MessageStore and TurnInput.
    async fn run_broadcast(
        &mut self,
        current_turn: usize,
        _order: BroadcastOrder,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        debug!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            execution_model = "broadcast",
            participant_count = self.participants.len(),
            has_context = self.context.is_some(),
            "Starting dialogue.run() in broadcast mode"
        );

        // Spawn broadcast tasks using helper method
        let mut pending = self.spawn_broadcast_tasks(current_turn);

        // Collect responses and create message entities
        let mut dialogue_turns = Vec::new();

        while let Some(Ok((idx, _name, result))) = pending.join_next().await {
            match result {
                Ok(content) => {
                    // Store response message
                    let speaker = self.participants[idx].to_speaker();
                    let metadata =
                        MessageMetadata::new().with_origin(MessageOrigin::AgentGenerated);
                    let response_message =
                        DialogueMessage::new(current_turn, speaker.clone(), content.clone())
                            .with_metadata(&metadata);
                    self.message_store.push(response_message);

                    // Create DialogueTurn for backward compatibility
                    dialogue_turns.push(DialogueTurn { speaker, content });
                }
                Err(err) => return Err(err),
            }
        }

        Ok(dialogue_turns)
    }

    /// New sequential implementation using MessageStore.
    ///
    /// In Sequential mode, each agent's output becomes the next agent's input.
    /// Only the final agent's response is returned.
    async fn run_sequential(
        &mut self,
        current_turn: usize,
        order: &SequentialOrder,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        debug!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            execution_model = "sequential",
            participant_count = self.participants.len(),
            has_context = self.context.is_some(),
            "Starting dialogue.run() in sequential mode"
        );

        // Build participant list
        let participants_info = self.get_participants_info();

        // Get unsent incoming messages (for first agent)
        let unsent_messages_incoming: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect message IDs to mark as sent (for first agent)
        let incoming_message_ids: Vec<_> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
            .iter()
            .map(|msg| msg.id)
            .collect();

        if !unsent_messages_incoming.is_empty() {
            trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                incoming_message_count = unsent_messages_incoming.len(),
                "Sequential mode: First agent will receive {} incoming messages",
                unsent_messages_incoming.len()
            );
        }

        let sequence_indices = self.resolve_sequential_indices(order)?;
        // Execute participants sequentially
        let mut final_turn = None;

        for (sequence_idx, participant_idx) in sequence_indices.iter().enumerate() {
            let participant_idx = *participant_idx;

            // Handle pending participant: apply JoiningStrategy and mark history as sent
            // Extract necessary data in a scope to release the borrow before calling join_pending_participant
            let joining_history_context = {
                let speaker = {
                    let participant = &self.participants[participant_idx];
                    participant.to_speaker()
                };
                self.join_pending_participant(speaker, current_turn)
            };

            // Get participant reference for the rest of the logic
            let participant = &self.participants[participant_idx];
            let agent = &participant.agent;
            let agent_name = participant.name().to_string();

            // Determine input messages based on position in sequence
            let (current_messages, messages_with_metadata, message_ids_to_mark) = if sequence_idx
                == 0
            {
                // First agent: combine previous turn's agent messages + unsent incoming messages
                let mut messages = Vec::new();
                let mut metadata_messages = Vec::new();

                // Add previous turn's agent messages if in multi-turn scenario
                if current_turn > 1 {
                    let prev_turn_messages: Vec<PayloadMessage> = self
                        .message_store
                        .messages_for_turn(current_turn - 1)
                        .into_iter()
                        .filter(|msg| matches!(msg.speaker, Speaker::Agent { .. }))
                        .map(PayloadMessage::from)
                        .collect();

                    trace!(
                        target = "llm_toolkit::dialogue",
                        turn = current_turn,
                        agent_idx = participant_idx,
                        sequence_idx,
                        agent_name = %agent_name,
                        prev_turn_message_count = prev_turn_messages.len(),
                        "Sequential mode: First agent receiving {} previous turn agent messages",
                        prev_turn_messages.len()
                    );

                    messages.extend(prev_turn_messages.clone());
                    metadata_messages.extend(prev_turn_messages);
                }

                // Add unsent incoming messages
                if let Some(context) = joining_history_context {
                    messages.extend(context)
                }
                messages.extend(unsent_messages_incoming.clone());
                metadata_messages.extend(unsent_messages_incoming.clone());

                // Don't mark incoming messages as sent yet - subsequent agents need to see them
                (messages, metadata_messages, vec![])
            } else {
                // Subsequent agents: get ALL previous agents' messages from current turn
                // (not just unsent, as they may have been marked sent by earlier agents in the chain)
                // Note: In sequential mode, current turn's agent messages are REQUIRED for the chain,
                // so we always include them regardless of joining_strategy
                let prev_agent_messages: Vec<PayloadMessage> = self
                    .message_store
                    .messages_for_turn(current_turn)
                    .into_iter()
                    .filter(|msg| matches!(msg.speaker, Speaker::Agent { .. }))
                    .map(PayloadMessage::from)
                    .collect();

                // Combine: previous agents' outputs + unsent incoming messages
                let mut messages: Vec<PayloadMessage> = joining_history_context.unwrap_or_default();
                messages.extend(prev_agent_messages.clone());
                messages.extend(unsent_messages_incoming.clone());

                let mut metadata_messages = prev_agent_messages.clone();
                metadata_messages.extend(unsent_messages_incoming.clone());

                trace!(
                    target = "llm_toolkit::dialogue",
                    turn = current_turn,
                        agent_idx = participant_idx,
                        sequence_idx,
                    agent_name = %agent_name,
                    prev_message_count = prev_agent_messages.len(),
                    incoming_message_count = unsent_messages_incoming.len(),
                    "Sequential mode: Agent {} receiving {} previous agent messages + {} incoming messages",
                    agent_name,
                    prev_agent_messages.len(),
                    unsent_messages_incoming.len()
                );

                // No need to mark previous agent messages as sent - they're already in the store
                (messages, metadata_messages, vec![])
            };

            // Build payload using TurnInput
            let turn_input = TurnInput::with_messages_and_context(
                current_messages,
                vec![], // context is integrated into messages
                participants_info.clone(),
                agent_name.clone(),
            );

            let messages = turn_input.to_messages();
            let mut input_payload = Payload::from_messages(messages);

            // Attach context if exists
            if let Some(ref context) = self.context {
                input_payload = input_payload.with_context(context.to_prompt());
            }

            // Apply metadata attachments
            input_payload =
                Self::apply_metadata_attachments(input_payload, &messages_with_metadata);

            // Add participants info
            input_payload = input_payload.with_participants(participants_info.clone());

            // Execute agent
            let response = agent.execute(input_payload).await?;

            // Store response message
            let speaker = participant.to_speaker();
            let metadata = MessageMetadata::new().with_origin(MessageOrigin::AgentGenerated);
            let response_message =
                DialogueMessage::new(current_turn, speaker.clone(), response.clone())
                    .with_metadata(&metadata);
            self.message_store.push(response_message);

            // Mark input messages as sent (after this agent has processed them)
            if !message_ids_to_mark.is_empty() {
                self.message_store.mark_all_as_sent(&message_ids_to_mark);
                trace!(
                    target = "llm_toolkit::dialogue",
                    turn = current_turn,
                    agent_idx = participant_idx,
                    sequence_idx,
                    agent_name = %agent_name,
                    marked_sent_count = message_ids_to_mark.len(),
                    "Marked {} messages as sent after agent {} execution",
                    message_ids_to_mark.len(),
                    agent_name
                );
            }

            // Keep track of final turn
            final_turn = Some(DialogueTurn {
                speaker,
                content: response,
            });
        }

        // Mark incoming messages as sent after all agents have processed them
        if !incoming_message_ids.is_empty() {
            self.message_store.mark_all_as_sent(&incoming_message_ids);
            trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                marked_sent_count = incoming_message_ids.len(),
                "Sequential mode: Marked {} incoming messages as sent after all agents",
                incoming_message_ids.len()
            );
        }

        // Return only the final turn
        Ok(final_turn.into_iter().collect())
    }

    /// New mentioned implementation using MessageStore and TurnInput.
    ///
    /// In Mentioned mode, only @mentioned participants respond. If no mentions are found,
    /// it falls back to broadcast mode (all participants respond).
    async fn run_mentioned(
        &mut self,
        current_turn: usize,
        strategy: MentionMatchStrategy,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        debug!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            execution_model = "mentioned",
            participant_count = self.participants.len(),
            has_context = self.context.is_some(),
            "Starting dialogue.run() in mentioned mode"
        );

        // Spawn tasks for mentioned participants (or all if no mentions)
        let mut pending = self.spawn_mentioned_tasks(current_turn, strategy);

        // Collect responses and create message entities
        let mut dialogue_turns = Vec::new();

        while let Some(Ok((idx, _name, result))) = pending.join_next().await {
            match result {
                Ok(content) => {
                    // Store response message
                    let speaker = self.participants[idx].to_speaker();
                    let metadata =
                        MessageMetadata::new().with_origin(MessageOrigin::AgentGenerated);
                    let response_message =
                        DialogueMessage::new(current_turn, speaker.clone(), content.clone())
                            .with_metadata(&metadata);
                    self.message_store.push(response_message);

                    // Create DialogueTurn for backward compatibility
                    dialogue_turns.push(DialogueTurn { speaker, content });
                }
                Err(err) => return Err(err),
            }
        }

        Ok(dialogue_turns)
    }

    /// Begins a dialogue session that yields turns incrementally.
    ///
    /// This method accepts any type that can be converted into a `Payload`, including:
    /// - `String` or `&str` for text-only input
    /// - `Payload` for multimodal input with attachments
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Text-only input (backward compatible)
    /// let mut session = dialogue.partial_session("Hello");
    ///
    /// // Multimodal input with attachment
    /// let payload = Payload::text("What's in this image?")
    ///     .with_attachment(Attachment::local("image.png"));
    /// let mut session = dialogue.partial_session(payload);
    /// ```
    pub fn partial_session(&mut self, initial_prompt: impl Into<Payload>) -> DialogueSession<'_> {
        // Don't override execution model - use the configured model
        self.partial_session_internal(initial_prompt, None)
    }

    /// Begins a dialogue session with a specified broadcast ordering policy.
    ///
    /// This method accepts any type that can be converted into a `Payload`, including:
    /// - `String` or `&str` for text-only input
    /// - `Payload` for multimodal input with attachments
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // With text and custom order
    /// let mut session = dialogue.partial_session_with_order(
    ///     "Hello",
    ///     BroadcastOrder::ParticipantOrder
    /// );
    ///
    /// // With multimodal payload and custom order
    /// let payload = Payload::text("Analyze this")
    ///     .with_attachment(Attachment::local("data.csv"));
    /// let mut session = dialogue.partial_session_with_order(
    ///     payload,
    ///     BroadcastOrder::ParticipantOrder
    /// );
    /// ```
    pub fn partial_session_with_order(
        &mut self,
        initial_prompt: impl Into<Payload>,
        broadcast_order: BroadcastOrder,
    ) -> DialogueSession<'_> {
        self.partial_session_internal(initial_prompt, Some(broadcast_order))
    }

    /// Internal implementation for partial_session with optional order override.
    fn partial_session_internal(
        &mut self,
        initial_prompt: impl Into<Payload>,
        broadcast_order_override: Option<BroadcastOrder>,
    ) -> DialogueSession<'_> {
        // Temporarily override execution model if broadcast order is specified
        let original_model = self.execution_model.clone();
        if let Some(broadcast_order) = broadcast_order_override {
            self.execution_model = ExecutionModel::OrderedBroadcast(broadcast_order);
        }

        let payload: Payload = initial_prompt.into();
        let current_turn = self.next_turn();

        // Store incoming payload for history/unsent tracking
        let (stored_prompt, _) = self.store_payload_messages(&payload, current_turn);

        // Check if agents should react to this message
        if !self.should_react(&payload) {
            crate::tracing::trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                participant_count = self.participants.len(),
                "Starting partial_session passed no react"
            );
            // Return a completed session (no agent reactions)
            // Restore original execution model
            let model = self.execution_model.clone();
            self.execution_model = original_model;
            return DialogueSession {
                dialogue: self,
                state: SessionState::Completed,
                model,
            };
        }

        // Store all incoming messages in MessageStore for Dialogue history management.
        // This is Dialogue's responsibility: maintain complete conversation history
        // regardless of ExecutionModel. Each model may also access Payload directly
        // during execution, but MessageStore serves as the authoritative, persistent record
        // for audit, debugging, and history retrieval (e.g., unsent messages, prev turns).
        //
        trace!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            prompt_length = stored_prompt.len(),
            total_store_size = self.message_store.len(),
            "Stored incoming payload in MessageStore"
        );

        let model = self.execution_model.clone();
        let state = match &model {
            ExecutionModel::Sequential => {
                // Sequential with default AsAdded order
                let participants_info = self.get_participants_info();

                let prev_agent_outputs: Vec<PayloadMessage> = if current_turn > 1 {
                    self.message_store
                        .messages_for_turn(current_turn - 1)
                        .into_iter()
                        .filter(|msg| matches!(msg.speaker, Speaker::Agent { .. }))
                        .map(PayloadMessage::from)
                        .collect()
                } else {
                    Vec::new()
                };

                match self.resolve_sequential_indices(&SequentialOrder::AsAdded) {
                    Ok(sequence) => SessionState::Sequential {
                        next_index: 0,
                        current_turn,
                        sequence,
                        payload,
                        prev_agent_outputs,
                        current_turn_outputs: Vec::new(),
                        participants_info,
                    },
                    Err(err) => {
                        error!(
                            target = "llm_toolkit::dialogue",
                            turn = current_turn,
                            execution_model = "sequential",
                            participant_count = self.participants.len(),
                            error = %err,
                            "Failed to resolve sequential order for partial session"
                        );
                        SessionState::Failed(Some(err))
                    }
                }
            }
            ExecutionModel::OrderedSequential(order) => {
                let participants_info = self.get_participants_info();

                let prev_agent_outputs: Vec<PayloadMessage> = if current_turn > 1 {
                    self.message_store
                        .messages_for_turn(current_turn - 1)
                        .into_iter()
                        .filter(|msg| matches!(msg.speaker, Speaker::Agent { .. }))
                        .map(PayloadMessage::from)
                        .collect()
                } else {
                    Vec::new()
                };

                match self.resolve_sequential_indices(order) {
                    Ok(sequence) => SessionState::Sequential {
                        next_index: 0,
                        current_turn,
                        sequence,
                        payload,
                        prev_agent_outputs,
                        current_turn_outputs: Vec::new(),
                        participants_info,
                    },
                    Err(err) => {
                        error!(
                            target = "llm_toolkit::dialogue",
                            turn = current_turn,
                            execution_model = "sequential",
                            participant_count = self.participants.len(),
                            error = %err,
                            "Failed to resolve sequential order for partial session"
                        );
                        SessionState::Failed(Some(err))
                    }
                }
            }
            ExecutionModel::Broadcast => {
                // Broadcast with default Completion order
                let pending = self.spawn_broadcast_tasks(current_turn);

                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    BroadcastOrder::Completion,
                    self.participants.len(),
                    current_turn,
                ))
            }
            ExecutionModel::OrderedBroadcast(order) => {
                // Spawn broadcast tasks using helper method
                let pending = self.spawn_broadcast_tasks(current_turn);

                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    order.clone(),
                    self.participants.len(),
                    current_turn,
                ))
            }
            ExecutionModel::Mentioned { strategy } => {
                // For Mentioned mode, spawn tasks for mentioned participants only
                let pending = self.spawn_mentioned_tasks(current_turn, *strategy);

                // Mentioned mode uses Broadcast state with Completion order
                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    BroadcastOrder::Completion,
                    self.participants.len(),
                    current_turn,
                ))
            }
            ExecutionModel::Moderator => {
                // For Moderator mode, we need to consult the moderator first
                // This is not supported in partial_session yet - use run() instead
                error!(
                    target = "llm_toolkit::dialogue",
                    "Moderator mode is not supported in partial_session, use run() instead"
                );
                SessionState::Failed(Some(AgentError::ExecutionFailed(
                    "Moderator mode requires run() method, not partial_session()".to_string(),
                )))
            }
        };

        DialogueSession {
            dialogue: self,
            state,
            model,
        }
    }

    /// Helper method to spawn broadcast tasks for all participants.
    ///
    /// Returns a JoinSet with pending agent executions.
    pub(super) fn spawn_broadcast_tasks(
        &mut self,
        current_turn: usize,
    ) -> JoinSet<(usize, String, Result<String, AgentError>)> {
        // Prepare broadcast context (participant info, joining histories, unsent messages)
        let ctx = self.prepare_broadcast_context(current_turn);

        let mut pending = JoinSet::new();

        for idx in 0..self.participants.len() {
            let participant: &Participant = &self.participants[idx];
            let joining_history_context = ctx.joining_history_contexts.get(idx);
            let agent: Arc<crate::AnyAgent<String>> = Arc::clone(&participant.agent);
            let participant_name = participant.name().to_string();

            let mut current_messages: Vec<PayloadMessage> = vec![];
            if let Some(Some(context)) = joining_history_context {
                current_messages.extend_from_slice(context)
            }

            // Regular participant: use unsent_from_agent
            let unsent_payload_messages: Vec<PayloadMessage> = ctx
                .unsent_from_agent
                .iter()
                .filter(|msg| msg.speaker.name() != participant_name)
                .cloned()
                .collect();

            current_messages.extend(unsent_payload_messages);

            current_messages.extend(ctx.unsent_incoming.clone());
            let messages_with_metadata = current_messages.clone();

            // Create payload with turn input formatting
            let turn_input = TurnInput::with_messages_and_context(
                current_messages,
                vec![],
                ctx.participants_info.clone(),
                participant_name.clone(),
            );

            let messages = turn_input.to_messages();
            let mut payload = Payload::from_messages(messages);

            // Attach context if exists
            if let Some(ref context) = self.context {
                payload = payload.with_context(context.to_prompt());
            }

            payload = Self::apply_metadata_attachments(payload, &messages_with_metadata);
            let input_payload = payload.with_participants(ctx.participants_info.clone());

            pending.spawn(async move {
                let result = agent.execute(input_payload).await;
                (idx, participant_name, result)
            });
        }

        // Mark unsent messages as sent to agents
        self.message_store
            .mark_all_as_sent(&ctx.message_ids_to_mark);

        if !ctx.message_ids_to_mark.is_empty() {
            trace!(
                target = "llm_toolkit::dialogue",
                marked_sent_count = ctx.message_ids_to_mark.len(),
                "Marked messages as sent_to_agents in MessageStore"
            );
        }

        pending
    }

    /// Helper method to spawn tasks for mentioned participants only.
    ///
    /// Extracts @mentions from unsent messages and spawns tasks only for those participants.
    /// If no mentions are found, falls back to spawning tasks for all participants (broadcast).
    ///
    /// Returns a JoinSet with pending agent executions.
    pub(super) fn spawn_mentioned_tasks(
        &mut self,
        current_turn: usize,
        strategy: MentionMatchStrategy,
    ) -> JoinSet<(usize, String, Result<String, AgentError>)> {
        // Get unsent incoming messages (for mention extraction and agent input)
        let unsent_messages_incoming: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect incoming message IDs to mark as sent later
        let incoming_message_ids: Vec<_> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
            .iter()
            .map(|msg| msg.id)
            .collect();

        // Get unsent agent-generated messages (for mention extraction and context)
        let unsent_messages_from_agent: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::AgentGenerated)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect agent message IDs to mark as sent later
        let agent_message_ids: Vec<_> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::AgentGenerated)
            .iter()
            .map(|msg| msg.id)
            .collect();

        trace!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            incoming_count = unsent_messages_incoming.len(),
            agent_count = unsent_messages_from_agent.len(),
            "Retrieved unsent messages from MessageStore for mention extraction"
        );

        // Extract text from unsent messages to find mentions
        // Check both incoming messages and agent messages for mentions
        let mentions_text = {
            let incoming_text = unsent_messages_incoming
                .iter()
                .map(|msg| msg.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let agent_text = unsent_messages_from_agent
                .iter()
                .map(|msg| msg.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            format!("{}\n{}", incoming_text, agent_text)
        };

        // Get all participant names (cloned to avoid borrow conflict with iter_mut later)
        let participant_names: Vec<String> = self
            .participants
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        let participant_name_refs: Vec<&str> =
            participant_names.iter().map(|s| s.as_str()).collect();

        // Extract mentions from the text using the specified strategy
        let mentioned_names =
            extract_mentions_with_strategy(&mentions_text, &participant_name_refs, strategy);

        trace!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            mentions_text_preview = &mentions_text[..mentions_text.len().min(100)],
            incoming_message_count = unsent_messages_incoming.len(),
            agent_message_count = unsent_messages_from_agent.len(),
            all_participants = ?participant_name_refs,
            mentioned = ?mentioned_names,
            "Extracting mentions for Mentioned execution mode"
        );

        // If no mentions found, fall back to broadcast mode (all participants)
        let target_participants: Vec<&str> = if mentioned_names.is_empty() {
            debug!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                "No mentions found, falling back to broadcast mode"
            );
            participant_name_refs
        } else {
            trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                mentioned_count = mentioned_names.len(),
                mentioned_participants = ?mentioned_names,
                "Mentions detected - executing selective participants"
            );
            mentioned_names
        };

        // Build participant list
        let participants_info = self.get_participants_info();

        // Log execution plan summary
        let executing_participants: Vec<_> = self
            .participants
            .iter()
            .filter(|p| target_participants.contains(&p.name()))
            .map(|p| p.name())
            .collect();

        let skipped_participants: Vec<_> = self
            .participants
            .iter()
            .filter(|p| !target_participants.contains(&p.name()))
            .map(|p| p.name())
            .collect();

        trace!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            executing_count = executing_participants.len(),
            executing_participants = ?executing_participants,
            skipped_count = skipped_participants.len(),
            skipped_participants = ?skipped_participants,
            "Mention-based execution plan determined"
        );

        // Collect indices of mentioned participants
        let mentioned_indices: Vec<usize> = self
            .participants
            .iter()
            .enumerate()
            .filter(|(_, p)| target_participants.contains(&p.name()))
            .map(|(idx, _)| idx)
            .collect();

        // Prepare joining history contexts for mentioned participants only
        let mut joining_history_contexts = vec![];
        for &idx in &mentioned_indices {
            let participant = &self.participants[idx];
            let speaker = participant.to_speaker();
            let joining_history_context = self.join_pending_participant(speaker, current_turn);
            joining_history_contexts.push(joining_history_context);
        }

        let mut pending = JoinSet::new();

        // Spawn tasks for mentioned participants only
        for (i, &idx) in mentioned_indices.iter().enumerate() {
            let participant = &self.participants[idx];
            let participant_name = participant.name().to_string();
            let agent = Arc::clone(&participant.agent);

            // Get joining history for this participant
            let joining_history_context = &joining_history_contexts[i];

            let mut current_messages: Vec<PayloadMessage> = vec![];
            if let Some(context) = joining_history_context {
                current_messages.extend(context.clone());
            }

            current_messages.extend(unsent_messages_incoming.clone());
            let messages_with_metadata = current_messages.clone();

            let turn_input = TurnInput::with_messages_and_context(
                current_messages.clone(),
                vec![],
                participants_info.clone(),
                participant_name.clone(),
            );

            let messages = turn_input.to_messages();
            let mut payload = Payload::from_messages(messages);

            if let Some(ref context) = self.context {
                payload = payload.with_context(context.to_prompt());
            }

            payload = Self::apply_metadata_attachments(payload, &messages_with_metadata);
            let input_payload = payload.with_participants(participants_info.clone());

            trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                participant = %participant_name,
                "Spawning task for mentioned participant"
            );

            pending.spawn(async move {
                let result = agent.execute(input_payload).await;
                (idx, participant_name, result)
            });
        }

        // Mark all unsent messages as sent to agents (including both agent and incoming messages)
        let mut all_message_ids = agent_message_ids;
        all_message_ids.extend(incoming_message_ids);

        self.message_store.mark_all_as_sent(&all_message_ids);

        if !all_message_ids.is_empty() {
            trace!(
                target = "llm_toolkit::dialogue",
                marked_sent_count = all_message_ids.len(),
                "Marked messages as sent_to_agents in MessageStore"
            );
        }

        pending
    }

    /// Formats the conversation history as a single string.
    ///
    /// This creates a formatted transcript of the dialogue that can be used
    /// as input for the next agent.
    #[cfg(test)]
    pub(crate) fn format_history(&self) -> String {
        self.history()
            .iter()
            .map(|turn| format!("[{}]: {}", turn.speaker.name(), turn.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extracts messages from a Payload and converts them to DialogueMessages.
    ///
    /// Returns a tuple of (messages, prompt_text) where:
    /// - messages: Vec of DialogueMessages to store
    /// - prompt_text: Combined text for agent execution
    fn extract_messages_from_payload(
        &self,
        payload: &Payload,
        turn: usize,
    ) -> (Vec<DialogueMessage>, String) {
        use crate::agent::PayloadContent;

        let mut messages = Vec::new();
        let mut text_parts = Vec::new();

        for content in payload.contents() {
            match content {
                PayloadContent::Message {
                    speaker,
                    content,
                    metadata,
                } => {
                    // Store as individual message with explicit speaker and metadata
                    let metadata = metadata
                        .clone()
                        .ensure_origin(MessageOrigin::IncomingPayload);
                    messages.push(
                        DialogueMessage::new(turn, speaker.clone(), content.clone())
                            .with_metadata(&metadata),
                    );
                    text_parts.push(content.as_str());
                }
                PayloadContent::Text(text) => {
                    // Text without explicit speaker is treated as User input
                    // TODO: Allow configuring default speaker (User vs System)
                    let metadata =
                        MessageMetadata::new().with_origin(MessageOrigin::IncomingPayload);
                    messages.push(
                        DialogueMessage::new(
                            turn,
                            Speaker::System, // For backward compatibility, treat as System
                            text.clone(),
                        )
                        .with_metadata(&metadata),
                    );
                    text_parts.push(text.as_str());
                }
                PayloadContent::Attachment(_)
                | PayloadContent::Participants(_)
                | PayloadContent::Document(_)
                | PayloadContent::Context(_) => {
                    // Attachments, Participants metadata, Documents, and Context don't create messages, just pass through
                }
            }
        }

        // If no messages were extracted, create a default System message
        if messages.is_empty() {
            let prompt_text = payload.to_text();
            messages.push(DialogueMessage::new(
                turn,
                Speaker::System,
                prompt_text.clone(),
            ));
            return (messages, prompt_text);
        }

        // Attach payload-level attachments to the first message for downstream retrieval.
        let attachments: Vec<_> = payload.attachments().into_iter().cloned().collect();
        if !attachments.is_empty() {
            if let Some(first_msg) = messages.first_mut() {
                let metadata = first_msg.metadata.clone().with_attachments(attachments);
                first_msg.metadata = metadata;
            } else {
                let metadata = MessageMetadata::new()
                    .with_origin(MessageOrigin::IncomingPayload)
                    .with_attachments(attachments);
                let attachment_message = DialogueMessage::new(turn, Speaker::System, String::new())
                    .with_metadata(&metadata);
                messages.push(attachment_message);
            }
        }

        let prompt_text = text_parts.join("\n");
        (messages, prompt_text)
    }

    /// Returns a reference to the conversation history.
    ///
    /// # Note
    ///
    /// This creates DialogueTurns on-the-fly from the MessageStore for backward compatibility.
    /// For new code, consider using `message_store()` directly.
    pub fn history(&self) -> Vec<DialogueTurn> {
        self.message_store
            .all_messages()
            .into_iter()
            .map(|msg| DialogueTurn {
                speaker: msg.speaker.clone(),
                content: msg.content.clone(),
            })
            .collect()
    }

    /// Returns a reference to the message store (new API).
    pub fn message_store(&self) -> &MessageStore {
        &self.message_store
    }

    /// Returns references to the personas of all participants.
    ///
    /// This provides access to participant information such as names, roles,
    /// backgrounds, and communication styles without exposing internal agent implementations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    ///
    /// let personas = dialogue.participants();
    /// for persona in personas {
    ///     println!("Participant: {} ({})", persona.name, persona.role);
    /// }
    /// ```
    pub fn participants(&self) -> Vec<&Persona> {
        self.participants.iter().map(|p| &p.persona).collect()
    }

    /// Returns the number of participants in the dialogue.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Saves conversation history to a JSON file.
    ///
    /// This is useful for session persistence, allowing you to save the dialogue
    /// state and resume it later.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the history will be saved
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an `AgentError` if serialization or file
    /// writing fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1);
    ///
    /// let turns = dialogue.run("Discuss architecture").await?;
    ///
    /// // Save history for later resumption
    /// dialogue.save_history("session_123.json")?;
    /// ```
    pub fn save_history(&self, path: impl AsRef<std::path::Path>) -> Result<(), AgentError> {
        let history_to_save = self.history(); // Use the method to get DialogueTurns
        let json = serde_json::to_string_pretty(&history_to_save).map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to serialize history: {}", e))
        })?;
        std::fs::write(path, json).map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to write history file: {}", e))
        })?;
        Ok(())
    }

    /// Loads conversation history from a JSON file.
    ///
    /// Use this to restore a saved conversation history, typically in combination
    /// with `with_history()` to resume a dialogue session.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to load the history from
    ///
    /// # Returns
    ///
    /// Returns a vector of `DialogueTurn` on success, or an `AgentError` if
    /// file reading or deserialization fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// // Load saved history
    /// let history = Dialogue::load_history("session_123.json")?;
    ///
    /// // Resume dialogue with loaded history
    /// let mut dialogue = Dialogue::broadcast()
    ///     .with_history(history)
    ///     .add_participant(persona1, agent1);
    ///
    /// let turns = dialogue.run("Continue discussion").await?;
    /// ```
    pub fn load_history(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to read history file: {}", e))
        })?;
        serde_json::from_str(&json).map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to deserialize history: {}", e))
        })
    }
}

// ============================================================================
// Agent Trait Implementation
// ============================================================================

/// Implements the Agent trait for Dialogue, enabling it to be used in Orchestrators.
///
/// This implementation allows Dialogue (multi-agent conversations) to be registered
/// and executed as a single agent in orchestration workflows, enabling flexible
/// composition: Dialogue → Orchestrator or Orchestrator → Dialogue.
///
/// # Output Type
///
/// The output type is `Vec<DialogueTurn>`, which contains all turns from the conversation.
/// Each turn includes the speaker and their contribution.
///
/// # Expertise
///
/// Returns a description of the dialogue's collective capabilities based on
/// execution model and participant count.
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::agent::dialogue::Dialogue;
/// use llm_toolkit::agent::AgentAdapter;
/// use llm_toolkit::orchestrator::Orchestrator;
///
/// // Create a dialogue
/// let dialogue = Dialogue::sequential()
///     .add_participant(persona1, agent1)
///     .add_participant(persona2, agent2);
///
/// // Register with orchestrator
/// orchestrator.add_agent(dialogue);
///
/// // Now the orchestrator can use it in StrategyMap like any other agent
/// ```
#[async_trait]
impl Agent for Dialogue {
    type Output = Vec<DialogueTurn>;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        // Note: This returns a static string since Agent::expertise requires &str.
        // The actual capabilities depend on participants, execution model, etc.
        &"Multi-agent dialogue facilitating collaborative conversations with diverse perspectives"
    }

    fn name(&self) -> String {
        // Generate a descriptive name based on execution model and participants
        if self.participants.is_empty() {
            "EmptyDialogue".to_string()
        } else {
            let model_str = match &self.execution_model {
                ExecutionModel::Sequential => "Sequential",
                ExecutionModel::OrderedSequential(_) => "Sequential",
                ExecutionModel::Broadcast => "Broadcast",
                ExecutionModel::OrderedBroadcast(_) => "Broadcast",
                ExecutionModel::Mentioned { .. } => "Mentioned",
                ExecutionModel::Moderator => "Moderator",
            };

            if self.participants.len() == 1 {
                format!(
                    "{}Dialogue({})",
                    model_str, self.participants[0].persona.name
                )
            } else {
                format!(
                    "{}Dialogue({} participants)",
                    model_str,
                    self.participants.len()
                )
            }
        }
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        // Dialogue::run requires &mut self, so we clone to make it mutable.
        // This is acceptable because:
        // 1. Participants contain Arc<dyn Agent>, so cloning is cheap (Arc::clone)
        // 2. MessageStore is cloned, but the orchestrator typically uses fresh dialogues per step
        // 3. This enables stateless execution from the orchestrator's perspective
        let mut dialogue_clone = self.clone();
        dialogue_clone.run(payload).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use tokio::time::{Duration, sleep};

    // Mock agent for testing
    #[derive(Clone)]
    struct MockAgent {
        name: String,
        responses: Vec<String>,
        call_count: std::sync::Arc<std::sync::Mutex<usize>>,
        payloads: std::sync::Arc<std::sync::Mutex<Vec<Payload>>>,
    }

    impl MockAgent {
        fn new(name: impl Into<String>, responses: Vec<String>) -> Self {
            Self {
                name: name.into(),
                responses,
                call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
                payloads: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        fn get_payloads(&self) -> Vec<Payload> {
            self.payloads.lock().unwrap().clone()
        }

        fn get_call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl Agent for MockAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            const EXPERTISE: &str = "Mock agent for testing";
            &EXPERTISE
        }

        fn name(&self) -> String {
            self.name.clone()
        }

        async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
            // Record the payload
            self.payloads.lock().unwrap().push(payload);

            let mut count = self.call_count.lock().unwrap();
            let response_idx = *count % self.responses.len();
            *count += 1;
            Ok(self.responses[response_idx].clone())
        }
    }

    #[tokio::test]
    async fn test_broadcast_strategy() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona1 = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent 1".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Tester".to_string(),
            background: "Test agent 2".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Agent1", vec!["Response 1".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Agent2", vec!["Response 2".to_string()]),
            );

        let turns = dialogue.run("Initial prompt".to_string()).await.unwrap();

        // Should return 2 turns (one from each participant)
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.name(), "Agent1");
        assert_eq!(turns[0].content, "Response 1");
        assert_eq!(turns[1].speaker.name(), "Agent2");
        assert_eq!(turns[1].content, "Response 2");

        // Check history: System + 2 participant responses
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[1].speaker.name(), "Agent1");
        assert_eq!(dialogue.history()[2].speaker.name(), "Agent2");
    }

    #[tokio::test]
    async fn test_dialogue_with_no_participants() {
        let mut dialogue = Dialogue::broadcast();
        // Running with no participants should complete without error
        let result = dialogue.run("Test".to_string()).await;
        assert!(result.is_ok());
        let turns = result.unwrap();
        assert_eq!(turns.len(), 0);
        // Should only have the system message
        assert_eq!(dialogue.history().len(), 1);
    }

    #[tokio::test]
    async fn test_dialogue_format_history() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();
        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(persona, MockAgent::new("Agent1", vec!["Hello".to_string()]));

        dialogue.run("Start".to_string()).await.unwrap();

        let formatted = dialogue.format_history();
        assert!(formatted.contains("[System]: Start"));
        assert!(formatted.contains("[Agent1]: Hello"));
    }

    #[tokio::test]
    async fn test_sequential_strategy() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::sequential();

        let persona1 = Persona {
            name: "Summarizer".to_string(),
            role: "Summarizer".to_string(),
            background: "Summarizes inputs".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Translator".to_string(),
            role: "Translator".to_string(),
            background: "Translates content".to_string(),
            communication_style: "Formal".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona3 = Persona {
            name: "Finalizer".to_string(),
            role: "Finalizer".to_string(),
            background: "Finalizes output".to_string(),
            communication_style: "Professional".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        // Create agents with distinct responses so we can track the chain
        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Summarizer", vec!["Summary: input received".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new(
                    "Translator",
                    vec!["Translated: previous output".to_string()],
                ),
            )
            .add_participant(
                persona3,
                MockAgent::new("Finalizer", vec!["Final output: all done".to_string()]),
            );

        let turns = dialogue.run("Initial prompt".to_string()).await.unwrap();

        // Sequential mode returns only the final turn
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Finalizer");
        assert_eq!(turns[0].content, "Final output: all done");

        // Check that history contains the correct number of turns
        // 1 (System) + 3 (participants) = 4 total
        assert_eq!(dialogue.history().len(), 4);

        // Verify the history structure
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "Initial prompt");

        assert_eq!(dialogue.history()[1].speaker.name(), "Summarizer");
        assert_eq!(dialogue.history()[1].content, "Summary: input received");

        assert_eq!(dialogue.history()[2].speaker.name(), "Translator");
        assert_eq!(dialogue.history()[2].content, "Translated: previous output");

        assert_eq!(dialogue.history()[3].speaker.name(), "Finalizer");
        assert_eq!(dialogue.history()[3].content, "Final output: all done");
    }

    #[tokio::test]
    async fn test_partial_session_sequential_yields_intermediate_turns() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::sequential();

        let persona1 = Persona {
            name: "Step1".to_string(),
            role: "Stage".to_string(),
            background: "first".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Step2".to_string(),
            role: "Stage".to_string(),
            background: "second".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1.clone(),
                MockAgent::new("Step1", vec!["S1 output".to_string()]),
            )
            .add_participant(
                persona2.clone(),
                MockAgent::new("Step2", vec!["S2 output".to_string()]),
            );

        let mut session = dialogue.partial_session("Initial".to_string());
        assert_eq!(
            session.execution_model(),
            ExecutionModel::OrderedSequential(SequentialOrder::AsAdded)
        );

        let first = session.next_turn().await.unwrap().unwrap();
        assert_eq!(first.speaker.name(), "Step1");
        assert_eq!(first.content, "S1 output");

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.speaker.name(), "Step2");
        assert_eq!(second.content, "S2 output");

        assert!(session.next_turn().await.is_none());

        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[1].speaker.name(), "Step1");
        assert_eq!(dialogue.history()[2].speaker.name(), "Step2");
    }

    #[derive(Clone)]
    struct DelayAgent {
        name: String,
        delay_ms: u64,
    }

    impl DelayAgent {
        fn new(name: impl Into<String>, delay_ms: u64) -> Self {
            Self {
                name: name.into(),
                delay_ms,
            }
        }
    }

    #[async_trait]
    impl Agent for DelayAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            const EXPERTISE: &str = "Delayed agent";
            &EXPERTISE
        }

        async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
            sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(format!("{} handled {}", self.name, intent.to_text()))
        }
    }

    #[tokio::test]
    async fn test_partial_session_broadcast_streams_responses() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let fast = Persona {
            name: "Fast".to_string(),
            role: "Fast responder".to_string(),
            background: "Quick replies".to_string(),
            communication_style: "Snappy".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let slow = Persona {
            name: "Slow".to_string(),
            role: "Slow responder".to_string(),
            background: "Takes time".to_string(),
            communication_style: "Measured".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(fast, DelayAgent::new("Fast", 10))
            .add_participant(slow, DelayAgent::new("Slow", 50));

        let mut session = dialogue.partial_session("Hello".to_string());
        assert_eq!(
            session.execution_model(),
            ExecutionModel::OrderedBroadcast(BroadcastOrder::Completion)
        );

        let first = session.next_turn().await.unwrap().unwrap();
        assert_eq!(first.speaker.name(), "Fast");
        assert!(first.content.contains("Fast handled"));

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.speaker.name(), "Slow");
        assert!(second.content.contains("Slow handled"));

        assert!(session.next_turn().await.is_none());
    }

    #[tokio::test]
    async fn test_partial_session_broadcast_ordered_mode_respects_participant_order() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let slow = Persona {
            name: "Slow".to_string(),
            role: "Deliberate responder".to_string(),
            background: "Prefers careful analysis".to_string(),
            communication_style: "Measured".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let fast = Persona {
            name: "Fast".to_string(),
            role: "Quick responder".to_string(),
            background: "Snappy insights".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(slow, DelayAgent::new("Slow", 50))
            .add_participant(fast, DelayAgent::new("Fast", 10));

        let mut session = dialogue
            .partial_session_with_order("Hello".to_string(), BroadcastOrder::ParticipantOrder);

        let first = session.next_turn().await.unwrap().unwrap();
        assert_eq!(first.speaker.name(), "Slow");
        assert!(first.content.contains("Slow handled"));

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.speaker.name(), "Fast");
        assert!(second.content.contains("Fast handled"));

        assert!(session.next_turn().await.is_none());
    }

    #[tokio::test]
    async fn test_from_persona_team_broadcast() {
        use crate::agent::persona::{Persona, PersonaTeam};

        let mut team = PersonaTeam::new("Test Team".to_string(), "Testing scenario".to_string());
        team.add_persona(Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Senior engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        });
        team.add_persona(Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
            visual_identity: None,
            capabilities: None,
        });
        team.execution_strategy =
            Some(ExecutionModel::OrderedBroadcast(BroadcastOrder::Completion));

        let llm = MockAgent::new("Mock", vec!["Response".to_string()]);
        let mut dialogue = Dialogue::from_persona_team(team, llm).unwrap();

        assert_eq!(dialogue.participant_count(), 2);

        let turns = dialogue.run("Test prompt".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2); // Broadcast returns all turns
    }

    #[tokio::test]
    async fn test_from_persona_team_sequential() {
        use crate::agent::persona::{Persona, PersonaTeam};

        let mut team = PersonaTeam::new(
            "Sequential Team".to_string(),
            "Sequential testing".to_string(),
        );
        team.add_persona(Persona {
            name: "First".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        });
        team.add_persona(Persona {
            name: "Second".to_string(),
            role: "Synthesizer".to_string(),
            background: "Content creator".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        });
        team.execution_strategy = Some(ExecutionModel::OrderedSequential(SequentialOrder::AsAdded));

        let llm = MockAgent::new("Mock", vec!["Step output".to_string()]);
        let mut dialogue = Dialogue::from_persona_team(team, llm).unwrap();

        let turns = dialogue.run("Process this".to_string()).await.unwrap();
        assert_eq!(turns.len(), 1); // Sequential returns only final turn
    }

    #[tokio::test]
    async fn test_add_participant_dynamically() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        // Start with one participant
        let initial_persona = Persona {
            name: "Initial".to_string(),
            role: "Initial Agent".to_string(),
            background: "Initial background".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        dialogue.add_participant(
            initial_persona,
            MockAgent::new("Initial", vec!["Response 1".to_string()]),
        );
        assert_eq!(dialogue.participant_count(), 1);

        // Add a participant dynamically
        let expert = Persona {
            name: "Expert".to_string(),
            role: "Domain Expert".to_string(),
            background: "20 years experience".to_string(),
            communication_style: "Authoritative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let llm = MockAgent::new("ExpertLLM", vec!["Expert response".to_string()]);
        dialogue.add_participant(expert, llm);

        assert_eq!(dialogue.participant_count(), 2);

        let turns = dialogue.run("Consult experts".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[tokio::test]
    async fn test_persona_team_round_trip() {
        use crate::agent::persona::{Persona, PersonaTeam};
        use tempfile::NamedTempFile;

        // Create team
        let mut team = PersonaTeam::new(
            "Round Trip Team".to_string(),
            "Testing save/load".to_string(),
        );
        team.add_persona(Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA specialist".to_string(),
            communication_style: "Thorough".to_string(),
            visual_identity: None,
            capabilities: None,
        });

        // Save
        let temp_file = NamedTempFile::new().unwrap();
        team.save(temp_file.path()).unwrap();

        // Load
        let loaded_team = PersonaTeam::load(temp_file.path()).unwrap();

        // Create dialogue from loaded team
        let llm = MockAgent::new("Mock", vec!["Response".to_string()]);
        let dialogue = Dialogue::from_persona_team(loaded_team, llm).unwrap();

        assert_eq!(dialogue.participant_count(), 1);
    }

    #[tokio::test]
    async fn test_remove_participant() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona1 = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        let persona3 = Persona {
            name: "Agent3".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        // Add 3 participants
        dialogue.add_participant(persona1, MockAgent::new("Agent1", vec!["R1".to_string()]));
        dialogue.add_participant(persona2, MockAgent::new("Agent2", vec!["R2".to_string()]));
        dialogue.add_participant(persona3, MockAgent::new("Agent3", vec!["R3".to_string()]));
        assert_eq!(dialogue.participant_count(), 3);

        // Remove middle participant by name
        dialogue.remove_participant("Agent2").unwrap();
        assert_eq!(dialogue.participant_count(), 2);

        // Verify remaining agents work
        let turns = dialogue.run("Test".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.name(), "Agent1");
        assert_eq!(turns[1].speaker.name(), "Agent3");
    }

    #[tokio::test]
    async fn test_remove_participant_not_found() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        dialogue.add_participant(persona, MockAgent::new("Agent1", vec!["R1".to_string()]));

        // Try to remove non-existent participant
        let result = dialogue.remove_participant("NonExistent");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("participant not found")
        );
    }

    #[tokio::test]
    async fn test_from_blueprint_with_predefined_participants() {
        use crate::agent::persona::{Persona, PersonaTeam};

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Senior engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let blueprint = DialogueBlueprint {
            agenda: "Feature Planning".to_string(),
            context: "Planning new feature".to_string(),
            participants: Some(vec![persona1, persona2]),
            execution_strategy: Some(ExecutionModel::OrderedBroadcast(BroadcastOrder::Completion)),
        };

        // Mock generator agent - won't be used since participants are provided
        #[derive(Clone)]
        struct MockGeneratorAgent;

        #[async_trait]
        impl Agent for MockGeneratorAgent {
            type Output = PersonaTeam;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Generator";
                &EXPERTISE
            }

            fn name(&self) -> String {
                "Generator".to_string()
            }

            async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
                // This should never be called when participants are provided
                panic!("Generator agent should not be called when participants are provided");
            }
        }

        let dialogue_agent = MockAgent::new("DialogueAgent", vec!["Response".to_string()]);

        let mut dialogue = Dialogue::from_blueprint(blueprint, MockGeneratorAgent, dialogue_agent)
            .await
            .unwrap();

        assert_eq!(dialogue.participant_count(), 2);

        let turns = dialogue.run("Test".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.name(), "Alice");
        assert_eq!(turns[1].speaker.name(), "Bob");
    }

    #[tokio::test]
    async fn test_guest_participant_workflow() {
        use crate::agent::persona::Persona;

        // Create core team
        let mut dialogue = Dialogue::broadcast();

        let core1 = Persona {
            name: "CoreMember1".to_string(),
            role: "Core Member".to_string(),
            background: "Core team member".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        let core2 = Persona {
            name: "CoreMember2".to_string(),
            role: "Core Member".to_string(),
            background: "Core team member".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            core1,
            MockAgent::new("CoreMember1", vec!["Core response 1".to_string()]),
        );
        dialogue.add_participant(
            core2,
            MockAgent::new("CoreMember2", vec!["Core response 2".to_string()]),
        );
        assert_eq!(dialogue.participant_count(), 2);

        // Add guest for specific topic
        let guest = Persona {
            name: "Guest Expert".to_string(),
            role: "Domain Specialist".to_string(),
            background: "Guest invited for this session".to_string(),
            communication_style: "Expert".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        let guest_llm = MockAgent::new("Guest", vec!["Guest insight".to_string()]);
        dialogue.add_participant(guest, guest_llm);
        assert_eq!(dialogue.participant_count(), 3);

        // Discussion with guest
        let with_guest = dialogue
            .run("Topic requiring expert input".to_string())
            .await
            .unwrap();
        assert_eq!(with_guest.len(), 3); // All 3 participants

        // Remove guest after their input (by name)
        dialogue.remove_participant("Guest Expert").unwrap();
        assert_eq!(dialogue.participant_count(), 2);

        // Continue with core team only
        let core_only = dialogue
            .run("Continue discussion".to_string())
            .await
            .unwrap();
        assert_eq!(core_only.len(), 2); // Back to 2 core members
    }

    #[tokio::test]
    async fn test_participants_getter() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        // Initially empty
        assert_eq!(dialogue.participants().len(), 0);

        // Add first participant
        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Senior engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        dialogue.add_participant(
            persona1.clone(),
            MockAgent::new("Alice", vec!["Response 1".to_string()]),
        );

        // Add second participant
        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        dialogue.add_participant(
            persona2.clone(),
            MockAgent::new("Bob", vec!["Response 2".to_string()]),
        );

        // Add third participant
        let persona3 = Persona {
            name: "Charlie".to_string(),
            role: "Manager".to_string(),
            background: "Product manager".to_string(),
            communication_style: "Strategic".to_string(),
            visual_identity: None,
            capabilities: None,
        };
        dialogue.add_participant(
            persona3.clone(),
            MockAgent::new("Charlie", vec!["Response 3".to_string()]),
        );

        // Get participants and verify
        let participants = dialogue.participants();
        assert_eq!(participants.len(), 3);

        // Check names
        assert_eq!(participants[0].name, "Alice");
        assert_eq!(participants[1].name, "Bob");
        assert_eq!(participants[2].name, "Charlie");

        // Check roles
        assert_eq!(participants[0].role, "Developer");
        assert_eq!(participants[1].role, "Designer");
        assert_eq!(participants[2].role, "Manager");

        // Check backgrounds
        assert_eq!(participants[0].background, "Senior engineer");
        assert_eq!(participants[1].background, "UX specialist");
        assert_eq!(participants[2].background, "Product manager");

        // Check communication styles
        assert_eq!(participants[0].communication_style, "Technical");
        assert_eq!(participants[1].communication_style, "User-focused");
        assert_eq!(participants[2].communication_style, "Strategic");

        // Remove one and verify list updates
        dialogue.remove_participant("Bob").unwrap();
        let participants_after_removal = dialogue.participants();
        assert_eq!(participants_after_removal.len(), 2);
        assert_eq!(participants_after_removal[0].name, "Alice");
        assert_eq!(participants_after_removal[1].name, "Charlie");
    }

    #[tokio::test]
    async fn test_partial_session_with_multimodal_payload() {
        use crate::agent::persona::Persona;
        use crate::attachment::Attachment;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Analyst".to_string(),
            role: "Image Analyst".to_string(),
            background: "Expert in image analysis".to_string(),
            communication_style: "Technical and precise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            persona,
            MockAgent::new("Analyst", vec!["Image analysis complete".to_string()]),
        );

        // Create multimodal payload
        let payload = Payload::text("Analyze this image")
            .with_attachment(Attachment::local("/test/image.png"));

        let mut session = dialogue.partial_session(payload);

        let turn = session.next_turn().await.unwrap().unwrap();
        assert_eq!(turn.speaker.name(), "Analyst");
        assert_eq!(turn.content, "Image analysis complete");

        // Verify history contains text representation
        assert_eq!(dialogue.history().len(), 2);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "Analyze this image");

        // Verify attachment is stored in MessageStore
        let messages = dialogue.message_store().all_messages();
        assert_eq!(messages.len(), 2);

        let first_message = messages[0];

        assert!(
            first_message.metadata.has_attachments,
            "First message should have attachments flag set"
        );
        assert_eq!(
            first_message.metadata.attachments.len(),
            1,
            "First message should contain 1 attachment"
        );

        // Verify the attachment is a Local variant with the correct path
        match &first_message.metadata.attachments[0] {
            Attachment::Local(path) => {
                assert_eq!(path.to_str().unwrap(), "/test/image.png");
            }
            _ => panic!("Expected Local attachment"),
        }
    }

    #[tokio::test]
    async fn test_run_with_string_literal_backward_compatibility() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            persona,
            MockAgent::new("Agent", vec!["Response".to_string()]),
        );

        // Test backward compatibility: should accept string literal
        let turns = dialogue.run("Hello").await.unwrap();

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content, "Response");
    }

    #[tokio::test]
    async fn test_run_with_multimodal_payload() {
        use crate::agent::persona::Persona;
        use crate::attachment::Attachment;

        let mut dialogue = Dialogue::sequential();

        let persona1 = Persona {
            name: "Analyzer".to_string(),
            role: "Data Analyzer".to_string(),
            background: "Analyzes data".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Summarizer".to_string(),
            role: "Summarizer".to_string(),
            background: "Summarizes results".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Analyzer", vec!["Data analyzed".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Summarizer", vec!["Summary complete".to_string()]),
            );

        // Create multimodal payload
        let payload = Payload::text("Process this data")
            .with_attachment(Attachment::local("/test/data.csv"))
            .with_attachment(Attachment::local("/test/metadata.json"));

        let turns = dialogue.run(payload).await.unwrap();

        // Sequential returns only final turn
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Summarizer");
        assert_eq!(turns[0].content, "Summary complete");

        // Verify history has both turns
        assert_eq!(dialogue.history().len(), 3); // System + 2 participants

        // Verify attachments are stored in MessageStore
        let messages = dialogue.message_store().all_messages();
        assert_eq!(messages.len(), 3);

        let first_message = messages[0];

        // Check that the message has attachments
        assert!(
            first_message.metadata.has_attachments,
            "First message should have attachments flag set"
        );
        assert_eq!(
            first_message.metadata.attachments.len(),
            2,
            "First message should contain 2 attachments"
        );

        // Verify attachment paths by matching on the enum variants
        let attachment_paths: Vec<String> = first_message
            .metadata
            .attachments
            .iter()
            .filter_map(|a| match a {
                Attachment::Local(path) => path.to_str().map(|s| s.to_string()),
                _ => None,
            })
            .collect();

        assert_eq!(attachment_paths.len(), 2);
        assert!(
            attachment_paths.contains(&"/test/data.csv".to_string()),
            "Should contain data.csv attachment"
        );
        assert!(
            attachment_paths.contains(&"/test/metadata.json".to_string()),
            "Should contain metadata.json attachment"
        );
    }

    #[tokio::test]
    async fn test_partial_session_with_ordered_broadcast_and_payload() {
        use crate::agent::persona::Persona;
        use crate::attachment::Attachment;

        let mut dialogue = Dialogue::broadcast();

        let persona1 = Persona {
            name: "First".to_string(),
            role: "First Responder".to_string(),
            background: "Quick analysis".to_string(),
            communication_style: "Fast".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Second".to_string(),
            role: "Second Responder".to_string(),
            background: "Detailed analysis".to_string(),
            communication_style: "Thorough".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(persona1, DelayAgent::new("First", 50))
            .add_participant(persona2, DelayAgent::new("Second", 10));

        // Create payload with attachment
        let payload = Payload::text("Analyze").with_attachment(Attachment::local("/test/file.txt"));

        let mut session =
            dialogue.partial_session_with_order(payload, BroadcastOrder::ParticipantOrder);

        // Should yield in participant order, not completion order
        let first = session.next_turn().await.unwrap().unwrap();
        assert_eq!(first.speaker.name(), "First");

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.speaker.name(), "Second");

        assert!(session.next_turn().await.is_none());
    }

    #[tokio::test]
    async fn test_with_history_injection() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            persona.clone(),
            MockAgent::new("Agent1", vec!["Response 1".to_string()]),
        );

        let turns = dialogue.run("Initial prompt".to_string()).await.unwrap();
        assert_eq!(turns.len(), 1);

        // Get the history
        let history = dialogue.history().to_vec();
        assert_eq!(history.len(), 2); // System + Agent1

        // Create a new dialogue with injected history
        let mut dialogue2 = Dialogue::broadcast().with_history(history);
        dialogue2.add_participant(
            persona,
            MockAgent::new("Agent1", vec!["Response 2".to_string()]),
        );

        // Verify the injected history is present
        assert_eq!(dialogue2.history().len(), 2);
        assert_eq!(dialogue2.history()[0].speaker.name(), "System");
        assert_eq!(dialogue2.history()[1].speaker.name(), "Agent1");
        assert_eq!(dialogue2.history()[1].content, "Response 1");

        // Run new dialogue - should add to existing history
        let new_turns = dialogue2.run("Continue".to_string()).await.unwrap();
        assert_eq!(new_turns.len(), 1);

        // Total history should now be 4: old (System + Agent1) + new (System + Agent1)
        assert_eq!(dialogue2.history().len(), 4);
    }

    #[tokio::test]
    async fn test_save_and_load_history() {
        use crate::agent::persona::Persona;
        use tempfile::NamedTempFile;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            persona,
            MockAgent::new("Agent1", vec!["Test response".to_string()]),
        );

        let turns = dialogue.run("Test prompt".to_string()).await.unwrap();
        assert_eq!(turns.len(), 1);

        // Save history to temp file
        let temp_file = NamedTempFile::new().unwrap();
        dialogue.save_history(temp_file.path()).unwrap();

        // Load history
        let loaded_history = Dialogue::load_history(temp_file.path()).unwrap();

        // Verify loaded history matches original
        assert_eq!(loaded_history.len(), dialogue.history().len());
        assert_eq!(loaded_history[0].speaker.name(), "System");
        assert_eq!(loaded_history[0].content, "Test prompt");
        assert_eq!(loaded_history[1].speaker.name(), "Agent1");
        assert_eq!(loaded_history[1].content, "Test response");
    }

    #[tokio::test]
    async fn test_session_resumption_workflow() {
        use crate::agent::persona::Persona;
        use tempfile::NamedTempFile;

        // Session 1: Initial conversation
        let mut session1 = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        session1.add_participant(
            persona.clone(),
            MockAgent::new("Agent1", vec!["First response".to_string()]),
        );

        let turns1 = session1.run("First prompt".to_string()).await.unwrap();
        assert_eq!(turns1.len(), 1);
        assert_eq!(turns1[0].content, "First response");

        // Save session
        let temp_file = NamedTempFile::new().unwrap();
        session1.save_history(temp_file.path()).unwrap();

        // --- Simulate process restart or session end ---

        // Session 2: Resume conversation
        let loaded_history = Dialogue::load_history(temp_file.path()).unwrap();

        let mut session2 = Dialogue::broadcast().with_history(loaded_history);
        session2.add_participant(
            persona,
            MockAgent::new("Agent1", vec!["Second response".to_string()]),
        );

        // Verify history is restored
        assert_eq!(session2.history().len(), 2); // System + Agent1 from session1

        // Continue conversation
        let turns2 = session2.run("Second prompt".to_string()).await.unwrap();
        assert_eq!(turns2.len(), 1);
        assert_eq!(turns2[0].content, "Second response");

        // Verify complete history
        assert_eq!(session2.history().len(), 4);
        // Session 1 turns
        assert_eq!(session2.history()[0].speaker.name(), "System");
        assert_eq!(session2.history()[0].content, "First prompt");
        assert_eq!(session2.history()[1].speaker.name(), "Agent1");
        assert_eq!(session2.history()[1].content, "First response");
        // Session 2 turns
        assert_eq!(session2.history()[2].speaker.name(), "System");
        assert_eq!(session2.history()[2].content, "Second prompt");
        assert_eq!(session2.history()[3].speaker.name(), "Agent1");
        assert_eq!(session2.history()[3].content, "Second response");
    }

    #[tokio::test]
    async fn test_dialogue_turn_serialization() {
        // Test that DialogueTurn can be serialized and deserialized
        let turn = DialogueTurn {
            speaker: Speaker::agent("TestAgent", "Tester"),
            content: "Test content".to_string(),
        };

        let json = serde_json::to_string(&turn).unwrap();
        let deserialized: DialogueTurn = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.speaker.name(), "TestAgent");
        assert_eq!(deserialized.speaker.role(), Some("Tester"));
        assert_eq!(deserialized.content, "Test content");
    }

    /// Test multi-turn broadcast to verify that each agent sees all messages from previous turns.
    ///
    /// Expected behavior:
    /// - Turn 1: U->A, U->B (both see only User's message)
    /// - Turn 2: U->A, U->B (both see: [System]: U1, [A]: response1, [B]: response1, [System]: U2)
    ///
    /// This ensures proper dialogue history is maintained across multiple broadcast rounds.
    #[tokio::test]
    async fn test_multi_turn_broadcast_history_visibility() {
        use crate::agent::persona::Persona;

        // Create a mock agent that echoes what it receives to verify history visibility
        #[derive(Clone)]
        struct EchoAgent {
            name: String,
        }

        #[async_trait]
        impl Agent for EchoAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Echo agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                // Echo back the input to verify what history was received
                Ok(format!(
                    "[EchoAgent]{} received: {}",
                    self.name,
                    payload.to_text()
                ))
            }
        }

        let mut dialogue = Dialogue::broadcast();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "Tester".to_string(),
            background: "Test agent A".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Tester".to_string(),
            background: "Test agent B".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona_a,
                EchoAgent {
                    name: "AgentA".to_string(),
                },
            )
            .add_participant(
                persona_b,
                EchoAgent {
                    name: "AgentB".to_string(),
                },
            );

        // Turn 1: Initial broadcast
        let turns1 = dialogue.run("First message").await.unwrap();
        assert_eq!(turns1.len(), 2);

        // Verify Turn 1 responses - agents should only see the first message
        assert!(turns1[0].content.contains("First message"));
        assert!(turns1[1].content.contains("First message"));

        // History should have: [System]: First message, [AgentA]: response, [AgentB]: response
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");

        // Turn 2: Second broadcast - agents should see all previous messages
        let turns2 = dialogue.run("Second message").await.unwrap();
        assert_eq!(turns2.len(), 2);

        // Verify Turn 2 responses - agents should see the ENTIRE history:
        // [System]: First message
        // [AgentA]: <their response>
        // [AgentB]: <their response>
        // [System]: Second message
        for turn in &turns2 {
            // Each agent should receive context from OTHER participants (new format)
            // Format: ## AgentName (Role)\n content...
            if turn.speaker.name() == "AgentA" {
                // AgentA should see AgentB's context
                assert!(
                    turn.content.contains("AgentB"),
                    "AgentA should see AgentB's context. Got: {}",
                    turn.content
                );
            } else if turn.speaker.name() == "AgentB" {
                // AgentB should see AgentA's context
                assert!(
                    turn.content.contains("AgentA"),
                    "AgentB should see AgentA's context. Got: {}",
                    turn.content
                );
            }

            // All agents should see the current task
            assert!(
                turn.content.contains("Second message"),
                "Agent {} should see the current task. Got: {}",
                turn.speaker.name(),
                turn.content
            );
        }

        // Total history should be:
        // Turn 1: System + AgentA + AgentB = 3
        // Turn 2: System + AgentA + AgentB = 3
        // Total = 6
        assert_eq!(dialogue.history().len(), 6);

        // Verify the complete history structure
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");
        assert_eq!(dialogue.history()[1].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[2].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[3].speaker.name(), "System");
        assert_eq!(dialogue.history()[3].content, "Second message");
        assert_eq!(dialogue.history()[4].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[5].speaker.name(), "AgentB");
    }

    /// Test to verify HistoryAwareAgent behavior with new context distribution.
    ///
    /// With the new implementation:
    /// - Dialogue distributes context from other participants
    /// - Each HistoryAwareAgent manages its OWN conversation history
    /// - This creates a clear separation: Dialogue context vs. Agent history
    /// - Both histories are visible, which is intentional and correct
    #[tokio::test]
    async fn test_dialogue_history_format_with_chat_agents() {
        use crate::agent::chat::Chat;
        use crate::agent::persona::Persona;

        // Create an echo agent that shows exactly what it receives
        #[derive(Clone)]
        struct VerboseEchoAgent {
            name: String,
        }

        #[async_trait]
        impl Agent for VerboseEchoAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Verbose echo agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                // Return the EXACT input to see what the agent receives
                let input = payload.to_text();

                // For testing: include key sections even if truncated
                let preview = if input.len() > 500 {
                    // Try to include "Recent History" section if it exists
                    if let Some(start) = input.find("# Recent History") {
                        let end = start + 500.min(input.len() - start);
                        format!("...{}...", &input[start..end])
                    } else if let Some(start) = input.find("# Participants") {
                        let end = start + 500.min(input.len() - start);
                        format!("...{}...", &input[start..end])
                    } else {
                        format!("{}...", &input[..500])
                    }
                } else {
                    input.clone()
                };

                Ok(format!(
                    "[{}] Received {} chars: {}",
                    self.name,
                    input.len(),
                    preview
                ))
            }
        }

        let mut dialogue = Dialogue::broadcast();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "Tester".to_string(),
            background: "Test agent A".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Tester".to_string(),
            background: "Test agent B".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        // Create Chat agents WITH history (as Dialogue does)
        let chat_a = Chat::new(VerboseEchoAgent {
            name: "AgentA".to_string(),
        })
        .with_persona(persona_a.clone())
        .with_history(true) // This is what Dialogue does
        .build();

        let chat_b = Chat::new(VerboseEchoAgent {
            name: "AgentB".to_string(),
        })
        .with_persona(persona_b.clone())
        .with_history(true)
        .build();

        dialogue.participants.push(Participant {
            persona: persona_a,
            agent: Arc::new(*chat_a),
            joining_strategy: None,
            has_sent_once: false,
        });

        dialogue.participants.push(Participant {
            persona: persona_b,
            agent: Arc::new(*chat_b),
            joining_strategy: None,
            has_sent_once: false,
        });

        // Turn 1
        let turns1 = dialogue.run("First message").await.unwrap();
        println!("\n=== Turn 1 ===");
        for turn in &turns1 {
            println!("[{}]: {}", turn.speaker.name(), turn.content);
        }

        // Turn 2 - Verify new context distribution works correctly
        let turns2 = dialogue.run("Second message").await.unwrap();
        println!("\n=== Turn 2 ===");
        for turn in &turns2 {
            println!("[{}]: {}", turn.speaker.name(), turn.content);

            // Check if the response contains "Previous Conversation" (from HistoryAwareAgent)
            // This is EXPECTED and CORRECT - each agent manages its own history
            if turn.content.contains("Previous Conversation") {
                println!(
                    "✓ Agent {} maintains its own conversation history (expected)",
                    turn.speaker.name()
                );
            }

            // Verify context from other participants is present
            if turn.speaker.name() == "AgentA" {
                assert!(
                    turn.content.contains("AgentB") || turn.content.contains("# Request"),
                    "AgentA should receive context or request"
                );
            } else if turn.speaker.name() == "AgentB" {
                assert!(
                    turn.content.contains("AgentA") || turn.content.contains("# Request"),
                    "AgentB should receive context or request"
                );
            }
        }

        // New implementation provides clear separation:
        // 1. Dialogue context: Context from other participants (## AgentName...)
        // 2. Agent history: HistoryAwareAgent's own conversation ("# Previous Conversation...")
        // Both are visible and serve different purposes - this is the intended design
    }

    #[tokio::test]
    async fn test_multi_message_payload() {
        use crate::agent::dialogue::message::Speaker;

        // Test that Payload::from_messages() correctly stores multiple messages
        // with proper speaker attribution

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(
            Persona {
                name: "Agent1".to_string(),
                role: "Tester".to_string(),
                background: "Test agent 1".to_string(),
                communication_style: "Direct".to_string(),
                visual_identity: None,
                capabilities: None,
            },
            MockAgent::new("Agent1", vec!["Response from Agent1".to_string()]),
        );
        dialogue.add_participant(
            Persona {
                name: "Agent2".to_string(),
                role: "Tester".to_string(),
                background: "Test agent 2".to_string(),
                communication_style: "Direct".to_string(),
                visual_identity: None,
                capabilities: None,
            },
            MockAgent::new("Agent2", vec!["Response from Agent2".to_string()]),
        );

        // Create multi-message payload with System + User messages
        let payload = Payload::from_messages(vec![
            PayloadMessage::system("System: Initializing conversation"),
            PayloadMessage::user("Alice", "Product Manager", "User: What should we build?"),
        ]);

        let _turns = dialogue.run(payload).await.unwrap();

        // Verify all messages are stored in history
        let history = dialogue.history();

        // Should have: System message, User message, Agent1 response, Agent2 response
        assert!(
            history.len() >= 4,
            "Expected at least 4 messages, got {}",
            history.len()
        );

        // Verify first message is System
        assert_eq!(history[0].speaker, Speaker::System);
        assert_eq!(history[0].content, "System: Initializing conversation");

        // Verify second message is User
        assert_eq!(
            history[1].speaker,
            Speaker::user("Alice", "Product Manager")
        );
        assert_eq!(history[1].content, "User: What should we build?");

        // Verify third and fourth are Agent responses
        assert!(matches!(history[2].speaker, Speaker::Agent { .. }));
        assert!(matches!(history[3].speaker, Speaker::Agent { .. }));
    }

    #[tokio::test]
    async fn test_multi_message_payload_sequential() {
        use crate::agent::dialogue::message::Speaker;

        // Test multi-message payload in sequential mode
        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(
            Persona {
                name: "Agent1".to_string(),
                role: "Analyzer".to_string(),
                background: "First agent".to_string(),
                communication_style: "Analytical".to_string(),
                visual_identity: None,
                capabilities: None,
            },
            MockAgent::new("Agent1", vec!["Analysis result".to_string()]),
        );
        dialogue.add_participant(
            Persona {
                name: "Agent2".to_string(),
                role: "Reviewer".to_string(),
                background: "Second agent".to_string(),
                communication_style: "Critical".to_string(),
                visual_identity: None,
                capabilities: None,
            },
            MockAgent::new("Agent2", vec!["Review complete".to_string()]),
        );

        // Create payload with multiple speakers
        let payload = Payload::from_messages(vec![
            PayloadMessage::system("Context: Project initialization"),
            PayloadMessage::user("Bob", "Engineer", "Request: Analyze architecture"),
        ]);

        let turns = dialogue.run(payload).await.unwrap();

        // Sequential mode returns only final agent's turn
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Agent2");

        // But history should contain all messages
        let history = dialogue.history();
        assert!(
            history.len() >= 4,
            "Expected at least 4 messages in history, got {}",
            history.len()
        );

        // Verify input messages are preserved
        assert_eq!(history[0].speaker, Speaker::System);
        assert_eq!(history[1].speaker, Speaker::user("Bob", "Engineer"));
    }

    /// Test to verify that partial_session correctly maintains history across multiple turns,
    /// just like run() does.
    #[tokio::test]
    async fn test_partial_session_multi_turn_history_continuity() {
        use crate::agent::persona::Persona;

        // Create an echo agent that shows what it receives
        #[derive(Clone)]
        struct HistoryEchoAgent {
            name: String,
        }

        #[async_trait]
        impl Agent for HistoryEchoAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "History echo agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                // Echo back input to verify what history was received
                Ok(format!("{} received: {}", self.name, payload.to_text()))
            }
        }

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            persona,
            HistoryEchoAgent {
                name: "Agent1".to_string(),
            },
        );

        // Turn 1: Execute first turn with partial_session
        let mut session1 = dialogue.partial_session("First message");
        let mut turn1_results = Vec::new();
        while let Some(Ok(turn)) = session1.next_turn().await {
            turn1_results.push(turn);
        }

        assert_eq!(turn1_results.len(), 1);
        assert!(turn1_results[0].content.contains("First message"));
        drop(session1);

        // Verify Turn 1 is stored in history
        assert_eq!(dialogue.history().len(), 2); // System + Agent1
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");
        assert_eq!(dialogue.history()[1].speaker.name(), "Agent1");

        // Turn 2: Execute second turn with partial_session
        let mut session2 = dialogue.partial_session("Second message");
        let mut turn2_results = Vec::new();
        while let Some(Ok(turn)) = session2.next_turn().await {
            turn2_results.push(turn);
        }

        assert_eq!(turn2_results.len(), 1);

        // Verify Turn 2 receives context from Turn 1 (via TurnInput formatting)
        // The agent should see context from other participants (in this case, none because solo)
        // but the current task should be "Second message"
        assert!(turn2_results[0].content.contains("Second message"));
        drop(session2);

        // Verify complete history contains both turns
        assert_eq!(dialogue.history().len(), 4); // System + Agent1 + System + Agent1
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");
        assert_eq!(dialogue.history()[1].speaker.name(), "Agent1");
        assert_eq!(dialogue.history()[2].speaker.name(), "System");
        assert_eq!(dialogue.history()[2].content, "Second message");
        assert_eq!(dialogue.history()[3].speaker.name(), "Agent1");

        // Verify current_turn increments correctly
        assert_eq!(dialogue.message_store.latest_turn(), 2);
    }

    /// Test partial_session with broadcast mode, multiple agents, and both Text and Messages.
    ///
    /// This test verifies:
    /// 1. Multiple agents in broadcast mode with partial_session
    /// 2. Text-only payloads are correctly recorded
    /// 3. Structured Messages (Payload::from_messages) are correctly recorded
    /// 4. All messages (System, User, Agent) are stored in MessageStore
    /// 5. Agents see each other's responses in subsequent turns
    #[tokio::test]
    async fn test_partial_session_broadcast_multi_agent_with_messages() {
        use crate::agent::persona::Persona;

        // Create an echo agent that shows what it receives
        #[derive(Clone)]
        struct EchoAgent {
            name: String,
        }

        #[async_trait]
        impl Agent for EchoAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Echo agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
                Ok(format!("[{}]", self.name))
            }
        }

        let mut dialogue = Dialogue::broadcast();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "TesterA".to_string(),
            background: "Test agent A".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "TesterB".to_string(),
            background: "Test agent B".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona_a,
                EchoAgent {
                    name: "AgentA".to_string(),
                },
            )
            .add_participant(
                persona_b,
                EchoAgent {
                    name: "AgentB".to_string(),
                },
            );

        // Turn 1: Text-only payload
        let mut session1 = dialogue.partial_session("First text message");
        let mut turn1_results = Vec::new();
        while let Some(Ok(turn)) = session1.next_turn().await {
            turn1_results.push(turn);
        }

        assert_eq!(turn1_results.len(), 2); // Two agents
        assert_eq!(turn1_results[0].content, "[AgentA]");
        assert_eq!(turn1_results[1].content, "[AgentB]");
        drop(session1);

        // Verify Turn 1 history: System + AgentA + AgentB
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First text message");
        assert_eq!(dialogue.history()[1].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[1].content, "[AgentA]");
        assert_eq!(dialogue.history()[2].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[2].content, "[AgentB]");

        // Turn 2: Structured Messages (multiple messages in one payload)
        let payload2 = Payload::from_messages(vec![
            PayloadMessage::user("User1", "Human", "User message 1"),
            PayloadMessage::system("System alert"),
            PayloadMessage::user("User2", "Human", "User message 2"),
        ]);

        let mut session2 = dialogue.partial_session(payload2);
        let mut turn2_results = Vec::new();
        while let Some(Ok(turn)) = session2.next_turn().await {
            turn2_results.push(turn);
        }

        assert_eq!(turn2_results.len(), 2); // Two agents
        drop(session2);

        // Verify Turn 2 history: previous 3 + new 3 messages + 2 agent responses
        // Total: 3 (turn1) + 3 (turn2 input) + 2 (turn2 responses) = 8
        assert_eq!(dialogue.history().len(), 8);

        // Turn 1 messages (unchanged)
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First text message");
        assert_eq!(dialogue.history()[1].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[2].speaker.name(), "AgentB");

        // Turn 2 input messages (structured)
        assert_eq!(dialogue.history()[3].speaker.name(), "User1");
        assert_eq!(dialogue.history()[3].content, "User message 1");
        assert_eq!(dialogue.history()[4].speaker.name(), "System");
        assert_eq!(dialogue.history()[4].content, "System alert");
        assert_eq!(dialogue.history()[5].speaker.name(), "User2");
        assert_eq!(dialogue.history()[5].content, "User message 2");

        // Turn 2 agent responses
        assert_eq!(dialogue.history()[6].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[6].content, "[AgentA]");
        assert_eq!(dialogue.history()[7].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[7].content, "[AgentB]");

        // Verify current_turn increments correctly
        assert_eq!(dialogue.message_store.latest_turn(), 2);
    }

    /// Test that partial_session multi-turn behavior matches run() multi-turn behavior
    #[tokio::test]
    async fn test_partial_session_vs_run_multi_turn_equivalence() {
        use crate::agent::persona::Persona;

        #[derive(Clone)]
        struct SimpleAgent {
            response: String,
        }

        #[async_trait]
        impl Agent for SimpleAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Simple agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                "SimpleAgent".to_string()
            }

            async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
                Ok(self.response.clone())
            }
        }

        let persona = Persona {
            name: "Agent".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        // Test with run()
        let mut dialogue_run = Dialogue::broadcast();
        dialogue_run.add_participant(
            persona.clone(),
            SimpleAgent {
                response: "Response".to_string(),
            },
        );

        dialogue_run.run("Turn 1").await.unwrap();
        dialogue_run.run("Turn 2").await.unwrap();

        let history_run = dialogue_run.history();

        // Test with partial_session
        let mut dialogue_partial = Dialogue::broadcast();
        dialogue_partial.add_participant(
            persona,
            SimpleAgent {
                response: "Response".to_string(),
            },
        );

        let mut session1 = dialogue_partial.partial_session("Turn 1");
        while (session1.next_turn().await).is_some() {}
        drop(session1);

        let mut session2 = dialogue_partial.partial_session("Turn 2");
        while (session2.next_turn().await).is_some() {}
        drop(session2);

        let history_partial = dialogue_partial.history();

        // Both should have identical history structure
        assert_eq!(history_run.len(), history_partial.len());
        assert_eq!(
            dialogue_run.message_store.latest_turn(),
            dialogue_partial.message_store.latest_turn()
        );

        // Verify both have the same message structure
        for (i, (run_msg, partial_msg)) in
            history_run.iter().zip(history_partial.iter()).enumerate()
        {
            assert_eq!(
                run_msg.speaker.name(),
                partial_msg.speaker.name(),
                "Speaker mismatch at index {}",
                i
            );
            assert_eq!(
                run_msg.content, partial_msg.content,
                "Content mismatch at index {}",
                i
            );
        }
    }

    /// Test multi-turn sequential (2 members) to verify message flow across turns.
    ///
    /// Expected behavior for 2 members (AgentA, AgentB) in Sequential mode:
    /// - Turn 1: System -> AgentA(sees System) -> AgentB(sees AgentA's output) -> Done (returns B1)
    /// - Turn 2: System -> AgentA(sees B1 + new System) -> AgentB(sees AgentA's new output) -> Done (returns B2)
    ///
    /// Key: First agent (AgentA) should receive previous turn's final output (B1).
    #[tokio::test]
    async fn test_multi_turn_sequential_2_members() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        // Create mock agents that echo what they receive to verify message flow
        #[derive(Clone)]
        struct TrackingAgent {
            name: String,
            responses: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for TrackingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Tracking agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                let input = payload.to_text();
                let response = format!("[{}] received input", self.name);

                // Record what was received for verification
                self.responses.lock().await.push(input.clone());

                Ok(response)
            }
        }

        let mut dialogue = Dialogue::sequential();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "First".to_string(),
            background: "First agent in chain".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Second".to_string(),
            background: "Second agent in chain".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_a_responses = Arc::new(Mutex::new(Vec::new()));
        let agent_b_responses = Arc::new(Mutex::new(Vec::new()));

        dialogue
            .add_participant(
                persona_a,
                TrackingAgent {
                    name: "AgentA".to_string(),
                    responses: Arc::clone(&agent_a_responses),
                },
            )
            .add_participant(
                persona_b,
                TrackingAgent {
                    name: "AgentB".to_string(),
                    responses: Arc::clone(&agent_b_responses),
                },
            );

        // Turn 1: Initial sequential execution
        println!("\n=== Turn 1 ===");
        let turns1 = dialogue.run("First message").await.unwrap();

        // Sequential mode returns only the final turn (AgentB's response)
        assert_eq!(turns1.len(), 1);
        assert_eq!(turns1[0].speaker.name(), "AgentB");
        assert_eq!(turns1[0].content, "[AgentB] received input");

        // Verify Turn 1 message flow
        let a_inputs_t1 = agent_a_responses.lock().await;
        let b_inputs_t1 = agent_b_responses.lock().await;

        println!("Turn 1 - AgentA received: {}", a_inputs_t1[0]);
        println!("Turn 1 - AgentB received: {}", b_inputs_t1[0]);

        // AgentA should see original message
        assert!(
            a_inputs_t1[0].contains("First message"),
            "AgentA should see 'First message' in Turn 1. Got: {}",
            a_inputs_t1[0]
        );

        // AgentB should see AgentA's response
        assert!(
            b_inputs_t1[0].contains("[AgentA] received input"),
            "AgentB should see AgentA's output in Turn 1. Got: {}",
            b_inputs_t1[0]
        );

        drop(a_inputs_t1);
        drop(b_inputs_t1);

        // History: System(Turn1) + AgentA(Turn1) + AgentB(Turn1) = 3
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");
        assert_eq!(dialogue.history()[1].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[1].content, "[AgentA] received input");
        assert_eq!(dialogue.history()[2].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[2].content, "[AgentB] received input");

        // Turn 2: Second sequential execution
        println!("\n=== Turn 2 ===");
        let turns2 = dialogue.run("Second message").await.unwrap();

        assert_eq!(turns2.len(), 1);
        assert_eq!(turns2[0].speaker.name(), "AgentB");
        assert_eq!(turns2[0].content, "[AgentB] received input");

        // Verify Turn 2 message flow
        let a_inputs_t2 = agent_a_responses.lock().await;
        let b_inputs_t2 = agent_b_responses.lock().await;

        println!("Turn 2 - AgentA received: {}", a_inputs_t2[1]);
        println!("Turn 2 - AgentB received: {}", b_inputs_t2[1]);

        // KEY TEST: AgentA should see ALL Turn 1 outputs:
        // 1. Its own Turn 1 output (AgentA)
        // 2. AgentB's Turn 1 output
        // 3. New system message (Second message)
        assert!(
            a_inputs_t2[1].contains("[AgentA] received input"),
            "AgentA should see its own Turn 1 output as context in Turn 2. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("[AgentB] received input"),
            "AgentA should see AgentB's Turn 1 output as context in Turn 2. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("Second message"),
            "AgentA should see new message in Turn 2. Got: {}",
            a_inputs_t2[1]
        );

        // AgentB should see AgentA's Turn 2 output + new message
        assert!(
            b_inputs_t2[1].contains("[AgentA] received input"),
            "AgentB should see AgentA's Turn 2 output. Got: {}",
            b_inputs_t2[1]
        );
        assert!(
            b_inputs_t2[1].contains("Second message"),
            "AgentB should see new message. Got: {}",
            b_inputs_t2[1]
        );

        // History: Turn1(3) + Turn2(3) = 6 messages
        assert_eq!(dialogue.history().len(), 6);
        assert_eq!(dialogue.history()[3].speaker.name(), "System"); // Turn 2 input
        assert_eq!(dialogue.history()[3].content, "Second message");
        assert_eq!(dialogue.history()[4].speaker.name(), "AgentA"); // Turn 2 AgentA
        assert_eq!(dialogue.history()[5].speaker.name(), "AgentB"); // Turn 2 AgentB (final)
    }

    /// Sequential session variant of the two-member multi-turn test using partial_session.
    ///
    /// Verifies that the streaming API exposes each intermediate turn while preserving the
    /// same message flow guarantees checked in `test_multi_turn_sequential_2_members`.
    #[tokio::test]
    async fn test_multi_turn_sequential_session_2_members() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        #[derive(Clone)]
        struct TrackingAgent {
            name: String,
            responses: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for TrackingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Tracking agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                let input = payload.to_text();
                let response = format!("[{}] received input", self.name);
                self.responses.lock().await.push(input);
                Ok(response)
            }
        }

        let mut dialogue = Dialogue::sequential();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "First".to_string(),
            background: "First agent in chain".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Second".to_string(),
            background: "Second agent in chain".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_a_responses = Arc::new(Mutex::new(Vec::new()));
        let agent_b_responses = Arc::new(Mutex::new(Vec::new()));

        dialogue
            .add_participant(
                persona_a,
                TrackingAgent {
                    name: "AgentA".to_string(),
                    responses: Arc::clone(&agent_a_responses),
                },
            )
            .add_participant(
                persona_b,
                TrackingAgent {
                    name: "AgentB".to_string(),
                    responses: Arc::clone(&agent_b_responses),
                },
            );

        // Turn 1: Stream sequential execution
        let mut session1 = dialogue.partial_session("First message");
        let mut turns1 = Vec::new();
        while let Some(result) = session1.next_turn().await {
            turns1.push(result.unwrap());
        }
        drop(session1);

        assert_eq!(turns1.len(), 2);
        assert_eq!(turns1[0].speaker.name(), "AgentA");
        assert_eq!(turns1[0].content, "[AgentA] received input");
        assert_eq!(turns1[1].speaker.name(), "AgentB");
        assert_eq!(turns1[1].content, "[AgentB] received input");

        let a_inputs_t1 = agent_a_responses.lock().await;
        let b_inputs_t1 = agent_b_responses.lock().await;

        assert!(
            a_inputs_t1[0].contains("First message"),
            "AgentA should see 'First message' in Turn 1. Got: {}",
            a_inputs_t1[0]
        );
        assert!(
            b_inputs_t1[0].contains("[AgentA] received input"),
            "AgentB should see AgentA's output in Turn 1. Got: {}",
            b_inputs_t1[0]
        );

        drop(a_inputs_t1);
        drop(b_inputs_t1);

        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");
        assert_eq!(dialogue.history()[1].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[1].content, "[AgentA] received input");
        assert_eq!(dialogue.history()[2].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[2].content, "[AgentB] received input");

        // Turn 2: Stream sequential execution with prior context
        let mut session2 = dialogue.partial_session("Second message");
        let mut turns2 = Vec::new();
        while let Some(result) = session2.next_turn().await {
            turns2.push(result.unwrap());
        }
        drop(session2);

        assert_eq!(turns2.len(), 2);
        assert_eq!(turns2[0].speaker.name(), "AgentA");
        assert_eq!(turns2[0].content, "[AgentA] received input");
        assert_eq!(turns2[1].speaker.name(), "AgentB");
        assert_eq!(turns2[1].content, "[AgentB] received input");

        let a_inputs_t2 = agent_a_responses.lock().await;
        let b_inputs_t2 = agent_b_responses.lock().await;

        assert!(
            a_inputs_t2[1].contains("[AgentA] received input"),
            "AgentA should see its own Turn 1 output as context in Turn 2. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("[AgentB] received input"),
            "AgentA should see AgentB's Turn 1 output as context in Turn 2. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("Second message"),
            "AgentA should see new message in Turn 2. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            b_inputs_t2[1].contains("[AgentA] received input"),
            "AgentB should see AgentA's Turn 2 output. Got: {}",
            b_inputs_t2[1]
        );
        assert!(
            b_inputs_t2[1].contains("Second message"),
            "AgentB should see new message. Got: {}",
            b_inputs_t2[1]
        );

        drop(a_inputs_t2);
        drop(b_inputs_t2);

        assert_eq!(dialogue.history().len(), 6);
        assert_eq!(dialogue.history()[3].speaker.name(), "System");
        assert_eq!(dialogue.history()[3].content, "Second message");
        assert_eq!(dialogue.history()[4].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[5].speaker.name(), "AgentB");
    }

    /// Test multi-turn sequential (3 members) to verify message flow across turns.
    ///
    /// Expected behavior for 3 members (A, B, C) in Sequential mode:
    /// - Turn 1: System -> A(sees System) -> B(sees A1) -> C(sees B1) -> Done (returns C1)
    /// - Turn 2: System -> A(sees C1 + new System) -> B(sees A2) -> C(sees B2) -> Done (returns C2)
    ///
    /// Key: Only the first agent (A) receives previous turn's final output (C1).
    #[tokio::test]
    async fn test_multi_turn_sequential_3_members() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        // Create mock agents that track what they receive
        #[derive(Clone)]
        struct TrackingAgent {
            name: String,
            responses: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for TrackingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Tracking agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                let input = payload.to_text();
                let response = format!("[{}] processed", self.name);

                // Record what was received for verification
                self.responses.lock().await.push(input.clone());

                Ok(response)
            }
        }

        let mut dialogue = Dialogue::sequential();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "First".to_string(),
            background: "First agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Second".to_string(),
            background: "Second agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_c = Persona {
            name: "AgentC".to_string(),
            role: "Third".to_string(),
            background: "Third agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_a_responses = Arc::new(Mutex::new(Vec::new()));
        let agent_b_responses = Arc::new(Mutex::new(Vec::new()));
        let agent_c_responses = Arc::new(Mutex::new(Vec::new()));

        dialogue
            .add_participant(
                persona_a,
                TrackingAgent {
                    name: "AgentA".to_string(),
                    responses: Arc::clone(&agent_a_responses),
                },
            )
            .add_participant(
                persona_b,
                TrackingAgent {
                    name: "AgentB".to_string(),
                    responses: Arc::clone(&agent_b_responses),
                },
            )
            .add_participant(
                persona_c,
                TrackingAgent {
                    name: "AgentC".to_string(),
                    responses: Arc::clone(&agent_c_responses),
                },
            );

        // Turn 1
        println!("\n=== Turn 1 (3 members) ===");
        let turns1 = dialogue.run("First message").await.unwrap();

        // Sequential mode returns only the final turn (AgentC's response)
        assert_eq!(turns1.len(), 1);
        assert_eq!(turns1[0].speaker.name(), "AgentC");

        // Verify Turn 1 message flow
        let a_inputs_t1 = agent_a_responses.lock().await;
        let b_inputs_t1 = agent_b_responses.lock().await;
        let c_inputs_t1 = agent_c_responses.lock().await;

        // A sees original message
        assert!(a_inputs_t1[0].contains("First message"));

        // B sees A's output
        assert!(b_inputs_t1[0].contains("[AgentA] processed"));

        // C sees B's output
        assert!(c_inputs_t1[0].contains("[AgentB] processed"));

        drop(a_inputs_t1);
        drop(b_inputs_t1);
        drop(c_inputs_t1);

        // History: System + A + B + C = 4
        assert_eq!(dialogue.history().len(), 4);

        // Turn 2
        println!("\n=== Turn 2 (3 members) ===");
        let turns2 = dialogue.run("Second message").await.unwrap();

        assert_eq!(turns2.len(), 1);
        assert_eq!(turns2[0].speaker.name(), "AgentC");

        // Verify Turn 2 message flow
        let a_inputs_t2 = agent_a_responses.lock().await;
        let b_inputs_t2 = agent_b_responses.lock().await;
        let c_inputs_t2 = agent_c_responses.lock().await;

        println!("Turn 2 - AgentA received: {}", a_inputs_t2[1]);
        println!("Turn 2 - AgentB received: {}", b_inputs_t2[1]);
        println!("Turn 2 - AgentC received: {}", c_inputs_t2[1]);

        // KEY TEST: A should see ALL Turn 1 outputs (A1, B1, C1) + new message
        assert!(
            a_inputs_t2[1].contains("[AgentA] processed"),
            "AgentA should see its own Turn 1 output. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("[AgentB] processed"),
            "AgentA should see AgentB's Turn 1 output. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("[AgentC] processed"),
            "AgentA should see AgentC's Turn 1 output. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("Second message"),
            "AgentA should see new message. Got: {}",
            a_inputs_t2[1]
        );

        // B should see A2's output + new message
        assert!(
            b_inputs_t2[1].contains("[AgentA] processed"),
            "AgentB should see AgentA's Turn 2 output. Got: {}",
            b_inputs_t2[1]
        );
        assert!(
            b_inputs_t2[1].contains("Second message"),
            "AgentB should see new message. Got: {}",
            b_inputs_t2[1]
        );

        // C should see BOTH A2 and B2 outputs + new message
        assert!(
            c_inputs_t2[1].contains("[AgentA] processed"),
            "AgentC should see AgentA's Turn 2 output. Got: {}",
            c_inputs_t2[1]
        );
        assert!(
            c_inputs_t2[1].contains("[AgentB] processed"),
            "AgentC should see AgentB's Turn 2 output. Got: {}",
            c_inputs_t2[1]
        );
        assert!(
            c_inputs_t2[1].contains("Second message"),
            "AgentC should see new message. Got: {}",
            c_inputs_t2[1]
        );

        // History: Turn1(4) + Turn2(4) = 8 messages
        assert_eq!(dialogue.history().len(), 8);
        assert_eq!(dialogue.history()[4].speaker.name(), "System"); // Turn 2 input
        assert_eq!(dialogue.history()[5].speaker.name(), "AgentA"); // Turn 2 A
        assert_eq!(dialogue.history()[6].speaker.name(), "AgentB"); // Turn 2 B
        assert_eq!(dialogue.history()[7].speaker.name(), "AgentC"); // Turn 2 C (final)
    }

    /// Sequential session variant for three participants to ensure partial_session mirrors run().
    #[tokio::test]
    async fn test_multi_turn_sequential_session_3_members() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        #[derive(Clone)]
        struct TrackingAgent {
            name: String,
            responses: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for TrackingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Tracking agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                let input = payload.to_text();
                let response = format!("[{}] processed", self.name);
                self.responses.lock().await.push(input);
                Ok(response)
            }
        }

        let mut dialogue = Dialogue::sequential();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "First".to_string(),
            background: "First agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Second".to_string(),
            background: "Second agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_c = Persona {
            name: "AgentC".to_string(),
            role: "Third".to_string(),
            background: "Third agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_a_responses = Arc::new(Mutex::new(Vec::new()));
        let agent_b_responses = Arc::new(Mutex::new(Vec::new()));
        let agent_c_responses = Arc::new(Mutex::new(Vec::new()));

        dialogue
            .add_participant(
                persona_a,
                TrackingAgent {
                    name: "AgentA".to_string(),
                    responses: Arc::clone(&agent_a_responses),
                },
            )
            .add_participant(
                persona_b,
                TrackingAgent {
                    name: "AgentB".to_string(),
                    responses: Arc::clone(&agent_b_responses),
                },
            )
            .add_participant(
                persona_c,
                TrackingAgent {
                    name: "AgentC".to_string(),
                    responses: Arc::clone(&agent_c_responses),
                },
            );

        // Turn 1: Streamed sequential execution
        let mut session1 = dialogue.partial_session("First message");
        let mut turns1 = Vec::new();
        while let Some(result) = session1.next_turn().await {
            turns1.push(result.unwrap());
        }
        drop(session1);

        assert_eq!(turns1.len(), 3);
        assert_eq!(turns1[0].speaker.name(), "AgentA");
        assert_eq!(turns1[0].content, "[AgentA] processed");
        assert_eq!(turns1[1].speaker.name(), "AgentB");
        assert_eq!(turns1[1].content, "[AgentB] processed");
        assert_eq!(turns1[2].speaker.name(), "AgentC");
        assert_eq!(turns1[2].content, "[AgentC] processed");

        let a_inputs_t1 = agent_a_responses.lock().await;
        let b_inputs_t1 = agent_b_responses.lock().await;
        let c_inputs_t1 = agent_c_responses.lock().await;

        assert!(
            a_inputs_t1[0].contains("First message"),
            "AgentA should see 'First message' in Turn 1. Got: {}",
            a_inputs_t1[0]
        );
        assert!(
            b_inputs_t1[0].contains("[AgentA] processed"),
            "AgentB should see AgentA's Turn 1 output. Got: {}",
            b_inputs_t1[0]
        );
        assert!(
            c_inputs_t1[0].contains("[AgentB] processed"),
            "AgentC should see AgentB's Turn 1 output. Got: {}",
            c_inputs_t1[0]
        );

        drop(a_inputs_t1);
        drop(b_inputs_t1);
        drop(c_inputs_t1);

        assert_eq!(dialogue.history().len(), 4);
        assert_eq!(dialogue.history()[0].speaker.name(), "System");
        assert_eq!(dialogue.history()[0].content, "First message");
        assert_eq!(dialogue.history()[1].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[2].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[3].speaker.name(), "AgentC");

        // Turn 2: Streamed sequential execution with prior context
        let mut session2 = dialogue.partial_session("Second message");
        let mut turns2 = Vec::new();
        while let Some(result) = session2.next_turn().await {
            turns2.push(result.unwrap());
        }
        drop(session2);

        assert_eq!(turns2.len(), 3);
        assert_eq!(turns2[0].speaker.name(), "AgentA");
        assert_eq!(turns2[0].content, "[AgentA] processed");
        assert_eq!(turns2[1].speaker.name(), "AgentB");
        assert_eq!(turns2[1].content, "[AgentB] processed");
        assert_eq!(turns2[2].speaker.name(), "AgentC");
        assert_eq!(turns2[2].content, "[AgentC] processed");

        let a_inputs_t2 = agent_a_responses.lock().await;
        let b_inputs_t2 = agent_b_responses.lock().await;
        let c_inputs_t2 = agent_c_responses.lock().await;

        assert!(
            a_inputs_t2[1].contains("[AgentA] processed"),
            "AgentA should see its own Turn 1 output. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("[AgentB] processed"),
            "AgentA should see AgentB's Turn 1 output. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("[AgentC] processed"),
            "AgentA should see AgentC's Turn 1 output. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            a_inputs_t2[1].contains("Second message"),
            "AgentA should see new message. Got: {}",
            a_inputs_t2[1]
        );
        assert!(
            b_inputs_t2[1].contains("[AgentA] processed"),
            "AgentB should see AgentA's Turn 2 output. Got: {}",
            b_inputs_t2[1]
        );
        assert!(
            b_inputs_t2[1].contains("Second message"),
            "AgentB should see new message. Got: {}",
            b_inputs_t2[1]
        );
        assert!(
            c_inputs_t2[1].contains("[AgentA] processed"),
            "AgentC should see AgentA's Turn 2 output. Got: {}",
            c_inputs_t2[1]
        );
        assert!(
            c_inputs_t2[1].contains("[AgentB] processed"),
            "AgentC should see AgentB's Turn 2 output. Got: {}",
            c_inputs_t2[1]
        );
        assert!(
            c_inputs_t2[1].contains("Second message"),
            "AgentC should see new message. Got: {}",
            c_inputs_t2[1]
        );

        drop(a_inputs_t2);
        drop(b_inputs_t2);
        drop(c_inputs_t2);

        assert_eq!(dialogue.history().len(), 8);
        assert_eq!(dialogue.history()[4].speaker.name(), "System");
        assert_eq!(dialogue.history()[4].content, "Second message");
        assert_eq!(dialogue.history()[5].speaker.name(), "AgentA");
        assert_eq!(dialogue.history()[6].speaker.name(), "AgentB");
        assert_eq!(dialogue.history()[7].speaker.name(), "AgentC");
    }

    /// Ensure partial sequential sessions prepend dialogue context for each participant.
    #[tokio::test]
    async fn test_partial_session_sequential_applies_context() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        #[derive(Clone)]
        struct TrackingAgent {
            name: String,
            payloads: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for TrackingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Context tracking agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                // Get full text representation including both Text and Message contents
                let mut full_text = String::new();

                // Add all messages (converted to text)
                for msg in payload.to_messages() {
                    full_text.push_str(&msg.content);
                    full_text.push('\n');
                }

                // Also add pure Text contents
                let text = payload.to_text();
                if !text.is_empty() {
                    full_text.push_str(&text);
                }

                self.payloads.lock().await.push(full_text);
                Ok(format!("[{}] ok", self.name))
            }
        }

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "First".to_string(),
            background: "First".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Second".to_string(),
            background: "Second".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let payloads = Arc::new(Mutex::new(Vec::new()));

        let mut dialogue = Dialogue::sequential();
        dialogue
            .with_talk_style(TalkStyle::Brainstorm)
            .add_participant(
                persona_a,
                TrackingAgent {
                    name: "AgentA".to_string(),
                    payloads: Arc::clone(&payloads),
                },
            )
            .add_participant(
                persona_b,
                TrackingAgent {
                    name: "AgentB".to_string(),
                    payloads: Arc::clone(&payloads),
                },
            );

        let mut session = dialogue.partial_session("Kickoff");
        while let Some(result) = session.next_turn().await {
            result.unwrap();
        }
        drop(session);

        let payloads = payloads.lock().await;
        assert!(
            payloads
                .iter()
                .all(|text| text.contains("Brainstorming Session")),
            "Each participant should receive brainstorming context. Payloads: {:?}",
            *payloads
        );
    }

    /// Ensure explicit sequential ordering is honored.
    #[tokio::test]
    async fn test_sequential_explicit_order_respected() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        #[derive(Clone)]
        struct RecordingAgent {
            name: String,
            log: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for RecordingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Order recording agent";
                &EXPERTISE
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                let _ = payload;
                self.log.lock().await.push(self.name.clone());
                Ok(format!("[{}] done", self.name))
            }
        }

        let personas = [
            Persona {
                name: "AgentA".to_string(),
                role: "First".to_string(),
                background: "First".to_string(),
                communication_style: "Direct".to_string(),
                visual_identity: None,
                capabilities: None,
            },
            Persona {
                name: "AgentB".to_string(),
                role: "Second".to_string(),
                background: "Second".to_string(),
                communication_style: "Direct".to_string(),
                visual_identity: None,
                capabilities: None,
            },
            Persona {
                name: "AgentC".to_string(),
                role: "Third".to_string(),
                background: "Third".to_string(),
                communication_style: "Direct".to_string(),
                visual_identity: None,
                capabilities: None,
            },
        ];

        let log = Arc::new(Mutex::new(Vec::new()));
        let mut dialogue = Dialogue::sequential_with_order(SequentialOrder::Explicit(vec![
            "AgentB".to_string(),
            "AgentA".to_string(),
        ]));

        for persona in personas {
            let persona_name = persona.name.clone();
            dialogue.add_participant(
                persona,
                RecordingAgent {
                    name: persona_name,
                    log: Arc::clone(&log),
                },
            );
        }

        dialogue.run("Start".to_string()).await.unwrap();

        let order = log.lock().await.clone();
        assert_eq!(
            order,
            vec![
                "AgentB".to_string(),
                "AgentA".to_string(),
                "AgentC".to_string()
            ],
            "Explicit order should run AgentB, then AgentA, then remaining AgentC"
        );
    }

    /// Ensure invalid sequential order names propagate as errors in streaming sessions.
    #[tokio::test]
    async fn test_partial_session_sequential_order_error_propagates() {
        use crate::agent::persona::Persona;

        #[derive(Clone)]
        struct PassthroughAgent;

        #[async_trait]
        impl Agent for PassthroughAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Passthrough";
                &EXPERTISE
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                Ok(payload.to_text())
            }
        }

        let persona = Persona {
            name: "AgentA".to_string(),
            role: "Only".to_string(),
            background: "Only".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let mut dialogue =
            Dialogue::sequential_with_order(SequentialOrder::Explicit(vec!["Ghost".to_string()]));
        dialogue.add_participant(persona, PassthroughAgent);

        let mut session = dialogue.partial_session("Hello world");
        match session.next_turn().await {
            Some(Err(AgentError::ExecutionFailed(message))) => {
                assert!(
                    message.contains("Ghost"),
                    "Error message should mention missing participant. Got {message}"
                );
            }
            other => panic!(
                "Expected ExecutionFailed error from invalid sequential order, got {:?}",
                other
            ),
        }

        assert!(
            session.next_turn().await.is_none(),
            "Session should complete after propagating the error"
        );
    }

    /// Test that DialogueContext is properly applied to the payload
    #[tokio::test]
    async fn test_dialogue_context_brainstorm() {
        use crate::agent::persona::Persona;
        use tokio::sync::Mutex;

        // Create a tracking agent to verify context is included
        #[derive(Clone)]
        struct TrackingAgent {
            name: String,
            received_payloads: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Agent for TrackingAgent {
            type Output = String;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Tracking agent";
                &EXPERTISE
            }

            fn name(&self) -> String {
                self.name.clone()
            }

            async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
                let payload_text = payload.to_text();
                self.received_payloads
                    .lock()
                    .await
                    .push(payload_text.clone());
                Ok(format!("[{}] responded", self.name))
            }
        }

        let mut dialogue = Dialogue::broadcast();
        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Participant".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let received_payloads = Arc::new(Mutex::new(Vec::new()));
        dialogue
            .with_talk_style(TalkStyle::Brainstorm)
            .add_participant(
                persona,
                TrackingAgent {
                    name: "Agent1".to_string(),
                    received_payloads: Arc::clone(&received_payloads),
                },
            );

        // Run dialogue
        dialogue.run("Let's generate some ideas").await.unwrap();

        // Verify context was included in payload
        let payloads = received_payloads.lock().await;
        assert_eq!(payloads.len(), 1, "Should have received one payload");

        let payload_text = &payloads[0];

        // Check for key context markers - they appear in the System message within PersonaAgent's output
        assert!(
            payload_text.contains("Brainstorming Session"),
            "Payload should contain context title. Got: {}",
            payload_text
        );
        assert!(
            payload_text.contains("Encourage wild ideas"),
            "Payload should contain context guidelines. Got: {}",
            payload_text
        );
        // The original prompt "Let's generate some ideas" is not visible in PersonaAgent's output
        // because PersonaAgent wraps everything in its own format. We verify the context was applied
        // by checking that the Brainstorm context instructions are present.
    }

    #[test]
    #[allow(deprecated)]
    fn test_extract_mentions() {
        // Test basic mention extraction
        let text = "@Alice what do you think?";
        let participants = vec!["Alice", "Bob", "Charlie"];
        let mentions = extract_mentions(text, &participants);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Alice"));

        // Test multiple mentions
        let text = "@Alice @Bob please discuss this";
        let mentions = extract_mentions(text, &participants);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.contains(&"Alice"));
        assert!(mentions.contains(&"Bob"));

        // Test duplicate mentions (should be deduplicated)
        let text = "@Alice what do you think? @Alice?";
        let mentions = extract_mentions(text, &participants);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Alice"));

        // Test no mentions
        let text = "What does everyone think?";
        let mentions = extract_mentions(text, &participants);
        assert_eq!(mentions.len(), 0);

        // Test mention that doesn't match any participant (should be ignored)
        let text = "@David @Alice what do you think?";
        let mentions = extract_mentions(text, &participants);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Alice"));
        assert!(!mentions.contains(&"David"));

        // Test exact matching (not partial)
        let participants = vec!["Alice", "Ali"];
        let text = "@Ali what do you think?";
        let mentions = extract_mentions(text, &participants);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Ali"));
        assert!(!mentions.contains(&"Alice"));
    }

    #[test]
    fn test_extract_mentions_with_strategy_exact_word() {
        // ExactWord strategy - matches non-whitespace after @
        let participants = vec!["Alice", "Bob", "Ayaka Nakamura"];
        let text = "@Alice @Bob what do you think?";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::ExactWord);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.contains(&"Alice"));
        assert!(mentions.contains(&"Bob"));

        // Should NOT match space-containing names (space breaks the match)
        let text = "@Ayaka @Nakamura please review";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::ExactWord);
        assert_eq!(
            mentions.len(),
            0,
            "ExactWord should not match partial words of space-containing names"
        );

        // Multibyte support - single word names
        let participants_jp = vec!["太郎", "花子", "Alice"];
        let text = "@太郎 @花子 please discuss";
        let mentions =
            extract_mentions_with_strategy(text, &participants_jp, MentionMatchStrategy::ExactWord);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.contains(&"太郎"));
        assert!(mentions.contains(&"花子"));
    }

    #[test]
    fn test_extract_mentions_with_strategy_name() {
        // Name strategy - matches full names including spaces
        let participants = vec!["Alice", "Bob", "Ayaka Nakamura", "John Smith"];

        // Full name mention with spaces
        let text = "@Ayaka Nakamura please review this";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Name);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Ayaka Nakamura"));

        // Multiple full name mentions
        let text = "@Ayaka Nakamura and @John Smith, please discuss";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Name);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.contains(&"Ayaka Nakamura"));
        assert!(mentions.contains(&"John Smith"));

        // Single-word names still work
        let text = "@Alice @Bob what do you think?";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Name);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.contains(&"Alice"));
        assert!(mentions.contains(&"Bob"));

        // Partial match should NOT work with Name strategy
        let text = "@Ayaka please review";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Name);
        assert_eq!(
            mentions.len(),
            0,
            "Name strategy requires exact full name match"
        );

        // Boundary check: "Ayaka" should not match when "@Ayaka Nakamura" is mentioned
        let participants_with_overlap = vec!["Ayaka", "Ayaka Nakamura", "Bob"];
        let text = "@Ayaka Nakamura please review";
        let mentions = extract_mentions_with_strategy(
            text,
            &participants_with_overlap,
            MentionMatchStrategy::Name,
        );
        assert_eq!(mentions.len(), 1, "Should only match full name");
        assert!(
            mentions.contains(&"Ayaka Nakamura"),
            "Should match 'Ayaka Nakamura', not 'Ayaka'"
        );
        assert!(
            !mentions.contains(&"Ayaka"),
            "'Ayaka' should not match in '@Ayaka Nakamura'"
        );

        // Exact single name should still match
        let text = "@Ayaka what do you think?";
        let mentions = extract_mentions_with_strategy(
            text,
            &participants_with_overlap,
            MentionMatchStrategy::Name,
        );
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Ayaka"));
    }

    #[test]
    fn test_extract_mentions_with_strategy_partial() {
        // Partial strategy - matches by prefix, selecting longest candidate
        let participants = vec!["Alice", "Ayaka Nakamura", "Ayaka Tanaka", "Bob"];

        // Prefix match - should match "Ayaka Nakamura" (longest match)
        let text = "@Ayaka please review";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Partial);
        assert_eq!(mentions.len(), 1);
        // Should match one of the "Ayaka" names - length determines which
        assert!(mentions.iter().any(|&name| name.starts_with("Ayaka")));

        // Exact single-word match
        let text = "@Alice what do you think?";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Partial);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"Alice"));

        // Multiple partial mentions
        let text = "@Ayaka and @Bob, please discuss";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Partial);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.iter().any(|&name| name.starts_with("Ayaka")));
        assert!(mentions.contains(&"Bob"));

        // No match
        let text = "@Charlie what do you think?";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Partial);
        assert_eq!(mentions.len(), 0);
    }

    #[test]
    fn test_extract_mentions_japanese_names() {
        // Test with Japanese names (hiragana/katakana)
        let participants = vec!["あやか なかむら", "太郎 山田", "Alice"];

        // Name strategy with Japanese names
        let text = "@あやか なかむら さん、お願いします";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Name);
        assert_eq!(mentions.len(), 1);
        assert!(mentions.contains(&"あやか なかむら"));

        // Mix of Japanese and English
        let text = "@Alice and @太郎 山田, please review";
        let mentions =
            extract_mentions_with_strategy(text, &participants, MentionMatchStrategy::Name);
        assert_eq!(mentions.len(), 2);
        assert!(mentions.contains(&"Alice"));
        assert!(mentions.contains(&"太郎 山田"));
    }

    #[tokio::test]
    async fn test_participants_method() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(persona1, MockAgent::new("Alice", vec!["Hi".to_string()]))
            .add_participant(persona2, MockAgent::new("Bob", vec!["Hello".to_string()]));

        let participants = dialogue.participant_names();
        assert_eq!(participants.len(), 2);
        assert!(participants.contains(&"Alice"));
        assert!(participants.contains(&"Bob"));
    }

    #[tokio::test]
    async fn test_mentioned_mode_with_mentions() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona3 = Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA engineer".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Alice", vec!["Alice's response".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Bob", vec!["Bob's response".to_string()]),
            )
            .add_participant(
                persona3,
                MockAgent::new("Charlie", vec!["Charlie's response".to_string()]),
            );

        // Only mention Alice and Bob
        let turns = dialogue
            .run("@Alice @Bob what do you think about this feature?")
            .await
            .unwrap();

        // Should only get responses from Alice and Bob, not Charlie
        assert_eq!(turns.len(), 2);
        let responders: Vec<&str> = turns.iter().map(|t| t.speaker.name()).collect();
        assert!(responders.contains(&"Alice"));
        assert!(responders.contains(&"Bob"));
        assert!(!responders.contains(&"Charlie"));
    }

    #[tokio::test]
    async fn test_mentioned_mode_fallback_to_broadcast() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Alice", vec!["Alice's response".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Bob", vec!["Bob's response".to_string()]),
            );

        // No mentions - should fall back to broadcast mode
        let turns = dialogue
            .run("What does everyone think about this?")
            .await
            .unwrap();

        // Should get responses from all participants (broadcast fallback)
        assert_eq!(turns.len(), 2);
        let responders: Vec<&str> = turns.iter().map(|t| t.speaker.name()).collect();
        assert!(responders.contains(&"Alice"));
        assert!(responders.contains(&"Bob"));
    }

    #[tokio::test]
    async fn test_mentioned_mode_single_mention() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Alice", vec!["Alice's response".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Bob", vec!["Bob's response".to_string()]),
            );

        // Only mention Alice
        let turns = dialogue.run("@Alice can you help?").await.unwrap();

        // Should only get response from Alice
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Alice");
        assert_eq!(turns[0].content, "Alice's response");
    }

    #[tokio::test]
    async fn test_mentioned_mode_multi_turn_context_propagation() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona3 = Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA engineer".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new(
                    "Alice",
                    vec![
                        "Alice: Turn 1 response".to_string(),
                        "Alice: Turn 2 response".to_string(),
                    ],
                ),
            )
            .add_participant(
                persona2,
                MockAgent::new(
                    "Bob",
                    vec![
                        "Bob: Turn 1 response".to_string(),
                        "Bob: Turn 2 response".to_string(),
                    ],
                ),
            )
            .add_participant(
                persona3,
                MockAgent::new("Charlie", vec!["Charlie: Turn 2 response".to_string()]),
            );

        // Turn 1: Mention Alice and Bob
        let turn1 = dialogue
            .run("@Alice @Bob what's your initial thoughts?")
            .await
            .unwrap();

        assert_eq!(turn1.len(), 2);
        let turn1_responders: Vec<&str> = turn1.iter().map(|t| t.speaker.name()).collect();
        assert!(turn1_responders.contains(&"Alice"));
        assert!(turn1_responders.contains(&"Bob"));

        // Turn 2: Mention Charlie - he should see Alice and Bob's Turn 1 responses
        let turn2 = dialogue
            .run("@Charlie what do you think about their responses?")
            .await
            .unwrap();

        assert_eq!(turn2.len(), 1);
        assert_eq!(turn2[0].speaker.name(), "Charlie");

        // Verify history contains all turns
        let history = dialogue.history();
        // Turn 1: System message + Alice + Bob
        // Turn 2: System message + Charlie
        assert_eq!(history.len(), 5);

        // Verify Turn 1 messages
        assert_eq!(history[0].speaker.name(), "System"); // Turn 1 prompt

        // Turn 1 responses (Alice and Bob, order may vary)
        let turn1_names: Vec<&str> = vec![history[1].speaker.name(), history[2].speaker.name()];
        assert!(turn1_names.contains(&"Alice"));
        assert!(turn1_names.contains(&"Bob"));

        // Turn 2 messages
        assert_eq!(history[3].speaker.name(), "System"); // Turn 2 prompt
        assert_eq!(history[4].speaker.name(), "Charlie"); // Turn 2 response
    }

    #[tokio::test]
    async fn test_mentioned_mode_partial_session() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Alice", vec!["Alice's response".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Bob", vec!["Bob's response".to_string()]),
            );

        // Use partial_session with mention
        let mut session = dialogue.partial_session("@Alice what do you think?");

        let mut turns = Vec::new();
        while let Some(result) = session.next_turn().await {
            turns.push(result.unwrap());
        }

        // Should only get response from Alice
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Alice");
        assert_eq!(turns[0].content, "Alice's response");
    }

    #[tokio::test]
    async fn test_mentioned_mode_three_members_progressive() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona3 = Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA engineer".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Alice", vec!["Alice response".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Bob", vec!["Bob response".to_string()]),
            )
            .add_participant(
                persona3,
                MockAgent::new("Charlie", vec!["Charlie response".to_string()]),
            );

        // Mention all three members
        let turns = dialogue
            .run("@Alice @Bob @Charlie everyone needs to respond")
            .await
            .unwrap();

        // All three should respond
        assert_eq!(turns.len(), 3);
        let responders: Vec<&str> = turns.iter().map(|t| t.speaker.name()).collect();
        assert!(responders.contains(&"Alice"));
        assert!(responders.contains(&"Bob"));
        assert!(responders.contains(&"Charlie"));
    }

    #[tokio::test]
    async fn test_mentioned_mode_receives_other_participants_context() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona3 = Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA engineer".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new(
                    "Alice",
                    vec![
                        "Alice: Turn 1".to_string(),
                        "Alice: Turn 2 after seeing Bob".to_string(),
                    ],
                ),
            )
            .add_participant(
                persona2,
                MockAgent::new("Bob", vec!["Bob: Turn 1".to_string()]),
            )
            .add_participant(
                persona3,
                MockAgent::new(
                    "Charlie",
                    vec!["Charlie: Turn 2 after seeing Alice and Bob".to_string()],
                ),
            );

        // Turn 1: Alice and Bob discuss
        let turn1 = dialogue
            .run("@Alice @Bob initial discussion")
            .await
            .unwrap();
        assert_eq!(turn1.len(), 2);

        // Turn 2: Charlie joins and should see both Alice and Bob's Turn 1 responses
        let turn2 = dialogue.run("@Charlie your thoughts?").await.unwrap();
        assert_eq!(turn2.len(), 1);
        assert_eq!(turn2[0].speaker.name(), "Charlie");

        // Verify the full dialogue history is preserved
        let history = dialogue.history();
        // Turn 1: System + Alice + Bob
        // Turn 2: System + Charlie
        assert_eq!(history.len(), 5);

        // Verify message ordering and speakers
        assert_eq!(history[0].speaker.name(), "System");

        let turn1_speakers: Vec<&str> = vec![history[1].speaker.name(), history[2].speaker.name()];
        assert!(turn1_speakers.contains(&"Alice"));
        assert!(turn1_speakers.contains(&"Bob"));

        assert_eq!(history[3].speaker.name(), "System");
        assert_eq!(history[4].speaker.name(), "Charlie");
    }

    #[tokio::test]
    async fn test_payload_with_both_messages_and_text() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "TestAgent".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue.add_participant(
            persona,
            MockAgent::new("TestAgent", vec!["Response".to_string()]),
        );

        // Create a Payload with both Messages and Text
        let payload = Payload::from_messages(vec![
            PayloadMessage::user("Alice", "Product Manager", "What should we do?"),
            PayloadMessage::system("Context: This is a test scenario"),
        ])
        .merge(Payload::text("Additional text content"));

        dialogue.run(payload).await.unwrap();

        // Verify MessageStore contains all content
        let history = dialogue.history();

        // Should have:
        // 1. User message from Alice
        // 2. System message (Context)
        // 3. System message (Additional text)
        // 4. Agent response
        assert_eq!(history.len(), 4);

        // Check first message (User from Alice)
        assert!(matches!(history[0].speaker, Speaker::User { .. }));
        assert_eq!(history[0].speaker.name(), "Alice");
        assert_eq!(history[0].content, "What should we do?");

        // Check second message (System - Context)
        assert!(matches!(history[1].speaker, Speaker::System));
        assert_eq!(history[1].content, "Context: This is a test scenario");

        // Check third message (System - Text)
        assert!(matches!(history[2].speaker, Speaker::System));
        assert_eq!(history[2].content, "Additional text content");

        // Check fourth message (Agent response)
        assert!(matches!(history[3].speaker, Speaker::Agent { .. }));
        assert_eq!(history[3].speaker.name(), "TestAgent");
    }

    #[tokio::test]
    async fn test_mentioned_mode_includes_previous_agent_outputs_in_mention_extraction() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::mentioned();

        let alice_persona = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob_persona = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Visual".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let charlie_persona = Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA engineer".to_string(),
            communication_style: "Detail-oriented".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        dialogue
            .add_participant(
                alice_persona,
                MockAgent::new(
                    "Alice",
                    vec!["I think we should use @Bob's design".to_string()],
                ),
            )
            .add_participant(
                bob_persona,
                MockAgent::new("Bob", vec!["Thanks Alice!".to_string()]),
            )
            .add_participant(
                charlie_persona,
                MockAgent::new("Charlie", vec!["I agree".to_string()]),
            );

        // Turn 1: Mention Alice
        let turn1 = dialogue.run("@Alice what do you think?").await.unwrap();
        assert_eq!(turn1.len(), 1);
        assert_eq!(turn1[0].speaker.name(), "Alice");
        assert_eq!(turn1[0].content, "I think we should use @Bob's design");

        // Turn 2: No explicit mention, but Alice's response mentions @Bob
        // The implementation should extract mentions from previous agent outputs
        let turn2 = dialogue.run("Continue the discussion").await.unwrap();

        // Bob should be activated because Alice mentioned @Bob in turn 1
        assert_eq!(turn2.len(), 1);
        assert_eq!(turn2[0].speaker.name(), "Bob");
        assert_eq!(turn2[0].content, "Thanks Alice!");

        // Verify history structure
        let history = dialogue.history();
        // Turn 1: System + Alice
        // Turn 2: System + Bob
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].speaker.name(), "System");
        assert_eq!(history[1].speaker.name(), "Alice");
        assert_eq!(history[2].speaker.name(), "System");
        assert_eq!(history[3].speaker.name(), "Bob");
    }

    // Helper agent that records received payloads for testing
    #[derive(Clone)]
    struct RecordingAgent {
        name: String,
        response: String,
        received_payloads: std::sync::Arc<std::sync::Mutex<Vec<Payload>>>,
    }

    impl RecordingAgent {
        fn new(name: impl Into<String>, response: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                response: response.into(),
                received_payloads: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        fn get_received_payloads(&self) -> Vec<Payload> {
            self.received_payloads.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Agent for RecordingAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            const EXPERTISE: &str = "Recording agent for testing";
            &EXPERTISE
        }

        fn name(&self) -> String {
            self.name.clone()
        }

        async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
            self.received_payloads.lock().unwrap().push(payload);
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_reaction_strategy_broadcast_context_info() {
        use crate::agent::dialogue::message::{MessageMetadata, MessageType};
        use crate::agent::persona::Persona;

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Assistant".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent = RecordingAgent::new("Agent1", "I can help with that");

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(persona, agent.clone());

        // Turn 1: Send ContextInfo message (should not trigger reaction)
        let context_payload = Payload::new().add_message_with_metadata(
            Speaker::System,
            "Analysis completed: 3 issues found",
            MessageMetadata::new().with_type(MessageType::ContextInfo),
        );

        let turns = dialogue.run(context_payload).await.unwrap();
        assert_eq!(turns.len(), 0, "ContextInfo should not trigger reaction");

        // Verify ContextInfo is stored for history/reference
        let history = dialogue.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "Analysis completed: 3 issues found");

        // Turn 2: Send User message (should trigger reaction)
        let user_payload = Payload::from_messages(vec![PayloadMessage::new(
            Speaker::user("Alice", "User"),
            "Tell me more about the issues",
        )]);

        let turns = dialogue.run(user_payload).await.unwrap();
        assert_eq!(turns.len(), 1, "User message should trigger reaction");
        assert_eq!(turns[0].speaker.name(), "Agent1");
        assert_eq!(turns[0].content, "I can help with that");

        // Verify agent received both ContextInfo and User message
        let received = agent.get_received_payloads();
        assert_eq!(received.len(), 1, "Agent should have been called once");

        let received_messages = received[0].to_messages();
        assert_eq!(
            received_messages.len(),
            2,
            "Agent should receive both ContextInfo and User message"
        );

        // First message should be ContextInfo
        assert_eq!(received_messages[0].speaker.name(), "System");
        assert_eq!(
            received_messages[0].content,
            "Analysis completed: 3 issues found"
        );
        assert!(
            received_messages[0]
                .metadata
                .is_type(&MessageType::ContextInfo),
            "Metadata should be preserved"
        );

        // Second message should be User message
        assert_eq!(received_messages[1].speaker.name(), "Alice");
        assert_eq!(
            received_messages[1].content,
            "Tell me more about the issues"
        );
    }

    #[tokio::test]
    async fn test_reaction_strategy_sequential_context_info() {
        use crate::agent::dialogue::message::{MessageMetadata, MessageType};
        use crate::agent::persona::Persona;

        let persona1 = Persona {
            name: "Agent1".to_string(),
            role: "Analyzer".to_string(),
            background: "First agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Reviewer".to_string(),
            background: "Second agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent1 = RecordingAgent::new("Agent1", "Analysis done");
        let agent2 = RecordingAgent::new("Agent2", "Review complete");

        let mut dialogue = Dialogue::sequential();
        dialogue
            .add_participant(persona1, agent1.clone())
            .add_participant(persona2, agent2.clone());

        // Turn 1: ContextInfo (should not trigger reaction, even with Always strategy)
        let context_payload = Payload::new().add_message_with_metadata(
            Speaker::System,
            "Background: Project uses Rust",
            MessageMetadata::new().with_type(MessageType::ContextInfo),
        );

        let turns = dialogue.run(context_payload).await.unwrap();
        assert_eq!(turns.len(), 0, "ContextInfo should not trigger reactions");

        // Turn 2: User message (triggers sequential execution)
        let user_payload = Payload::from_messages(vec![PayloadMessage::new(
            Speaker::user("Bob", "User"),
            "Analyze the code",
        )]);

        let turns = dialogue.run(user_payload).await.unwrap();
        assert_eq!(turns.len(), 1); // Sequential returns only final turn
        assert_eq!(turns[0].speaker.name(), "Agent2");

        // Verify Agent1 executed once (Turn 2: User message; Turn 1: ContextInfo did not trigger)
        let agent1_received = agent1.get_received_payloads();
        assert_eq!(
            agent1_received.len(),
            1,
            "Agent1 should execute once for User message"
        );

        // Agent1 received ContextInfo in history + User message
        let agent1_payload = agent1_received[0].to_messages();
        assert!(
            agent1_payload
                .iter()
                .any(|m| m.content == "Analyze the code"),
            "Should contain user message"
        );
        assert!(
            agent1_payload
                .iter()
                .any(|m| m.content == "Background: Project uses Rust"),
            "ContextInfo should be in history"
        );

        // Verify Agent2 executed once as well (sequential: receives Agent1's output + user input)
        let agent2_received = agent2.get_received_payloads();
        assert_eq!(agent2_received.len(), 1, "Agent2 should execute once");
    }

    #[tokio::test]
    async fn test_reaction_strategy_mentioned_context_info() {
        use crate::agent::dialogue::message::{MessageMetadata, MessageType};
        use crate::agent::persona::Persona;

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "First agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Second agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent1 = RecordingAgent::new("Alice", "I'll handle it");
        let agent2 = RecordingAgent::new("Bob", "Sounds good");

        let mut dialogue = Dialogue::mentioned();
        dialogue
            .add_participant(persona1, agent1.clone())
            .add_participant(persona2, agent2.clone());

        // Turn 1: ContextInfo (should not trigger reaction, even with Always strategy)
        let context_payload = Payload::new().add_message_with_metadata(
            Speaker::System,
            "Note: Use async/await",
            MessageMetadata::new().with_type(MessageType::ContextInfo),
        );

        let turns = dialogue.run(context_payload).await.unwrap();
        assert_eq!(turns.len(), 0, "ContextInfo should not trigger reactions");

        // Turn 2: Mention only Alice
        let user_payload = Payload::from_messages(vec![PayloadMessage::new(
            Speaker::user("User", "User"),
            "@Alice can you implement this?",
        )]);

        let turns = dialogue.run(user_payload).await.unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Alice");

        // Verify Alice executed once (Turn 2: @mention, Turn 1: ContextInfo did not trigger)
        let alice_received = agent1.get_received_payloads();
        assert_eq!(
            alice_received.len(),
            1,
            "Alice should execute once for @mention"
        );

        // Alice received ContextInfo in history + @mention message
        let alice_payload = alice_received[0].to_messages();
        assert!(alice_payload.iter().any(|m| m.content.contains("@Alice")));
        // ContextInfo should be in history but did not trigger this execution
        assert!(
            alice_payload
                .iter()
                .any(|m| m.content == "Note: Use async/await")
        );

        // Verify Bob did not execute (no mentions for Bob)
        let bob_received = agent2.get_received_payloads();
        assert_eq!(
            bob_received.len(),
            0,
            "Bob should not execute (no mentions)"
        );
    }

    #[tokio::test]
    async fn test_reaction_strategy_partial_session() {
        use crate::agent::dialogue::message::{MessageMetadata, MessageType};
        use crate::agent::persona::Persona;

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Assistant".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent = RecordingAgent::new("Agent1", "Understood");

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(persona, agent.clone());

        // Turn 1: ContextInfo via partial_session
        let context_payload = Payload::new().add_message_with_metadata(
            Speaker::System,
            "System ready",
            MessageMetadata::new().with_type(MessageType::ContextInfo),
        );

        let mut session = dialogue.partial_session(context_payload);
        let mut turn_count = 0;
        while let Some(result) = session.next_turn().await {
            result.unwrap();
            turn_count += 1;
        }
        assert_eq!(turn_count, 0, "ContextInfo should not produce turns");

        // Turn 2: User message via partial_session
        let user_payload = Payload::from_messages(vec![PayloadMessage::new(
            Speaker::user("User", "User"),
            "Hello",
        )]);

        let mut session = dialogue.partial_session(user_payload);
        let mut turn_count = 0;
        while let Some(result) = session.next_turn().await {
            let turn = result.unwrap();
            assert_eq!(turn.speaker.name(), "Agent1");
            assert_eq!(turn.content, "Understood");
            turn_count += 1;
        }
        assert_eq!(turn_count, 1);

        // Verify agent received both messages
        let received = agent.get_received_payloads();
        assert_eq!(received.len(), 1);
        let messages = received[0].to_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "System ready");
        assert!(messages[0].metadata.is_type(&MessageType::ContextInfo));
        assert_eq!(messages[1].content, "Hello");
    }

    #[tokio::test]
    async fn test_multiple_context_info_accumulation() {
        use crate::agent::dialogue::message::{MessageMetadata, MessageType};
        use crate::agent::persona::Persona;

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Assistant".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent = RecordingAgent::new("Agent1", "Got it");

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(persona, agent.clone());

        // Send multiple ContextInfo messages
        for i in 1..=3 {
            let context_payload = Payload::new().add_message_with_metadata(
                Speaker::System,
                format!("Context {}", i),
                MessageMetadata::new().with_type(MessageType::ContextInfo),
            );
            let turns = dialogue.run(context_payload).await.unwrap();
            assert_eq!(turns.len(), 0);
        }

        // Verify all ContextInfo messages are stored
        let history = dialogue.history();
        assert_eq!(history.len(), 3);

        // Send User message
        let user_payload = Payload::from_messages(vec![PayloadMessage::new(
            Speaker::user("User", "User"),
            "Process all context",
        )]);

        let turns = dialogue.run(user_payload).await.unwrap();
        assert_eq!(turns.len(), 1);

        // Verify agent received all accumulated ContextInfo + User message
        let received = agent.get_received_payloads();
        assert_eq!(received.len(), 1);
        let messages = received[0].to_messages();
        assert_eq!(
            messages.len(),
            4,
            "Should receive 3 ContextInfo + 1 User message"
        );

        // Verify all ContextInfo messages are present with correct metadata
        for (i, message) in messages.iter().enumerate().take(3) {
            assert_eq!(message.content, format!("Context {}", i + 1));
            assert!(message.metadata.is_type(&MessageType::ContextInfo));
        }
        assert_eq!(messages[3].content, "Process all context");
    }

    #[tokio::test]
    async fn test_add_agent_with_context_config() {
        use crate::agent::persona::{ContextConfig, PersonaAgent};

        let mock_agent = MockAgent::new("Alice", vec!["Response from Alice".to_string()]);

        let persona = Persona {
            name: "Alice".to_string(),
            role: "Engineer".to_string(),
            background: "Senior developer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        // Configure PersonaAgent before adding to dialogue
        let config = ContextConfig {
            long_conversation_threshold: 1000,
            recent_messages_count: 10,
            participants_after_context: true,
            include_trailing_prompt: true,
        };

        let persona_agent =
            PersonaAgent::new(mock_agent, persona.clone()).with_context_config(config);

        // Use add_agent instead of add_participant
        let mut dialogue = Dialogue::sequential();
        dialogue.add_agent(persona, persona_agent);

        // Verify participant was added
        assert_eq!(dialogue.participant_count(), 1);
        assert_eq!(dialogue.participant_names(), vec!["Alice"]);

        // Execute dialogue
        let payload = Payload::text("Test message");
        let turns = dialogue.run(payload).await.unwrap();

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.name(), "Alice");
        assert_eq!(turns[0].content, "Response from Alice");
    }

    // ========================================================================
    // Tests for Dialogue as Agent
    // ========================================================================

    #[tokio::test]
    async fn test_dialogue_as_agent_basic() {
        // Create a dialogue with mock agents
        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Engineer".to_string(),
            background: "Senior developer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent1 = MockAgent::new("Alice", vec!["Technical perspective".to_string()]);
        let agent2 = MockAgent::new("Bob", vec!["Design perspective".to_string()]);

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(persona1, agent1);
        dialogue.add_participant(persona2, agent2);

        // Use Dialogue as an Agent
        let agent_name = dialogue.name();
        assert!(agent_name.contains("Broadcast"));
        assert!(agent_name.contains("2 participants"));

        let expertise = dialogue.expertise();
        assert!(expertise.contains("Multi-agent dialogue"));

        // Execute as agent
        let payload = Payload::text("Discuss the new feature");
        let output = dialogue.execute(payload).await.unwrap();

        // Should return Vec<DialogueTurn>
        assert_eq!(output.len(), 2);
        assert!(output.iter().any(|turn| turn.speaker.name() == "Alice"));
        assert!(output.iter().any(|turn| turn.speaker.name() == "Bob"));
    }

    #[tokio::test]
    async fn test_dialogue_as_agent_sequential() {
        let persona1 = Persona {
            name: "Analyzer".to_string(),
            role: "Data Analyst".to_string(),
            background: "Statistics expert".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Writer".to_string(),
            role: "Technical Writer".to_string(),
            background: "Documentation specialist".to_string(),
            communication_style: "Clear and concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent1 = MockAgent::new("Analyzer", vec!["Data shows trend X".to_string()]);
        let agent2 = MockAgent::new("Writer", vec!["Documented the findings".to_string()]);

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(persona1, agent1);
        dialogue.add_participant(persona2, agent2);

        // Verify name reflects execution model
        let agent_name = dialogue.name();
        assert!(agent_name.contains("Sequential"));

        // Execute as agent
        let payload = Payload::text("Analyze and document the data");
        let output = dialogue.execute(payload).await.unwrap();

        // Sequential returns only the last turn
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].speaker.name(), "Writer");
        assert_eq!(output[0].content, "Documented the findings");
    }

    #[tokio::test]
    async fn test_dialogue_as_agent_clone_independence() {
        // Verify that cloning for Agent::execute doesn't affect original dialogue
        let persona = Persona {
            name: "Agent".to_string(),
            role: "Assistant".to_string(),
            background: "Helper".to_string(),
            communication_style: "Friendly".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent = MockAgent::new("Agent", vec!["Response 1".to_string()]);

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(persona, agent);

        // Execute multiple times as Agent
        let payload1 = Payload::text("First request");
        let output1 = dialogue.execute(payload1).await.unwrap();
        assert_eq!(output1.len(), 1);

        let payload2 = Payload::text("Second request");
        let output2 = dialogue.execute(payload2).await.unwrap();
        assert_eq!(output2.len(), 1);

        // Each execution should be independent (cloned dialogue)
        // Original dialogue should remain unchanged
        assert_eq!(dialogue.history().len(), 0);
    }

    // ========================================================================
    // Tests for Moderator Execution Model
    // ========================================================================

    #[tokio::test]
    async fn test_moderator_execution_model() {
        // Create a mock moderator that always returns OrderedSequential
        #[derive(Clone)]
        struct MockModerator;

        #[async_trait]
        impl Agent for MockModerator {
            type Output = ExecutionModel;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Mock moderator for testing";
                &EXPERTISE
            }

            fn name(&self) -> String {
                "MockModerator".to_string()
            }

            async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
                // Always return Sequential with explicit order: Bob -> Alice
                Ok(ExecutionModel::OrderedSequential(
                    SequentialOrder::Explicit(vec!["Bob".to_string(), "Alice".to_string()]),
                ))
            }
        }

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Engineer".to_string(),
            background: "Developer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent1 = MockAgent::new("Alice", vec!["Alice's response".to_string()]);
        let agent2 = MockAgent::new("Bob", vec!["Bob's response".to_string()]);

        let mut dialogue = Dialogue::moderator();
        dialogue.with_moderator(MockModerator);
        dialogue.add_participant(persona1, agent1);
        dialogue.add_participant(persona2, agent2);

        // Execute - moderator should decide to use Sequential with Bob -> Alice order
        let payload = Payload::text("Discuss the feature");
        let output = dialogue.run(payload).await.unwrap();

        // Sequential returns only the last turn (Alice)
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].speaker.name(), "Alice");
        assert_eq!(output[0].content, "Alice's response");
    }

    #[tokio::test]
    async fn test_moderator_without_agent_fails() {
        let persona = Persona {
            name: "Alice".to_string(),
            role: "Engineer".to_string(),
            background: "Developer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent = MockAgent::new("Alice", vec!["Response".to_string()]);

        // Create dialogue with Moderator but without setting moderator agent
        let mut dialogue = Dialogue::moderator();
        dialogue.add_participant(persona, agent);

        // Should fail because moderator is not set
        let payload = Payload::text("Test");
        let result = dialogue.run(payload).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AgentError::ExecutionFailed(msg) => {
                assert!(msg.contains("moderator agent"));
            }
            _ => panic!("Expected ExecutionFailed error"),
        }
    }

    #[tokio::test]
    async fn test_moderator_prevents_infinite_recursion() {
        // Create a moderator that returns Moderator (invalid)
        #[derive(Clone)]
        struct BadModerator;

        #[async_trait]
        impl Agent for BadModerator {
            type Output = ExecutionModel;
            type Expertise = &'static str;

            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "Bad moderator";
                &EXPERTISE
            }

            fn name(&self) -> String {
                "BadModerator".to_string()
            }

            async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
                Ok(ExecutionModel::Moderator) // Invalid: returns Moderator
            }
        }

        let persona = Persona {
            name: "Alice".to_string(),
            role: "Engineer".to_string(),
            background: "Developer".to_string(),
            communication_style: "Technical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent = MockAgent::new("Alice", vec!["Response".to_string()]);

        let mut dialogue = Dialogue::moderator();
        dialogue.with_moderator(BadModerator);
        dialogue.add_participant(persona, agent);

        // Should fail to prevent infinite recursion
        let payload = Payload::text("Test");
        let result = dialogue.run(payload).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AgentError::ExecutionFailed(msg) => {
                assert!(msg.contains("infinite recursion"));
            }
            _ => panic!("Expected ExecutionFailed error about infinite recursion"),
        }
    }

    #[tokio::test]
    async fn test_join_in_progress_with_fresh_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec!["Alice turn 1".to_string(), "Alice turn 2".to_string()],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec!["Bob turn 1".to_string(), "Bob turn 2".to_string()],
        );

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(alice, agent_alice.clone());
        dialogue.add_participant(bob, agent_bob.clone());

        // Turn 1: Initial conversation
        let _turn1 = dialogue.run("What's the plan?").await.unwrap();

        // Turn 2: Conversation continues
        let _turn2 = dialogue.run("Let's proceed").await.unwrap();

        // Now add a consultant mid-dialogue with Fresh strategy (no history)
        let consultant = Persona {
            name: "Carol".to_string(),
            role: "Security Consultant".to_string(),
            background: "Security expert".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new("Carol", vec!["Carol's fresh perspective".to_string()]);
        let carol_clone = agent_carol.clone();

        // Join with Fresh strategy - should see NO history
        dialogue.join_in_progress(consultant, agent_carol, JoiningStrategy::Fresh);

        // Turn 3: Carol participates
        let turn3 = dialogue.run("Carol, what do you think?").await.unwrap();

        // Verify Carol responded
        assert!(turn3.iter().any(|t| t.speaker.name() == "Carol"));

        // Verify Carol was called exactly once
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received NO historical messages (Fresh strategy)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Count messages from Turn 1 and Turn 2 (Alice and Bob's responses)
        // Fresh strategy means Carol should NOT see these
        let historical_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
            .collect();

        assert_eq!(
            historical_messages.len(),
            0,
            "Fresh strategy: Carol should not see historical messages from Alice or Bob"
        );

        // Fresh strategy: Carol should NOT see any messages on first turn (only Persona/Participants)
        let system_messages: Vec<_> = messages
            .iter()
            .filter(|msg| matches!(msg.speaker, Speaker::System))
            .collect();

        assert!(
            system_messages.is_empty(),
            "Fresh strategy: Carol should NOT see any system messages on initial turn"
        );

        // Turn 4: Verify Carol now receives Turn 3 messages in subsequent turns
        let turn4 = dialogue.run("Let's continue").await.unwrap();
        assert!(turn4.iter().any(|t| t.speaker.name() == "Carol"));

        // Carol should have been called twice now
        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // Now Carol should see Alice and Bob's Turn 3 responses (unsent messages)
        let turn3_agent_messages: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| {
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && !matches!(msg.speaker, Speaker::System)
            })
            .collect();

        assert!(
            !turn3_agent_messages.is_empty(),
            "In Turn 4, Carol should see Turn 3 messages from Alice and Bob"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_with_full_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        let alice = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice turn 1".to_string(),
                "Alice turn 2".to_string(),
                "Alice turn 3".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(alice, agent_alice);

        // Turn 1 & 2: Build up some history
        let _turn1 = dialogue.run("First topic").await.unwrap();
        let _turn2 = dialogue.run("Second topic").await.unwrap();

        // Verify Alice was called twice
        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice should be called twice"
        );

        // Join with Full strategy - should see ALL history
        let bob = Persona {
            name: "Bob".to_string(),
            role: "New Member".to_string(),
            background: "Just joined".to_string(),
            communication_style: "Curious".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_bob = MockAgent::new(
            "Bob",
            vec!["Bob caught up".to_string(), "Bob turn 4".to_string()],
        );
        let bob_clone = agent_bob.clone();

        dialogue.join_in_progress(bob, agent_bob, JoiningStrategy::Full);

        // Turn 3: Bob participates
        let turn3 = dialogue.run("Bob, your thoughts?").await.unwrap();

        assert!(turn3.iter().any(|t| t.speaker.name() == "Bob"));

        // Verify Bob was called once
        assert_eq!(bob_clone.get_call_count(), 1, "Bob should be called once");

        // Verify Bob received FULL history (Alice's Turn 1 and Turn 2 responses)
        let bob_payloads = bob_clone.get_payloads();
        assert_eq!(bob_payloads.len(), 1, "Bob should receive 1 payload");

        let bob_first_payload = &bob_payloads[0];
        let messages = bob_first_payload.to_messages();

        // Bob should see ALL of Alice's previous responses (Turn 1 and Turn 2)
        let alice_historical_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_historical_messages.len(),
            2,
            "Full strategy: Bob should see ALL 2 historical messages from Alice (Turn 1 and Turn 2)"
        );

        // Verify the content of Alice's historical messages
        let alice_contents: Vec<&str> = alice_historical_messages
            .iter()
            .map(|msg| msg.content.as_str())
            .collect();

        assert!(
            alice_contents.contains(&"Alice turn 1"),
            "Bob should see Alice's Turn 1 response"
        );
        assert!(
            alice_contents.contains(&"Alice turn 2"),
            "Bob should see Alice's Turn 2 response"
        );

        // Turn 4: Verify Bob receives only new messages (not historical ones again)
        let turn4 = dialogue.run("Let's continue").await.unwrap();
        assert!(turn4.iter().any(|t| t.speaker.name() == "Bob"));

        // Bob should have been called twice now
        assert_eq!(bob_clone.get_call_count(), 2, "Bob should be called twice");

        let bob_second_payload = &bob_clone.get_payloads()[1];
        let turn4_messages = bob_second_payload.to_messages();

        // Bob should only see Turn 3 messages (Alice's Turn 3 response)
        // NOT Turn 1 and Turn 2 again (those were already marked as sent)
        let alice_turn4_messages: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_turn4_messages.len(),
            1,
            "In Turn 4, Bob should only see Alice's Turn 3 response (not Turn 1 and 2 again)"
        );

        // Verify it's Alice's Turn 3 response
        assert_eq!(
            alice_turn4_messages[0].content, "Alice turn 3",
            "Bob should see Alice's Turn 3 response in Turn 4"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_with_recent_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        let alice = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice turn 1".to_string(),
                "Alice turn 2".to_string(),
                "Alice turn 3".to_string(),
                "Alice turn 4".to_string(),
                "Alice turn 5".to_string(),
                "Alice turn 6".to_string(),
                "Alice turn 7".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();

        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(alice, agent_alice);

        // Build up 5 turns of history
        for i in 1..=5 {
            let _ = dialogue.run(format!("Message {}", i)).await.unwrap();
        }

        // Verify Alice was called 5 times
        assert_eq!(
            alice_clone.get_call_count(),
            5,
            "Alice should be called 5 times"
        );

        // Join with Recent(2) strategy - should only see last 2 turns (Turn 4 and Turn 5)
        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Focused".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_bob = MockAgent::new(
            "Bob",
            vec!["Bob reviews recent".to_string(), "Bob turn 7".to_string()],
        );
        let bob_clone = agent_bob.clone();

        dialogue.join_in_progress(bob, agent_bob, JoiningStrategy::recent_with_turns(2));

        // Turn 6: Bob participates
        let turn6 = dialogue.run("Bob, review please").await.unwrap();

        assert!(turn6.iter().any(|t| t.speaker.name() == "Bob"));

        // Verify Bob was called once
        assert_eq!(bob_clone.get_call_count(), 1, "Bob should be called once");

        // Verify Bob received ONLY recent 2 turns (Turn 4 and Turn 5, NOT Turn 1-3)
        let bob_payloads = bob_clone.get_payloads();
        assert_eq!(bob_payloads.len(), 1, "Bob should receive 1 payload");

        let bob_first_payload = &bob_payloads[0];
        let messages = bob_first_payload.to_messages();

        // Bob should see the last 2 Alice responses (Turn 4 and Turn 5)
        let alice_historical_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_historical_messages.len(),
            2,
            "Recent(2) strategy: Bob should see 2 recent historical messages from Alice"
        );

        // Verify the content - should be Turn 4 and Turn 5
        let alice_contents: Vec<&str> = alice_historical_messages
            .iter()
            .map(|msg| msg.content.as_str())
            .collect();

        assert!(
            alice_contents.contains(&"Alice turn 4"),
            "Bob should see Alice's Turn 4 response"
        );
        assert!(
            alice_contents.contains(&"Alice turn 5"),
            "Bob should see Alice's Turn 5 response"
        );

        // Bob should NOT see Turn 1, 2, or 3
        assert!(
            !alice_contents.contains(&"Alice turn 1"),
            "Bob should NOT see Alice's Turn 1 response (too old)"
        );
        assert!(
            !alice_contents.contains(&"Alice turn 2"),
            "Bob should NOT see Alice's Turn 2 response (too old)"
        );
        assert!(
            !alice_contents.contains(&"Alice turn 3"),
            "Bob should NOT see Alice's Turn 3 response (too old)"
        );

        // Turn 7: Verify Bob receives only new messages
        let turn7 = dialogue.run("Let's continue").await.unwrap();
        assert!(turn7.iter().any(|t| t.speaker.name() == "Bob"));

        // Bob should have been called twice now
        assert_eq!(bob_clone.get_call_count(), 2, "Bob should be called twice");

        let bob_second_payload = &bob_clone.get_payloads()[1];
        let turn7_messages = bob_second_payload.to_messages();

        // Bob should only see Turn 6 messages (Alice's Turn 6 response)
        let alice_turn7_messages: Vec<_> = turn7_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_turn7_messages.len(),
            1,
            "In Turn 7, Bob should only see Alice's Turn 6 response"
        );

        assert_eq!(
            alice_turn7_messages[0].content, "Alice turn 6",
            "Bob should see Alice's Turn 6 response in Turn 7"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_mentioned_mode_with_fresh_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice turn 1".to_string(),
                "Alice turn 2".to_string(),
                "Alice turn 3".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec!["Bob turn 1".to_string(), "Bob turn 2".to_string()],
        );
        let alice_clone = agent_alice.clone();

        let mut dialogue = Dialogue::mentioned();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Turn 1: Mention Alice
        let _turn1 = dialogue.run("@Alice what's the plan?").await.unwrap();

        // Turn 2: Mention Bob
        let _turn2 = dialogue.run("@Bob your thoughts?").await.unwrap();

        // Verify Alice was called only once (Turn 1)
        assert_eq!(
            alice_clone.get_call_count(),
            1,
            "Alice should be called once (only Turn 1)"
        );

        // Now add Carol mid-dialogue with Fresh strategy
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Security Consultant".to_string(),
            background: "Security expert".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol's fresh analysis".to_string(),
                "Carol turn 4".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::Fresh);

        // Turn 3: Mention Carol (first time she participates)
        let turn3 = dialogue.run("@Carol security review please").await.unwrap();

        assert!(turn3.iter().any(|t| t.speaker.name() == "Carol"));

        // Verify Carol was called once
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received NO historical messages (Fresh strategy)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should NOT see Alice or Bob's previous responses
        let historical_agent_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
            .collect();

        assert_eq!(
            historical_agent_messages.len(),
            0,
            "Fresh strategy in mentioned mode: Carol should not see historical messages"
        );

        // Turn 4: Mention both Carol and Alice
        let _turn4 = dialogue
            .run("@Carol and @Alice continue discussion")
            .await
            .unwrap();

        // Carol should be called twice now
        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );
        // Alice should be called twice now (Turn 1 and Turn 4)
        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // In Turn 4, Carol and Alice execute in parallel
        // Carol should NOT see Alice's Turn 4 response (they execute concurrently)
        // Carol should only see Turn 4 incoming message
        let alice_messages: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_messages.len(),
            0,
            "In Turn 4, Carol should NOT see Alice's Turn 4 response (parallel execution)"
        );

        // Verify Carol does NOT see Alice's Turn 1 response (historical, marked as sent)
        let has_alice_turn1 = turn4_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Alice" && msg.content.contains("turn 1"));

        assert!(
            !has_alice_turn1,
            "Carol should NOT see Alice's Turn 1 response in Turn 4 (marked as sent)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_sequential_mode_with_fresh_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants for sequential processing
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Critical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice analyzed: turn 1".to_string(),
                "Alice analyzed: turn 2".to_string(),
                "Alice analyzed: turn 3".to_string(),
                "Alice analyzed: turn 4".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob reviewed: turn 1".to_string(),
                "Bob reviewed: turn 2".to_string(),
                "Bob reviewed: turn 3".to_string(),
                "Bob reviewed: turn 4".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();
        let bob_clone = agent_bob.clone();

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Turn 1: Alice → Bob (sequential)
        let _turn1 = dialogue.run("Analyze this data").await.unwrap();

        // Turn 2: Alice → Bob (sequential)
        let _turn2 = dialogue.run("Continue analysis").await.unwrap();

        // Verify Alice and Bob were called twice each
        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice should be called twice"
        );
        assert_eq!(bob_clone.get_call_count(), 2, "Bob should be called twice");

        // Add Carol mid-dialogue with Fresh strategy
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Summarizer".to_string(),
            background: "Summary specialist".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol summarized".to_string(),
                "Carol summary 2".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        // Carol joins with Fresh strategy (joins at end of sequence)
        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::Fresh);

        // Turn 3: Alice → Bob → Carol (sequential, Carol is now at the end)
        let turn3 = dialogue.run("Final analysis").await.unwrap();

        // In sequential mode, only the last agent's output is returned
        assert_eq!(
            turn3.len(),
            1,
            "Sequential mode returns only last agent's output"
        );
        assert_eq!(
            turn3[0].speaker.name(),
            "Carol",
            "Last agent should be Carol"
        );

        // Verify all agents were called
        assert_eq!(
            alice_clone.get_call_count(),
            3,
            "Alice should be called 3 times"
        );
        assert_eq!(
            bob_clone.get_call_count(),
            3,
            "Bob should be called 3 times"
        );
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received NO historical messages (Fresh strategy)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should NOT see Turn 1 or Turn 2 messages from Alice or Bob
        let turn1_turn2_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && (content.contains("turn 1") || content.contains("turn 2"))
            })
            .collect();

        assert_eq!(
            turn1_turn2_messages.len(),
            0,
            "Fresh strategy in sequential mode: Carol should not see Turn 1 or Turn 2 historical messages"
        );

        // Carol SHOULD see Bob's Turn 3 output (immediate predecessor in the chain)
        let bob_turn3_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 3"))
            .collect();

        assert_eq!(
            bob_turn3_messages.len(),
            1,
            "Carol should see Bob's Turn 3 output (her immediate input in sequential chain)"
        );

        // Turn 4: Verify Carol receives only differential updates
        let _turn4 = dialogue.run("Continue").await.unwrap();

        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // Carol should see Bob's Turn 4 output (current chain input)
        // but NOT Turn 1, 2, or 3 historical messages
        let bob_historical: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Bob"
                    && (msg.content.contains("turn 1") || msg.content.contains("turn 2"))
            })
            .collect();

        assert_eq!(
            bob_historical.len(),
            0,
            "In Turn 4, Carol should NOT see Bob's Turn 1 or Turn 2 (marked as sent)"
        );

        // Carol should see Bob's Turn 4 output (new message in the chain)
        let bob_turn4: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 4"))
            .collect();

        assert_eq!(
            bob_turn4.len(),
            1,
            "In Turn 4, Carol should see Bob's Turn 4 output (new chain input)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_partial_session_sequential_with_fresh_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Critical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice analyzed: turn 1".to_string(),
                "Alice analyzed: turn 2".to_string(),
                "Alice analyzed: turn 3".to_string(),
                "Alice analyzed: turn 4".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob reviewed: turn 1".to_string(),
                "Bob reviewed: turn 2".to_string(),
                "Bob reviewed: turn 3".to_string(),
                "Bob reviewed: turn 4".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Turn 1: Use partial_session (streaming API)
        let mut session1 = dialogue.partial_session("Analyze this");
        while let Some(turn) = session1.next_turn().await {
            turn.unwrap();
        }

        // Turn 2: Continue with partial_session
        let mut session2 = dialogue.partial_session("Continue analysis");
        while let Some(turn) = session2.next_turn().await {
            turn.unwrap();
        }

        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice called twice via partial_session"
        );

        // Add Carol with Fresh strategy
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Summarizer".to_string(),
            background: "Summary specialist".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol summarized".to_string(),
                "Carol summary 2".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::Fresh);

        // Turn 3: Use partial_session with Carol
        let mut session3 = dialogue.partial_session("Final summary");
        let mut turn_count = 0;
        while let Some(turn) = session3.next_turn().await {
            turn.unwrap();
            turn_count += 1;
        }

        // Sequential mode executes all participants (Alice → Bob → Carol)
        assert_eq!(
            turn_count, 3,
            "Should have 3 turns in sequential partial_session"
        );
        assert_eq!(carol_clone.get_call_count(), 1, "Carol called once");

        // Verify Carol received NO historical messages (Fresh strategy)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should NOT see Turn 1 or Turn 2 historical messages
        let historical_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && (content.contains("turn 1") || content.contains("turn 2"))
            })
            .collect();

        assert_eq!(
            historical_messages.len(),
            0,
            "Fresh strategy in partial_session sequential: Carol should not see Turn 1 or Turn 2"
        );

        // Carol SHOULD see Bob's Turn 3 output (immediate predecessor in chain)
        let bob_turn3: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 3"))
            .collect();

        assert_eq!(
            bob_turn3.len(),
            1,
            "Carol should see Bob's Turn 3 output via partial_session"
        );

        // Turn 4: Verify differential updates via partial_session
        let mut session4 = dialogue.partial_session("Continue");
        while let Some(turn) = session4.next_turn().await {
            turn.unwrap();
        }

        assert_eq!(carol_clone.get_call_count(), 2, "Carol called twice");

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // Carol should NOT see Turn 1, 2 historical messages
        let historical_turn4: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Bob"
                    && (msg.content.contains("turn 1") || msg.content.contains("turn 2"))
            })
            .collect();

        assert_eq!(
            historical_turn4.len(),
            0,
            "In Turn 4 via partial_session, Carol should NOT see Turn 1 or Turn 2 (marked as sent)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_mentioned_mode_with_full_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice turn 1".to_string(),
                "Alice turn 2".to_string(),
                "Alice turn 3".to_string(),
                "Alice turn 4".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob turn 1".to_string(),
                "Bob turn 2".to_string(),
                "Bob turn 3".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();
        let bob_clone = agent_bob.clone();

        let mut dialogue = Dialogue::mentioned();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Turn 1: Mention Alice
        let _turn1 = dialogue.run("@Alice what's the plan?").await.unwrap();

        // Turn 2: Mention Bob
        let _turn2 = dialogue.run("@Bob your thoughts?").await.unwrap();

        // Verify call counts
        assert_eq!(
            alice_clone.get_call_count(),
            1,
            "Alice should be called once (Turn 1)"
        );
        assert_eq!(
            bob_clone.get_call_count(),
            1,
            "Bob should be called once (Turn 2)"
        );

        // Add Carol mid-dialogue with Full strategy - should see ALL history
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Security Consultant".to_string(),
            background: "Security expert".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol's full analysis".to_string(),
                "Carol turn 4".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::Full);

        // Turn 3: Mention Carol (first time she participates)
        let turn3 = dialogue.run("@Carol security review please").await.unwrap();

        assert!(turn3.iter().any(|t| t.speaker.name() == "Carol"));
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received FULL history (Alice's Turn 1 and Bob's Turn 2)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should see ALL historical messages from Alice and Bob
        let alice_historical: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();
        let bob_historical: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob")
            .collect();

        assert_eq!(
            alice_historical.len(),
            1,
            "Full strategy in mentioned mode: Carol should see Alice's Turn 1"
        );
        assert_eq!(
            bob_historical.len(),
            1,
            "Full strategy in mentioned mode: Carol should see Bob's Turn 2"
        );

        // Verify content
        assert_eq!(alice_historical[0].content, "Alice turn 1");
        assert_eq!(bob_historical[0].content, "Bob turn 1");

        // Turn 4: Mention both Carol and Alice
        let _turn4 = dialogue
            .run("@Carol and @Alice continue discussion")
            .await
            .unwrap();

        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );
        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // In Turn 4, Carol and Alice execute in parallel
        // Carol should NOT see Alice's Turn 4 response (concurrent execution)
        // Carol should only see historical messages that haven't been marked as sent yet
        let alice_turn4_messages: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_turn4_messages.len(),
            0,
            "In Turn 4, Carol should NOT see Alice's concurrent Turn 4 response"
        );

        // Verify Carol does NOT see Turn 1 and Turn 2 responses again (marked as sent)
        let has_alice_turn1 = turn4_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Alice" && msg.content.contains("turn 1"));
        let has_bob_turn1 = turn4_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 1"));

        assert!(
            !has_alice_turn1,
            "Carol should NOT see Alice's Turn 1 in Turn 4 (marked as sent)"
        );
        assert!(
            !has_bob_turn1,
            "Carol should NOT see Bob's Turn 1 in Turn 4 (marked as sent)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_mentioned_mode_with_recent_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Backend engineer".to_string(),
            communication_style: "Direct".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UI/UX specialist".to_string(),
            communication_style: "Creative".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let dave = Persona {
            name: "Dave".to_string(),
            role: "QA Engineer".to_string(),
            background: "Testing specialist".to_string(),
            communication_style: "Meticulous".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice turn 1".to_string(),
                "Alice turn 2".to_string(),
                "Alice turn 3".to_string(),
                "Alice turn 4".to_string(),
                "Alice turn 5".to_string(),
                "Alice turn 6".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob turn 1".to_string(),
                "Bob turn 3".to_string(),
                "Bob turn 5".to_string(),
            ],
        );
        let agent_dave = MockAgent::new(
            "Dave",
            vec!["Dave turn 2".to_string(), "Dave turn 4".to_string()],
        );

        let mut dialogue = Dialogue::mentioned();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);
        dialogue.add_participant(dave, agent_dave);

        // Build up 5 turns of history with various mention patterns
        let _turn1 = dialogue.run("@Alice and @Bob start").await.unwrap();
        let _turn2 = dialogue.run("@Dave check this").await.unwrap();
        let _turn3 = dialogue.run("@Alice and @Bob continue").await.unwrap();
        let _turn4 = dialogue.run("@Dave verify").await.unwrap();
        let _turn5 = dialogue.run("@Alice and @Bob finalize").await.unwrap();

        // Add Carol with Recent(2) strategy - should see only last 2 turns (Turn 4, 5)
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Security Consultant".to_string(),
            background: "Security expert".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol's recent analysis".to_string(),
                "Carol turn 7".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::recent_with_turns(2));

        // Turn 6: Mention Carol (first time she participates)
        let turn6 = dialogue.run("@Carol security review").await.unwrap();

        assert!(turn6.iter().any(|t| t.speaker.name() == "Carol"));
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received ONLY recent 2 turns (Turn 4 and Turn 5)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Collect all agent messages from history
        let agent_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Alice"
                    || msg.speaker.name() == "Bob"
                    || msg.speaker.name() == "Dave"
            })
            .collect();

        // Carol should see messages from Turn 4 and Turn 5 only
        // Turn 4: Dave's response ("Dave turn 4")
        // Turn 5: Alice's and Bob's responses ("Alice turn 3" - Alice's 3rd call, "Bob turn 5" - Bob's 3rd call)
        let has_dave_turn4 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Dave" && msg.content.contains("turn 4"));
        let has_alice_turn3 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Alice" && msg.content.contains("turn 3"));
        let has_bob_turn5 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 5"));

        assert!(
            has_dave_turn4,
            "Recent(2) strategy: Carol should see Dave's Turn 4"
        );
        assert!(
            has_alice_turn3,
            "Recent(2) strategy: Carol should see Alice's response in Turn 5 (her 3rd call)"
        );
        assert!(
            has_bob_turn5,
            "Recent(2) strategy: Carol should see Bob's response in Turn 5 (his 3rd call)"
        );

        // Carol should NOT see Turn 1, 2, or 3 responses
        let has_alice_turn1 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Alice" && msg.content.contains("turn 1"));
        let has_bob_turn1 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 1"));
        let has_dave_turn2 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Dave" && msg.content.contains("turn 2"));
        let has_bob_turn3 = agent_messages
            .iter()
            .any(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 3"));

        assert!(
            !has_alice_turn1,
            "Recent(2) strategy: Carol should NOT see Alice's Turn 1 (too old)"
        );
        assert!(
            !has_bob_turn1,
            "Recent(2) strategy: Carol should NOT see Bob's Turn 1 (too old)"
        );
        assert!(
            !has_dave_turn2,
            "Recent(2) strategy: Carol should NOT see Dave's Turn 2 (too old)"
        );
        assert!(
            !has_bob_turn3,
            "Recent(2) strategy: Carol should NOT see Bob's Turn 3 (too old)"
        );

        // Turn 7: Mention both Carol and Alice
        let _turn7 = dialogue.run("@Carol and @Alice continue").await.unwrap();

        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn7_messages = carol_second_payload.to_messages();

        // In Turn 7, Carol should NOT see Alice's concurrent Turn 7 response
        // and should NOT see old historical messages (marked as sent)
        let alice_turn7: Vec<_> = turn7_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Alice")
            .collect();

        assert_eq!(
            alice_turn7.len(),
            0,
            "In Turn 7, Carol should NOT see Alice's concurrent response"
        );

        // Verify no historical messages are resent
        let has_historical = turn7_messages.iter().any(|msg| {
            msg.content.contains("turn 1")
                || msg.content.contains("turn 2")
                || msg.content.contains("turn 3")
                || msg.content.contains("turn 4")
                || msg.content.contains("turn 5")
        });

        assert!(
            !has_historical,
            "In Turn 7, Carol should NOT see any historical messages (marked as sent)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_sequential_mode_with_full_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants for sequential processing
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Critical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice analyzed: turn 1".to_string(),
                "Alice analyzed: turn 2".to_string(),
                "Alice analyzed: turn 3".to_string(),
                "Alice analyzed: turn 4".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob reviewed: turn 1".to_string(),
                "Bob reviewed: turn 2".to_string(),
                "Bob reviewed: turn 3".to_string(),
                "Bob reviewed: turn 4".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();
        let bob_clone = agent_bob.clone();

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Turn 1 & 2: Alice → Bob (sequential)
        let _turn1 = dialogue.run("Analyze this data").await.unwrap();
        let _turn2 = dialogue.run("Continue analysis").await.unwrap();

        // Verify Alice and Bob were called twice each
        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice should be called twice"
        );
        assert_eq!(bob_clone.get_call_count(), 2, "Bob should be called twice");

        // Add Carol mid-dialogue with Full strategy - should see ALL history
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Summarizer".to_string(),
            background: "Summary specialist".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol full summary".to_string(),
                "Carol summary 2".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        // Carol joins with Full strategy (joins at end of sequence)
        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::Full);

        // Turn 3: Alice → Bob → Carol (sequential)
        let turn3 = dialogue.run("Final analysis").await.unwrap();

        // In sequential mode, only the last agent's output is returned
        assert_eq!(
            turn3.len(),
            1,
            "Sequential mode returns only last agent's output"
        );
        assert_eq!(
            turn3[0].speaker.name(),
            "Carol",
            "Last agent should be Carol"
        );

        // Verify all agents were called
        assert_eq!(
            alice_clone.get_call_count(),
            3,
            "Alice should be called 3 times"
        );
        assert_eq!(
            bob_clone.get_call_count(),
            3,
            "Bob should be called 3 times"
        );
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received FULL history (Turn 1 and Turn 2 from Alice and Bob)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should see ALL Turn 1 and Turn 2 historical messages from Alice and Bob
        let turn1_turn2_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && (content.contains("turn 1") || content.contains("turn 2"))
            })
            .collect();

        assert_eq!(
            turn1_turn2_messages.len(),
            4,
            "Full strategy in sequential mode: Carol should see ALL 4 historical messages (Alice Turn 1, Bob Turn 1, Alice Turn 2, Bob Turn 2)"
        );

        // Carol SHOULD also see Bob's Turn 3 output (immediate predecessor in the chain)
        let bob_turn3_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 3"))
            .collect();

        assert_eq!(
            bob_turn3_messages.len(),
            1,
            "Carol should see Bob's Turn 3 output (her immediate input in sequential chain)"
        );

        // Turn 4: Verify Carol receives only differential updates
        let _turn4 = dialogue.run("Continue").await.unwrap();

        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // Carol should see Bob's Turn 4 output (current chain input)
        // but NOT Turn 1, 2, or 3 historical messages (marked as sent)
        let bob_historical: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Bob"
                    && (msg.content.contains("turn 1")
                        || msg.content.contains("turn 2")
                        || msg.content.contains("turn 3"))
            })
            .collect();

        assert_eq!(
            bob_historical.len(),
            0,
            "In Turn 4, Carol should NOT see Bob's Turn 1, 2, or 3 (marked as sent)"
        );

        // Carol should see Bob's Turn 4 output (new message in the chain)
        let bob_turn4: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 4"))
            .collect();

        assert_eq!(
            bob_turn4.len(),
            1,
            "In Turn 4, Carol should see Bob's Turn 4 output (new chain input)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_sequential_mode_with_recent_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants for sequential processing
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Critical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice analyzed: turn 1".to_string(),
                "Alice analyzed: turn 2".to_string(),
                "Alice analyzed: turn 3".to_string(),
                "Alice analyzed: turn 4".to_string(),
                "Alice analyzed: turn 5".to_string(),
                "Alice analyzed: turn 6".to_string(),
                "Alice analyzed: turn 7".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob reviewed: turn 1".to_string(),
                "Bob reviewed: turn 2".to_string(),
                "Bob reviewed: turn 3".to_string(),
                "Bob reviewed: turn 4".to_string(),
                "Bob reviewed: turn 5".to_string(),
                "Bob reviewed: turn 6".to_string(),
                "Bob reviewed: turn 7".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();
        let bob_clone = agent_bob.clone();

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Build up 5 turns of history: Alice → Bob
        for i in 1..=5 {
            let _ = dialogue.run(format!("Message {}", i)).await.unwrap();
        }

        // Verify Alice and Bob were called 5 times each
        assert_eq!(
            alice_clone.get_call_count(),
            5,
            "Alice should be called 5 times"
        );
        assert_eq!(
            bob_clone.get_call_count(),
            5,
            "Bob should be called 5 times"
        );

        // Add Carol mid-dialogue with Recent(2) strategy
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Summarizer".to_string(),
            background: "Summary specialist".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol recent summary".to_string(),
                "Carol summary 2".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        // Carol joins with Recent(2) strategy (joins at end of sequence)
        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::recent_with_turns(2));

        // Turn 6: Alice → Bob → Carol (sequential)
        let turn6 = dialogue.run("Recent analysis").await.unwrap();

        // In sequential mode, only the last agent's output is returned
        assert_eq!(
            turn6.len(),
            1,
            "Sequential mode returns only last agent's output"
        );
        assert_eq!(
            turn6[0].speaker.name(),
            "Carol",
            "Last agent should be Carol"
        );

        // Verify all agents were called
        assert_eq!(
            alice_clone.get_call_count(),
            6,
            "Alice should be called 6 times"
        );
        assert_eq!(
            bob_clone.get_call_count(),
            6,
            "Bob should be called 6 times"
        );
        assert_eq!(
            carol_clone.get_call_count(),
            1,
            "Carol should be called once"
        );

        // Verify Carol received ONLY recent 2 turns (Turn 4 and Turn 5)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should see recent Turn 4 and Turn 5 historical messages
        let turn4_turn5_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && (content.contains("turn 4") || content.contains("turn 5"))
            })
            .collect();

        assert_eq!(
            turn4_turn5_messages.len(),
            4,
            "Recent(2) strategy in sequential mode: Carol should see 4 recent messages (Alice Turn 4, Bob Turn 4, Alice Turn 5, Bob Turn 5)"
        );

        // Carol should NOT see Turn 1, 2, or 3
        let turn1_turn2_turn3: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                content.contains("turn 1")
                    || content.contains("turn 2")
                    || content.contains("turn 3")
            })
            .collect();

        assert_eq!(
            turn1_turn2_turn3.len(),
            0,
            "Recent(2) strategy: Carol should NOT see Turn 1, 2, or 3 (too old)"
        );

        // Carol SHOULD also see Bob's Turn 6 output (immediate predecessor in the chain)
        let bob_turn6_messages: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 6"))
            .collect();

        assert_eq!(
            bob_turn6_messages.len(),
            1,
            "Carol should see Bob's Turn 6 output (her immediate input in sequential chain)"
        );

        // Turn 7: Verify Carol receives only differential updates
        let _turn7 = dialogue.run("Continue").await.unwrap();

        assert_eq!(
            carol_clone.get_call_count(),
            2,
            "Carol should be called twice"
        );

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn7_messages = carol_second_payload.to_messages();

        // Carol should see Bob's Turn 7 output (current chain input)
        // but NOT any historical messages (marked as sent)
        let bob_historical: Vec<_> = turn7_messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Bob"
                    && (msg.content.contains("turn 1")
                        || msg.content.contains("turn 2")
                        || msg.content.contains("turn 3")
                        || msg.content.contains("turn 4")
                        || msg.content.contains("turn 5")
                        || msg.content.contains("turn 6"))
            })
            .collect();

        assert_eq!(
            bob_historical.len(),
            0,
            "In Turn 7, Carol should NOT see Bob's historical turns (marked as sent)"
        );

        // Carol should see Bob's Turn 7 output (new message in the chain)
        let bob_turn7: Vec<_> = turn7_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 7"))
            .collect();

        assert_eq!(
            bob_turn7.len(),
            1,
            "In Turn 7, Carol should see Bob's Turn 7 output (new chain input)"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_partial_session_sequential_with_full_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Critical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice analyzed: turn 1".to_string(),
                "Alice analyzed: turn 2".to_string(),
                "Alice analyzed: turn 3".to_string(),
                "Alice analyzed: turn 4".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob reviewed: turn 1".to_string(),
                "Bob reviewed: turn 2".to_string(),
                "Bob reviewed: turn 3".to_string(),
                "Bob reviewed: turn 4".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Turn 1 & 2: Use partial_session
        let mut session1 = dialogue.partial_session("Analyze this");
        while let Some(turn) = session1.next_turn().await {
            turn.unwrap();
        }

        let mut session2 = dialogue.partial_session("Continue analysis");
        while let Some(turn) = session2.next_turn().await {
            turn.unwrap();
        }

        assert_eq!(
            alice_clone.get_call_count(),
            2,
            "Alice called twice via partial_session"
        );

        // Add Carol with Full strategy - should see ALL history
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Summarizer".to_string(),
            background: "Summary specialist".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol full summary".to_string(),
                "Carol summary 2".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::Full);

        // Turn 3: Use partial_session with Carol
        let mut session3 = dialogue.partial_session("Final summary");
        let mut turn_count = 0;
        while let Some(turn) = session3.next_turn().await {
            turn.unwrap();
            turn_count += 1;
        }

        // Sequential mode executes all participants (Alice → Bob → Carol)
        assert_eq!(
            turn_count, 3,
            "Should have 3 turns in sequential partial_session"
        );
        assert_eq!(carol_clone.get_call_count(), 1, "Carol called once");

        // Verify Carol received FULL history (Turn 1 and Turn 2)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should see ALL Turn 1 and Turn 2 historical messages
        let historical_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && (content.contains("turn 1") || content.contains("turn 2"))
            })
            .collect();

        assert_eq!(
            historical_messages.len(),
            4,
            "Full strategy in partial_session sequential: Carol should see ALL 4 historical messages (Alice Turn 1, Bob Turn 1, Alice Turn 2, Bob Turn 2)"
        );

        // Carol SHOULD see Bob's Turn 3 output (immediate predecessor in chain)
        let bob_turn3: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 3"))
            .collect();

        assert_eq!(
            bob_turn3.len(),
            1,
            "Carol should see Bob's Turn 3 output via partial_session"
        );

        // Turn 4: Verify differential updates via partial_session
        let mut session4 = dialogue.partial_session("Continue");
        while let Some(turn) = session4.next_turn().await {
            turn.unwrap();
        }

        assert_eq!(carol_clone.get_call_count(), 2, "Carol called twice");

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn4_messages = carol_second_payload.to_messages();

        // Carol should NOT see Turn 1, 2, or 3 historical messages
        let historical_turn4: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Bob"
                    && (msg.content.contains("turn 1")
                        || msg.content.contains("turn 2")
                        || msg.content.contains("turn 3"))
            })
            .collect();

        assert_eq!(
            historical_turn4.len(),
            0,
            "In Turn 4 via partial_session, Carol should NOT see Turn 1, 2, or 3 (marked as sent)"
        );

        // Carol should see Bob's Turn 4 output (new chain input)
        let bob_turn4: Vec<_> = turn4_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 4"))
            .collect();

        assert_eq!(
            bob_turn4.len(),
            1,
            "In Turn 4 via partial_session, Carol should see Bob's Turn 4 output"
        );
    }

    #[tokio::test]
    async fn test_join_in_progress_partial_session_sequential_with_recent_strategy() {
        use crate::agent::dialogue::joining_strategy::JoiningStrategy;
        use crate::agent::persona::Persona;

        // Create initial participants
        let alice = Persona {
            name: "Alice".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let bob = Persona {
            name: "Bob".to_string(),
            role: "Reviewer".to_string(),
            background: "Code reviewer".to_string(),
            communication_style: "Critical".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_alice = MockAgent::new(
            "Alice",
            vec![
                "Alice analyzed: turn 1".to_string(),
                "Alice analyzed: turn 2".to_string(),
                "Alice analyzed: turn 3".to_string(),
                "Alice analyzed: turn 4".to_string(),
                "Alice analyzed: turn 5".to_string(),
                "Alice analyzed: turn 6".to_string(),
                "Alice analyzed: turn 7".to_string(),
            ],
        );
        let agent_bob = MockAgent::new(
            "Bob",
            vec![
                "Bob reviewed: turn 1".to_string(),
                "Bob reviewed: turn 2".to_string(),
                "Bob reviewed: turn 3".to_string(),
                "Bob reviewed: turn 4".to_string(),
                "Bob reviewed: turn 5".to_string(),
                "Bob reviewed: turn 6".to_string(),
                "Bob reviewed: turn 7".to_string(),
            ],
        );
        let alice_clone = agent_alice.clone();

        let mut dialogue = Dialogue::sequential();
        dialogue.add_participant(alice, agent_alice);
        dialogue.add_participant(bob, agent_bob);

        // Build up 5 turns of history using partial_session
        for i in 1..=5 {
            let mut session = dialogue.partial_session(format!("Message {}", i));
            while let Some(turn) = session.next_turn().await {
                turn.unwrap();
            }
        }

        assert_eq!(
            alice_clone.get_call_count(),
            5,
            "Alice called 5 times via partial_session"
        );

        // Add Carol with Recent(2) strategy - should see only last 2 turns
        let carol = Persona {
            name: "Carol".to_string(),
            role: "Summarizer".to_string(),
            background: "Summary specialist".to_string(),
            communication_style: "Concise".to_string(),
            visual_identity: None,
            capabilities: None,
        };

        let agent_carol = MockAgent::new(
            "Carol",
            vec![
                "Carol recent summary".to_string(),
                "Carol summary 2".to_string(),
            ],
        );
        let carol_clone = agent_carol.clone();

        dialogue.join_in_progress(carol, agent_carol, JoiningStrategy::recent_with_turns(2));

        // Turn 6: Use partial_session with Carol
        let mut session6 = dialogue.partial_session("Recent summary");
        let mut turn_count = 0;
        while let Some(turn) = session6.next_turn().await {
            turn.unwrap();
            turn_count += 1;
        }

        // Sequential mode executes all participants (Alice → Bob → Carol)
        assert_eq!(
            turn_count, 3,
            "Should have 3 turns in sequential partial_session"
        );
        assert_eq!(carol_clone.get_call_count(), 1, "Carol called once");

        // Verify Carol received ONLY recent 2 turns (Turn 4 and Turn 5)
        let carol_payloads = carol_clone.get_payloads();
        assert_eq!(carol_payloads.len(), 1, "Carol should receive 1 payload");

        let carol_first_payload = &carol_payloads[0];
        let messages = carol_first_payload.to_messages();

        // Carol should see recent Turn 4 and Turn 5 historical messages
        let turn4_turn5_messages: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                (msg.speaker.name() == "Alice" || msg.speaker.name() == "Bob")
                    && (content.contains("turn 4") || content.contains("turn 5"))
            })
            .collect();

        assert_eq!(
            turn4_turn5_messages.len(),
            4,
            "Recent(2) strategy in partial_session sequential: Carol should see 4 recent messages (Alice Turn 4, Bob Turn 4, Alice Turn 5, Bob Turn 5)"
        );

        // Carol should NOT see Turn 1, 2, or 3
        let turn1_turn2_turn3: Vec<_> = messages
            .iter()
            .filter(|msg| {
                let content = msg.content.as_str();
                content.contains("turn 1")
                    || content.contains("turn 2")
                    || content.contains("turn 3")
            })
            .collect();

        assert_eq!(
            turn1_turn2_turn3.len(),
            0,
            "Recent(2) strategy: Carol should NOT see Turn 1, 2, or 3 (too old)"
        );

        // Carol SHOULD see Bob's Turn 6 output (immediate predecessor in chain)
        let bob_turn6: Vec<_> = messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 6"))
            .collect();

        assert_eq!(
            bob_turn6.len(),
            1,
            "Carol should see Bob's Turn 6 output via partial_session"
        );

        // Turn 7: Verify differential updates via partial_session
        let mut session7 = dialogue.partial_session("Continue");
        while let Some(turn) = session7.next_turn().await {
            turn.unwrap();
        }

        assert_eq!(carol_clone.get_call_count(), 2, "Carol called twice");

        let carol_second_payload = &carol_clone.get_payloads()[1];
        let turn7_messages = carol_second_payload.to_messages();

        // Carol should NOT see any historical messages (marked as sent)
        let historical_turn7: Vec<_> = turn7_messages
            .iter()
            .filter(|msg| {
                msg.speaker.name() == "Bob"
                    && (msg.content.contains("turn 1")
                        || msg.content.contains("turn 2")
                        || msg.content.contains("turn 3")
                        || msg.content.contains("turn 4")
                        || msg.content.contains("turn 5")
                        || msg.content.contains("turn 6"))
            })
            .collect();

        assert_eq!(
            historical_turn7.len(),
            0,
            "In Turn 7 via partial_session, Carol should NOT see historical turns (marked as sent)"
        );

        // Carol should see Bob's Turn 7 output (new chain input)
        let bob_turn7: Vec<_> = turn7_messages
            .iter()
            .filter(|msg| msg.speaker.name() == "Bob" && msg.content.contains("turn 7"))
            .collect();

        assert_eq!(
            bob_turn7.len(),
            1,
            "In Turn 7 via partial_session, Carol should see Bob's Turn 7 output"
        );
    }
}
