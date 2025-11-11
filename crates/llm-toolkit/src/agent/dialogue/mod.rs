//! Dialogue component for multi-agent conversational interactions.
//!
//! This module provides abstractions for managing turn-based dialogues between
//! multiple agents, with configurable turn-taking strategies.
//!
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
pub mod message;
pub mod session;
pub mod state;
pub mod store;
pub mod turn_input;

use crate::ToPrompt;
use crate::agent::chat::Chat;
use crate::agent::persona::Persona;
use crate::agent::{Agent, AgentError, Payload, PayloadMessage};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{debug, trace};

// Re-export key types
pub use context::{DialogueContext, TalkStyle};
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
/// ```rust,ignore
/// let text = "@Alice @Bob what do you think? @Alice?";
/// let participants = vec!["Alice", "Bob", "Charlie"];
/// let mentions = extract_mentions(text, &participants);
/// assert_eq!(mentions, vec!["Alice", "Bob"]);
/// ```
fn extract_mentions<'a>(text: &str, participant_names: &'a [&'a str]) -> Vec<&'a str> {
    use std::collections::HashSet;

    // Compile regex each time (simpler approach without external dependencies)
    // Performance impact is minimal for typical dialogue use cases
    let mention_regex = Regex::new(r"@(\w+)").expect("Invalid regex pattern");

    let mut mentioned = HashSet::new();

    // Extract all @mentions from text
    for cap in mention_regex.captures_iter(text) {
        if let Some(mention) = cap.get(1) {
            let mention_str = mention.as_str();
            // Exact match against participant names
            if let Some(&matched_name) = participant_names.iter().find(|&&name| name == mention_str)
            {
                mentioned.insert(matched_name);
            }
        }
    }

    mentioned.into_iter().collect()
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

/// Represents the execution model for dialogue strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionModel {
    /// All participants respond in parallel to the same input.
    Broadcast,
    /// Participants execute sequentially, with output chained as input.
    Sequential,
    /// Only @mentioned participants respond (falls back to Broadcast if no mentions).
    ///
    /// Supports multiple mentions like "@Alice @Bob what do you think?"
    /// If no mentions are found in the message, behaves like Broadcast mode.
    /// Future: Can be extended to `Mentioned { mode: MentionMode }` for strict mode.
    Mentioned,
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

/// Internal representation of a dialogue participant.
///
/// Wraps a persona and its associated agent implementation.
pub(super) struct Participant {
    pub(super) persona: Persona,
    pub(super) agent: Arc<dyn Agent<Output = String>>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastOrder {
    /// Yields turns as soon as each participant finishes (default).
    Completion,
    /// Buffers responses and yields them in the original participant order.
    ParticipantOrder,
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
pub struct Dialogue {
    pub(super) participants: Vec<Participant>,

    /// Message store for all dialogue messages
    pub(super) message_store: MessageStore,

    pub(super) execution_model: ExecutionModel,

    /// Optional dialogue context that shapes conversation tone and behavior
    pub(super) context: Option<DialogueContext>,

    /// Strategy for determining when agents should react to messages
    pub(super) reaction_strategy: ReactionStrategy,
}

impl Dialogue {
    // Constructor is in constructor.rs

    /// Creates a single Participant from a Persona and LLM agent.
    ///
    /// This is a private helper that encapsulates the standard participant
    /// creation pattern used throughout the module.
    fn create_participant<T>(persona: Persona, llm_agent: T) -> Participant
    where
        T: Agent<Output = String> + 'static,
    {
        let chat_agent = Chat::new(llm_agent)
            .with_persona(persona.clone())
            .with_history(true)
            .build();

        Participant {
            persona,
            agent: Arc::new(chat_agent),
        }
    }

    /// Creates multiple Participants from a list of Personas.
    ///
    /// This helper converts a Vec<Persona> into Vec<Participant> by
    /// creating a Chat agent for each persona with the provided base agent.
    fn create_participants<T>(personas: Vec<Persona>, llm_agent: T) -> Vec<Participant>
    where
        T: Agent<Output = String> + Clone + 'static,
    {
        personas
            .into_iter()
            .map(|persona| Self::create_participant(persona, llm_agent.clone()))
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

    pub fn add_participant<T>(&mut self, persona: Persona, llm_agent: T) -> &mut Self
    where
        T: Agent<Output = String> + 'static,
    {
        self.participants
            .push(Self::create_participant(persona, llm_agent));

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
        let current_turn = self.message_store.current_turn() + 1;
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
        match self.execution_model {
            ExecutionModel::Broadcast => self.run_broadcast(current_turn).await,
            ExecutionModel::Sequential => self.run_sequential(current_turn).await,
            ExecutionModel::Mentioned => self.run_mentioned(current_turn).await,
        }
    }

    /// New broadcast implementation using MessageStore and TurnInput.
    async fn run_broadcast(
        &mut self,
        current_turn: usize,
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
        let mut pending = self.spawn_broadcast_tasks();

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

        // Execute participants sequentially
        let mut final_turn = None;

        for (idx, participant) in self.participants.iter().enumerate() {
            let agent = &participant.agent;
            let agent_name = participant.name().to_string();

            // Determine input messages based on position in sequence
            let (current_messages, messages_with_metadata, message_ids_to_mark) = if idx == 0 {
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
                        agent_idx = idx,
                        agent_name = %agent_name,
                        prev_turn_message_count = prev_turn_messages.len(),
                        "Sequential mode: First agent receiving {} previous turn agent messages",
                        prev_turn_messages.len()
                    );

                    messages.extend(prev_turn_messages.clone());
                    metadata_messages.extend(prev_turn_messages);
                }

                // Add unsent incoming messages
                messages.extend(unsent_messages_incoming.clone());
                metadata_messages.extend(unsent_messages_incoming.clone());

                // Don't mark incoming messages as sent yet - subsequent agents need to see them
                (messages, metadata_messages, vec![])
            } else {
                // Subsequent agents: get ALL previous agents' messages from current turn
                // (not just unsent, as they may have been marked sent by earlier agents in the chain)
                let prev_agent_messages: Vec<PayloadMessage> = self
                    .message_store
                    .messages_for_turn(current_turn)
                    .into_iter()
                    .filter(|msg| matches!(msg.speaker, Speaker::Agent { .. }))
                    .map(PayloadMessage::from)
                    .collect();

                // Combine: previous agents' outputs + unsent incoming messages
                let mut messages = prev_agent_messages.clone();
                messages.extend(unsent_messages_incoming.clone());

                let mut metadata_messages = prev_agent_messages.clone();
                metadata_messages.extend(unsent_messages_incoming.clone());

                trace!(
                    target = "llm_toolkit::dialogue",
                    turn = current_turn,
                    agent_idx = idx,
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

            // Prepend context if exists
            if let Some(ref context) = self.context {
                input_payload = input_payload.prepend_system(context.to_prompt());
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
                    agent_idx = idx,
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
        let mut pending = self.spawn_mentioned_tasks(current_turn);

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
        self.partial_session_with_order(initial_prompt, BroadcastOrder::Completion)
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
        let payload: Payload = initial_prompt.into();
        let current_turn = self.message_store.current_turn() + 1;

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
            let model = self.execution_model;
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

        let model = self.execution_model;
        let state = match model {
            ExecutionModel::Broadcast => {
                // Spawn broadcast tasks using helper method
                let pending = self.spawn_broadcast_tasks();

                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    broadcast_order,
                    self.participants.len(),
                    current_turn,
                ))
            }
            ExecutionModel::Mentioned => {
                // For Mentioned mode, spawn tasks for mentioned participants only
                let pending = self.spawn_mentioned_tasks(current_turn);

                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    broadcast_order,
                    self.participants.len(),
                    current_turn,
                ))
            }
            ExecutionModel::Sequential => {
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

                SessionState::Sequential {
                    next_index: 0,
                    current_turn,
                    payload,
                    prev_agent_outputs,
                    current_turn_outputs: Vec::new(),
                    participants_info,
                }
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
    ) -> JoinSet<(usize, String, Result<String, AgentError>)> {
        // Build participant list
        let participants_info = self.get_participants_info();

        // Get unsent agent-generated messages from MessageStore
        let unsent_messages_from_agent: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::AgentGenerated)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect message IDs to mark as sent after spawning tasks
        let mut unsent_message_ids: Vec<_> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::AgentGenerated)
            .iter()
            .map(|msg| msg.id)
            .collect();

        let unsent_messages_incoming: Vec<PayloadMessage> = self
            .message_store
            .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
            .into_iter()
            .map(PayloadMessage::from)
            .collect();

        // Collect message IDs to mark as sent after spawning tasks
        unsent_message_ids.extend(
            self.message_store
                .unsent_messages_with_origin(MessageOrigin::IncomingPayload)
                .iter()
                .map(|msg| msg.id),
        );

        let mut pending = JoinSet::new();

        for (idx, participant) in self.participants.iter().enumerate() {
            let agent = Arc::clone(&participant.agent);
            let name = participant.name().to_string();

            // Combine: [unsent messages (excluding self)] + [new intent]
            let mut current_messages = unsent_messages_from_agent
                .iter()
                .filter(|msg| msg.speaker.name() != name)
                .cloned()
                .collect::<Vec<_>>();

            current_messages.extend(unsent_messages_incoming.clone());
            let messages_with_metadata = current_messages.clone();

            // current_messages now contains everything needed for this agent's turn:
            // 1. Previous turn agent outputs (excluding self)
            // 2. Unsent messages (excluding self)
            // 3. Current Payload content (Messages + Text as System message)
            let turn_input = TurnInput::with_messages_and_context(
                current_messages,
                vec![], // context is now integrated into current_messages
                participants_info.clone(),
                name.clone(),
            );

            // Create payload with Messages (for structured dialogue history)
            let messages = turn_input.to_messages();
            let mut input_payload = Payload::from_messages(messages);
            if let Some(ref context) = self.context {
                input_payload = input_payload.prepend_system(context.to_prompt());
            }
            input_payload =
                Self::apply_metadata_attachments(input_payload, &messages_with_metadata);

            // Add Participants metadata
            input_payload = input_payload.with_participants(participants_info.clone());

            pending.spawn(async move {
                let result = agent.execute(input_payload).await;
                (idx, name, result)
            });
        }

        // Mark unsent messages as sent to agents
        self.message_store.mark_all_as_sent(&unsent_message_ids);

        if !unsent_message_ids.is_empty() {
            trace!(
                target = "llm_toolkit::dialogue",
                marked_sent_count = unsent_message_ids.len(),
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

        // Get all participant names
        let participant_names = self.participant_names();

        // Extract mentions from the text
        let mentioned_names = extract_mentions(&mentions_text, &participant_names);

        trace!(
            target = "llm_toolkit::dialogue",
            turn = current_turn,
            mentions_text_preview = &mentions_text[..mentions_text.len().min(100)],
            incoming_message_count = unsent_messages_incoming.len(),
            agent_message_count = unsent_messages_from_agent.len(),
            all_participants = ?participant_names,
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
            participant_names
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

        let mut pending = JoinSet::new();

        // Only spawn tasks for mentioned participants (or all if no mentions)
        for (idx, participant) in self.participants.iter().enumerate() {
            let name = participant.name();

            // Skip if this participant was not mentioned
            if !target_participants.contains(&name) {
                trace!(
                    target = "llm_toolkit::dialogue",
                    participant = name,
                    reason = "not_mentioned",
                    "Skipping participant"
                );
                continue;
            }

            let agent = Arc::clone(&participant.agent);
            let name = name.to_string();

            // Combine: [unsent agent messages (excluding self)] + [incoming messages]
            let mut current_messages = unsent_messages_from_agent
                .iter()
                .filter(|msg| msg.speaker.name() != name)
                .cloned()
                .collect::<Vec<_>>();

            current_messages.extend(unsent_messages_incoming.clone());
            let messages_with_metadata = current_messages.clone();

            // current_messages now contains everything needed for this agent's turn:
            // 1. Unsent agent messages (excluding self)
            // 2. Incoming messages from MessageStore
            let turn_input = TurnInput::with_messages_and_context(
                current_messages.clone(),
                vec![], // context is now integrated into current_messages
                participants_info.clone(),
                name.clone(),
            );

            // Create payload with Messages (for structured dialogue history)
            let messages = turn_input.to_messages();
            let mut input_payload = Payload::from_messages(messages);

            // Prepend context if exists
            if let Some(ref context) = self.context {
                input_payload = input_payload.prepend_system(context.to_prompt());
            }

            // Apply metadata attachments
            input_payload =
                Self::apply_metadata_attachments(input_payload, &messages_with_metadata);

            // Add Participants metadata
            input_payload = input_payload.with_participants(participants_info.clone());

            trace!(
                target = "llm_toolkit::dialogue",
                turn = current_turn,
                participant = %name,
                message_count = current_messages.len(),
                "Spawning task for mentioned participant"
            );

            pending.spawn(async move {
                let result = agent.execute(input_payload).await;
                (idx, name, result)
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
                | PayloadContent::Document(_) => {
                    // Attachments, Participants metadata, and Documents don't create messages, just pass through
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
    }

    impl MockAgent {
        fn new(name: impl Into<String>, responses: Vec<String>) -> Self {
            Self {
                name: name.into(),
                responses,
                call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
            }
        }
    }

    #[async_trait]
    impl Agent for MockAgent {
        type Output = String;

        fn expertise(&self) -> &str {
            "Mock agent for testing"
        }

        fn name(&self) -> String {
            self.name.clone()
        }

        async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
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
        assert_eq!(session.execution_model(), ExecutionModel::Sequential);

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

        fn expertise(&self) -> &str {
            "Delayed agent"
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
        assert_eq!(session.execution_model(), ExecutionModel::Broadcast);

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
        team.execution_strategy = Some(ExecutionModel::Broadcast);

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
        team.execution_strategy = Some(ExecutionModel::Sequential);

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
            execution_strategy: Some(ExecutionModel::Broadcast),
        };

        // Mock generator agent - won't be used since participants are provided
        #[derive(Clone)]
        struct MockGeneratorAgent;

        #[async_trait]
        impl Agent for MockGeneratorAgent {
            type Output = PersonaTeam;

            fn expertise(&self) -> &str {
                "Generator"
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

            fn expertise(&self) -> &str {
                "Echo agent"
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

            fn expertise(&self) -> &str {
                "Verbose echo agent"
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
            agent: Arc::new(chat_a),
        });

        dialogue.participants.push(Participant {
            persona: persona_b,
            agent: Arc::new(chat_b),
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

            fn expertise(&self) -> &str {
                "History echo agent"
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
        assert_eq!(dialogue.message_store.current_turn(), 2);
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

            fn expertise(&self) -> &str {
                "Echo agent"
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
        assert_eq!(dialogue.message_store.current_turn(), 2);
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

            fn expertise(&self) -> &str {
                "Simple agent"
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
            dialogue_run.message_store.current_turn(),
            dialogue_partial.message_store.current_turn()
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

            fn expertise(&self) -> &str {
                "Tracking agent"
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

            fn expertise(&self) -> &str {
                "Tracking agent"
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

            fn expertise(&self) -> &str {
                "Tracking agent"
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

            fn expertise(&self) -> &str {
                "Tracking agent"
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

            fn expertise(&self) -> &str {
                "Context tracking agent"
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

            fn expertise(&self) -> &str {
                "Tracking agent"
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

        fn expertise(&self) -> &str {
            "Recording agent for testing"
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
}
