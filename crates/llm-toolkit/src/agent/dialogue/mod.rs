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

pub mod message;
pub mod session;
pub mod state;
pub mod store;
pub mod turn_input;

use crate::ToPrompt;
use crate::agent::chat::Chat;
use crate::agent::persona::{Persona, PersonaTeam, PersonaTeamGenerationRequest};
use crate::agent::{Agent, AgentError, Payload};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::task::JoinSet;

// Re-export key types
pub use message::{
    DialogueMessage, MessageId, MessageMetadata, Speaker, format_messages_to_prompt,
};
pub use session::DialogueSession;
pub use store::MessageStore;
pub use turn_input::{ContextMessage, ParticipantInfo, TurnInput};

// Internal modules (not re-exported)
use state::{BroadcastState, SessionState};

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

/// Represents a single turn in the dialogue.
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
}

impl Dialogue {
    /// Creates a new dialogue with the specified execution model.
    ///
    /// This is private - use `broadcast()` or `sequential()` instead.
    fn new(execution_model: ExecutionModel) -> Self {
        Self {
            participants: Vec::new(),
            message_store: MessageStore::new(),
            execution_model,
        }
    }

    /// Creates a new dialogue with broadcast execution.
    ///
    /// In broadcast mode, all participants respond in parallel to the same input.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(agent1)
    ///     .add_participant(agent2);
    /// ```
    pub fn broadcast() -> Self {
        Self::new(ExecutionModel::Broadcast)
    }

    /// Creates a new dialogue with sequential execution.
    ///
    /// In sequential mode, the output of one participant becomes the input to the next.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::sequential()
    ///     .add_participant(persona1, summarizer)
    ///     .add_participant(persona2, translator)
    ///     .add_participant(persona3, formatter);
    /// ```
    pub fn sequential() -> Self {
        Self::new(ExecutionModel::Sequential)
    }

    /// Sets initial conversation history for session resumption.
    ///
    /// This method allows you to inject a saved conversation history into a new
    /// dialogue instance, enabling session restoration and continuation of
    /// previous discussions.
    ///
    /// Following the Orchestrator Step pattern, this creates a new dialogue
    /// instance with pre-populated history rather than mutating existing state.
    ///
    /// # Arguments
    ///
    /// * `history` - A vector of `DialogueTurn` representing the conversation history
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// // Session 1: Initial conversation
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    /// let turns = dialogue.run("Discuss project architecture").await?;
    /// dialogue.save_history("session_123.json")?;
    ///
    /// // --- Process restart or session end ---
    ///
    /// // Session 2: Resume conversation
    /// let saved_history = Dialogue::load_history("session_123.json")?;
    /// let mut dialogue = Dialogue::broadcast()
    ///     .with_history(saved_history)  // ← Inject saved history
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    ///
    /// // Continue from where we left off
    /// let more_turns = dialogue.run("Continue from last discussion").await?;
    /// ```
    pub fn with_history(mut self, history: Vec<DialogueTurn>) -> Self {
        // Convert DialogueTurn to DialogueMessage and populate MessageStore
        // Assume each DialogueTurn is from turn 1, in order
        let mut turn_counter = 1;

        for dialogue_turn in history {
            let message = DialogueMessage {
                id: MessageId::new(),
                turn: turn_counter,
                speaker: dialogue_turn.speaker.clone(),
                content: dialogue_turn.content,
                timestamp: message::current_unix_timestamp(),
                metadata: Default::default(),
            };

            self.message_store.push(message);

            // Increment turn when we see a System message
            if matches!(dialogue_turn.speaker, Speaker::System) {
                turn_counter += 1;
            }
        }

        self
    }

    /// Creates a Dialogue from a blueprint.
    ///
    /// If the blueprint contains pre-defined participants, they are used directly.
    /// Otherwise, an LLM generates a team of personas based on the blueprint's context.
    ///
    /// # Arguments
    ///
    /// * `blueprint` - The dialogue blueprint containing agenda, context, and optional participants
    /// * `generator_agent` - LLM agent for generating personas (used only if blueprint.participants is None)
    /// * `dialogue_agent` - LLM agent for the actual dialogue interactions
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::{Dialogue, DialogueBlueprint};
    /// use llm_toolkit::agent::impls::{ClaudeCodeAgent, ClaudeCodeJsonAgent};
    ///
    /// // Create blueprint with auto-generated team
    /// let blueprint = DialogueBlueprint {
    ///     agenda: "1on1 Feature Planning".to_string(),
    ///     context: "Product planning meeting for new 1on1 feature in HR SaaS".to_string(),
    ///     participants: None,  // Will be auto-generated
    ///     execution_strategy: Some(ExecutionModel::Broadcast),
    /// };
    ///
    /// let mut dialogue = Dialogue::from_blueprint(
    ///     blueprint,
    ///     ClaudeCodeJsonAgent::new(),  // For team generation
    ///     ClaudeCodeAgent::new(),       // For dialogue
    /// ).await?;
    /// ```
    pub async fn from_blueprint<G, D>(
        blueprint: DialogueBlueprint,
        generator_agent: G,
        dialogue_agent: D,
    ) -> Result<Self, AgentError>
    where
        G: Agent<Output = PersonaTeam>,
        D: Agent<Output = String> + Clone + 'static,
    {
        // Determine execution model from blueprint
        let execution_model = blueprint
            .execution_strategy
            .unwrap_or(ExecutionModel::Broadcast);

        let mut dialogue = Self {
            participants: Vec::new(),
            message_store: MessageStore::new(),
            execution_model,
        };

        // Use provided participants or generate them
        let personas = match blueprint.participants {
            Some(personas) => personas,
            None => {
                // Generate PersonaTeam using LLM
                let request = PersonaTeamGenerationRequest::new(blueprint.context);
                let prompt = request.to_prompt();
                let team = generator_agent.execute(prompt.into()).await?;
                team.personas
            }
        };

        // Build participants from personas
        for persona in personas {
            let chat_agent = Chat::new(dialogue_agent.clone())
                .with_persona(persona.clone())
                .with_history(true)
                .build();

            dialogue.participants.push(Participant {
                persona,
                agent: Arc::new(chat_agent),
            });
        }

        Ok(dialogue)
    }

    /// Creates a Dialogue from a pre-generated PersonaTeam.
    ///
    /// This is useful for loading and reusing persona teams across different tasks.
    ///
    /// # Arguments
    ///
    /// * `team` - The PersonaTeam to create participants from
    /// * `llm_agent` - The base LLM agent to use for all participants
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::persona::PersonaTeam;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// // Load team from JSON
    /// let team = PersonaTeam::load("teams/dev_team.json")?;
    ///
    /// // Create dialogue
    /// let mut dialogue = Dialogue::from_persona_team(
    ///     team,
    ///     ClaudeCodeAgent::new(),
    /// )?;
    ///
    /// let result = dialogue.run("Discuss API design").await?;
    /// ```
    pub fn from_persona_team<T>(team: PersonaTeam, llm_agent: T) -> Result<Self, AgentError>
    where
        T: Agent<Output = String> + Clone + 'static,
    {
        // Determine execution model from team hint
        let execution_model = team.execution_strategy.unwrap_or(ExecutionModel::Broadcast);

        let mut dialogue = Self {
            participants: Vec::new(),
            message_store: MessageStore::new(),
            execution_model,
        };

        // Build participants from personas
        for persona in team.personas {
            let chat_agent = Chat::new(llm_agent.clone())
                .with_persona(persona.clone())
                .with_history(true)
                .build();

            dialogue.participants.push(Participant {
                persona,
                agent: Arc::new(chat_agent),
            });
        }

        Ok(dialogue)
    }

    /// Adds a participant to the dialogue dynamically.
    ///
    /// Unlike StrategyMap (which has a fixed execution plan), Dialogue
    /// supports adding participants at runtime for flexible conversation scenarios.
    ///
    /// # Arguments
    ///
    /// * `persona` - The persona to add
    /// * `llm_agent` - The LLM agent to use for this participant
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::persona::Persona;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// let mut dialogue = Dialogue::broadcast();
    /// // ... initial setup ...
    ///
    /// // Mid-discussion: bring in a domain expert
    /// let expert = Persona {
    ///     name: "Dr. Smith".to_string(),
    ///     role: "Security Consultant".to_string(),
    ///     background: "20 years in enterprise security...".to_string(),
    ///     communication_style: "Detail-oriented and cautious...".to_string(),
    /// };
    ///
    /// dialogue.add_participant(expert, ClaudeCodeAgent::new());
    /// ```
    pub fn add_participant<T>(&mut self, persona: Persona, llm_agent: T) -> &mut Self
    where
        T: Agent<Output = String> + 'static,
    {
        let chat_agent = Chat::new(llm_agent)
            .with_persona(persona.clone())
            .with_history(true)
            .build();

        self.participants.push(Participant {
            persona,
            agent: Arc::new(chat_agent),
        });

        self
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
        // Convert to Payload first
        let payload: Payload = initial_prompt.into();
        let prompt_text = payload.to_text();
        let current_turn = self.message_store.current_turn() + 1;

        // Store system message in MessageStore
        let system_message =
            DialogueMessage::new(current_turn, Speaker::System, prompt_text.clone());
        self.message_store.push(system_message);

        let model = self.execution_model;
        let state = match model {
            ExecutionModel::Broadcast => {
                // Build participant list
                let participants_info: Vec<ParticipantInfo> = self
                    .participants
                    .iter()
                    .map(|p| {
                        ParticipantInfo::new(
                            p.persona.name.clone(),
                            p.persona.role.clone(),
                            p.persona.background.clone(),
                        )
                    })
                    .collect();

                // Get context from previous turn (for multi-turn dialogues)
                let context = if current_turn > 1 {
                    self.message_store
                        .messages_for_turn(current_turn - 1)
                        .into_iter()
                        .filter_map(|msg| {
                            if let Speaker::Agent { name, role } = &msg.speaker {
                                Some(ContextMessage::with_metadata(
                                    name.clone(),
                                    role.clone(),
                                    msg.content.clone(),
                                    msg.turn,
                                    msg.timestamp,
                                ))
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                let mut pending = JoinSet::new();

                for (idx, participant) in self.participants.iter().enumerate() {
                    let agent = Arc::clone(&participant.agent);
                    let name = participant.name().to_string();

                    // Create TurnInput with participant-specific information
                    let turn_input = TurnInput::with_dialogue_context(
                        prompt_text.clone(),
                        context.clone(),
                        participants_info.clone(),
                        name.clone(),
                    );

                    // Create payload with Messages (for structured dialogue history)
                    let messages = turn_input.to_messages();
                    let mut input_payload = Payload::from_messages(messages);

                    // Add Participants metadata
                    input_payload = input_payload.with_participants(participants_info.clone());

                    // Copy attachments from original payload if any
                    let final_payload = if payload.has_attachments() {
                        let mut p = input_payload;
                        for attachment in payload.attachments() {
                            p = p.with_attachment(attachment.clone());
                        }
                        p
                    } else {
                        input_payload
                    };

                    pending.spawn(async move {
                        let result = agent.execute(final_payload).await;
                        (idx, name, result)
                    });
                }

                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    broadcast_order,
                    self.participants.len(),
                    current_turn,
                ))
            }
            ExecutionModel::Sequential => SessionState::Sequential {
                next_index: 0,
                current_input: prompt_text,
                current_turn,
            },
        };

        DialogueSession {
            dialogue: self,
            state,
            model,
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
        // Use new implementation for both modes
        match self.execution_model {
            ExecutionModel::Broadcast => self.run_broadcast_new(initial_prompt).await,
            ExecutionModel::Sequential => self.run_sequential_new(initial_prompt).await,
        }
    }

    /// New broadcast implementation using MessageStore and TurnInput.
    async fn run_broadcast_new(
        &mut self,
        initial_prompt: impl Into<Payload>,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        let payload: Payload = initial_prompt.into();
        let current_turn = self.message_store.current_turn() + 1;

        // 1. Extract and store messages from payload
        let (input_messages, prompt_text) =
            self.extract_messages_from_payload(&payload, current_turn);

        for msg in input_messages {
            self.message_store.push(msg);
        }

        // 2. Build participant list
        let participants_info: Vec<ParticipantInfo> = self
            .participants
            .iter()
            .map(|p| {
                ParticipantInfo::new(
                    p.persona.name.clone(),
                    p.persona.role.clone(),
                    p.persona.background.clone(),
                )
            })
            .collect();

        // 3. Get context from previous turn (other agents' responses)
        let context = if current_turn > 1 {
            self.message_store
                .messages_for_turn(current_turn - 1)
                .into_iter()
                .filter_map(|msg| {
                    if let Speaker::Agent { name, role } = &msg.speaker {
                        Some(ContextMessage::with_metadata(
                            name.clone(),
                            role.clone(),
                            msg.content.clone(),
                            msg.turn,
                            msg.timestamp,
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // 4. Broadcast to all agents (each with their own TurnInput)
        let mut pending = JoinSet::new();

        for (idx, participant) in self.participants.iter().enumerate() {
            let agent = Arc::clone(&participant.agent);
            let name = participant.name().to_string();

            // Create TurnInput with participant-specific information
            let turn_input = TurnInput::with_dialogue_context(
                prompt_text.clone(),
                context.clone(),
                participants_info.clone(),
                name.clone(),
            );

            // Create payload with Messages (for structured dialogue history)
            let messages = turn_input.to_messages();
            let mut input_payload = Payload::from_messages(messages);

            // Add Participants metadata
            input_payload = input_payload.with_participants(participants_info.clone());

            // Copy attachments from original payload if any
            let final_payload = if payload.has_attachments() {
                let mut p = input_payload;
                for attachment in payload.attachments() {
                    p = p.with_attachment(attachment.clone());
                }
                p
            } else {
                input_payload
            };

            pending.spawn(async move {
                let result = agent.execute(final_payload).await;
                (idx, name, result)
            });
        }

        // 5. Collect responses and create message entities
        let mut dialogue_turns = Vec::new();

        while let Some(Ok((idx, name, result))) = pending.join_next().await {
            match result {
                Ok(content) => {
                    // Store response message
                    let response_message = DialogueMessage::new(
                        current_turn,
                        Speaker::agent(name.clone(), self.participants[idx].persona.role.clone()),
                        content.clone(),
                    );
                    self.message_store.push(response_message);

                    // Create DialogueTurn for backward compatibility
                    dialogue_turns.push(DialogueTurn {
                        speaker: Speaker::agent(name, self.participants[idx].persona.role.clone()),
                        content,
                    });
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
    async fn run_sequential_new(
        &mut self,
        initial_prompt: impl Into<Payload>,
    ) -> Result<Vec<DialogueTurn>, AgentError> {
        let payload: Payload = initial_prompt.into();
        let current_turn = self.message_store.current_turn() + 1;

        // 1. Extract and store messages from payload
        let (input_messages, prompt_text) =
            self.extract_messages_from_payload(&payload, current_turn);

        for msg in input_messages {
            self.message_store.push(msg);
        }

        // 2. Build participant list
        let participants_info: Vec<ParticipantInfo> = self
            .participants
            .iter()
            .map(|p| {
                ParticipantInfo::new(
                    p.persona.name.clone(),
                    p.persona.role.clone(),
                    p.persona.background.clone(),
                )
            })
            .collect();

        // 3. Chain through participants sequentially
        let mut current_input = prompt_text;
        let mut final_turn = None;

        for (idx, participant) in self.participants.iter().enumerate() {
            let agent = &participant.agent;
            let name = participant.name().to_string();

            // Create payload with Message (for history tracking) + Text (for to_text())
            let mut input_payload =
                Payload::from_messages(vec![(Speaker::System, current_input.clone())])
                    .with_text(current_input.clone());

            // Add Participants metadata
            input_payload = input_payload.with_participants(participants_info.clone());

            // For first agent, copy attachments from original payload if any
            let final_payload = if idx == 0 && payload.has_attachments() {
                let mut p = input_payload;
                for attachment in payload.attachments() {
                    p = p.with_attachment(attachment.clone());
                }
                p
            } else {
                input_payload
            };

            // Execute agent
            let response = agent.execute(final_payload).await?;

            // Store response message
            let response_message = DialogueMessage::new(
                current_turn,
                Speaker::agent(name.clone(), participant.persona.role.clone()),
                response.clone(),
            );
            self.message_store.push(response_message);

            // Chain output to next input
            current_input = response.clone();

            // Keep track of final turn
            final_turn = Some(DialogueTurn {
                speaker: Speaker::agent(name, participant.persona.role.clone()),
                content: response,
            });
        }

        // 3. Return only the final turn
        Ok(final_turn.into_iter().collect())
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
                PayloadContent::Message { speaker, content } => {
                    // Store as individual message with explicit speaker
                    messages.push(DialogueMessage::new(turn, speaker.clone(), content.clone()));
                    text_parts.push(content.as_str());
                }
                PayloadContent::Text(text) => {
                    // Text without explicit speaker is treated as User input
                    // TODO: Allow configuring default speaker (User vs System)
                    messages.push(DialogueMessage::new(
                        turn,
                        Speaker::System, // For backward compatibility, treat as System
                        text.clone(),
                    ));
                    text_parts.push(text.as_str());
                }
                PayloadContent::Attachment(_) | PayloadContent::Participants(_) => {
                    // Attachments and Participants metadata don't create messages, just pass through
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
        };

        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Tester".to_string(),
            background: "Test agent 2".to_string(),
            communication_style: "Direct".to_string(),
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
        };

        let persona2 = Persona {
            name: "Translator".to_string(),
            role: "Translator".to_string(),
            background: "Translates content".to_string(),
            communication_style: "Formal".to_string(),
        };

        let persona3 = Persona {
            name: "Finalizer".to_string(),
            role: "Finalizer".to_string(),
            background: "Finalizes output".to_string(),
            communication_style: "Professional".to_string(),
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
        };

        let persona2 = Persona {
            name: "Step2".to_string(),
            role: "Stage".to_string(),
            background: "second".to_string(),
            communication_style: "Direct".to_string(),
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
        };

        let slow = Persona {
            name: "Slow".to_string(),
            role: "Slow responder".to_string(),
            background: "Takes time".to_string(),
            communication_style: "Measured".to_string(),
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
        };

        let fast = Persona {
            name: "Fast".to_string(),
            role: "Quick responder".to_string(),
            background: "Snappy insights".to_string(),
            communication_style: "Direct".to_string(),
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
        });
        team.add_persona(Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
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
        });
        team.add_persona(Persona {
            name: "Second".to_string(),
            role: "Synthesizer".to_string(),
            background: "Content creator".to_string(),
            communication_style: "Creative".to_string(),
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
        };
        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
        };
        let persona3 = Persona {
            name: "Agent3".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
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
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
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
        };
        let core2 = Persona {
            name: "CoreMember2".to_string(),
            role: "Core Member".to_string(),
            background: "Core team member".to_string(),
            communication_style: "Direct".to_string(),
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
        };

        let persona2 = Persona {
            name: "Summarizer".to_string(),
            role: "Summarizer".to_string(),
            background: "Summarizes results".to_string(),
            communication_style: "Concise".to_string(),
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
        };

        let persona2 = Persona {
            name: "Second".to_string(),
            role: "Second Responder".to_string(),
            background: "Detailed analysis".to_string(),
            communication_style: "Thorough".to_string(),
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
                Ok(format!("{} received: {}", self.name, payload.to_text()))
            }
        }

        let mut dialogue = Dialogue::broadcast();

        let persona_a = Persona {
            name: "AgentA".to_string(),
            role: "Tester".to_string(),
            background: "Test agent A".to_string(),
            communication_style: "Direct".to_string(),
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Tester".to_string(),
            background: "Test agent B".to_string(),
            communication_style: "Direct".to_string(),
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
        };

        let persona_b = Persona {
            name: "AgentB".to_string(),
            role: "Tester".to_string(),
            background: "Test agent B".to_string(),
            communication_style: "Direct".to_string(),
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
            },
            MockAgent::new("Agent1", vec!["Response from Agent1".to_string()]),
        );
        dialogue.add_participant(
            Persona {
                name: "Agent2".to_string(),
                role: "Tester".to_string(),
                background: "Test agent 2".to_string(),
                communication_style: "Direct".to_string(),
            },
            MockAgent::new("Agent2", vec!["Response from Agent2".to_string()]),
        );

        // Create multi-message payload with System + User messages
        let payload = Payload::from_messages(vec![
            (
                Speaker::System,
                "System: Initializing conversation".to_string(),
            ),
            (
                Speaker::user("Alice", "Product Manager"),
                "User: What should we build?".to_string(),
            ),
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
            },
            MockAgent::new("Agent1", vec!["Analysis result".to_string()]),
        );
        dialogue.add_participant(
            Persona {
                name: "Agent2".to_string(),
                role: "Reviewer".to_string(),
                background: "Second agent".to_string(),
                communication_style: "Critical".to_string(),
            },
            MockAgent::new("Agent2", vec!["Review complete".to_string()]),
        );

        // Create payload with multiple speakers
        let payload = Payload::from_messages(vec![
            (
                Speaker::System,
                "Context: Project initialization".to_string(),
            ),
            (
                Speaker::user("Bob", "Engineer"),
                "Request: Analyze architecture".to_string(),
            ),
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

        let mut session2 = dialogue_partial.partial_session("Turn 2");
        while (session2.next_turn().await).is_some() {}

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
}
