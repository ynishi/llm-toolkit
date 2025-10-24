//! Dialogue component for multi-agent conversational interactions.
//!
//! This module provides abstractions for managing turn-based dialogues between
//! multiple agents, with configurable turn-taking strategies.

use super::chat::Chat;
use super::{Agent, AgentError, Payload};
use crate::ToPrompt;
use crate::agent::persona::{Persona, PersonaTeam, PersonaTeamGenerationRequest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{error, info};

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
#[derive(Debug, Clone)]
pub struct DialogueTurn {
    pub participant_name: String,
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
struct Participant {
    persona: Persona,
    agent: Arc<dyn Agent<Output = String>>,
}

impl Participant {
    /// Returns the name of the participant from their persona.
    fn name(&self) -> &str {
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

struct BroadcastState {
    pending: JoinSet<(usize, String, Result<String, AgentError>)>,
    order: BroadcastOrder,
    buffered: Vec<Option<Result<String, AgentError>>>,
    next_emit: usize,
}

impl BroadcastState {
    fn new(
        pending: JoinSet<(usize, String, Result<String, AgentError>)>,
        order: BroadcastOrder,
        participant_count: usize,
    ) -> Self {
        let buffered = match order {
            BroadcastOrder::Completion => Vec::new(),
            BroadcastOrder::ParticipantOrder => std::iter::repeat_with(|| None)
                .take(participant_count)
                .collect::<Vec<Option<Result<String, AgentError>>>>(),
        };

        Self {
            pending,
            order,
            buffered,
            next_emit: 0,
        }
    }

    fn record_result(&mut self, idx: usize, result: Result<String, AgentError>) {
        if matches!(self.order, BroadcastOrder::ParticipantOrder) && idx < self.buffered.len() {
            self.buffered[idx] = Some(result);
        }
    }

    fn try_emit(&mut self, dialogue: &mut Dialogue) -> Option<Result<DialogueTurn, AgentError>> {
        if self.order != BroadcastOrder::ParticipantOrder {
            return None;
        }

        let participant_total = dialogue.participants.len();

        if self.next_emit >= participant_total {
            return None;
        }

        let idx = self.next_emit;
        let slot_ready = self
            .buffered
            .get(idx)
            .and_then(|slot| slot.as_ref())
            .is_some();

        if !slot_ready {
            return None;
        }

        let result = self.buffered[idx].take().expect("checked is_some");
        self.next_emit += 1;

        match result {
            Ok(content) => {
                let participant_name = dialogue.participants[idx].name().to_string();
                let turn = DialogueTurn {
                    participant_name: participant_name.clone(),
                    content: content.clone(),
                };
                dialogue.history.push(turn.clone());
                info!(
                    target = "llm_toolkit::dialogue",
                    mode = ?ExecutionModel::Broadcast,
                    participant = %participant_name,
                    participant_index = idx,
                    total_participants = participant_total,
                    event = "dialogue_turn_emitted"
                );
                Some(Ok(turn))
            }
            Err(err) => Some(Err(err)),
        }
    }
}

enum SessionState {
    Broadcast(BroadcastState),
    Sequential {
        next_index: usize,
        current_input: String,
    },
    Completed,
}

/// Represents an in-flight dialogue execution that can yield turns incrementally.
pub struct DialogueSession<'a> {
    dialogue: &'a mut Dialogue,
    state: SessionState,
    model: ExecutionModel,
}

impl<'a> DialogueSession<'a> {
    /// Returns the execution model backing this session.
    pub fn execution_model(&self) -> ExecutionModel {
        self.model
    }

    /// Retrieves the next available dialogue turn.
    ///
    /// Returns `None` when the session is complete.
    pub async fn next_turn(&mut self) -> Option<Result<DialogueTurn, AgentError>> {
        let participant_total = self.dialogue.participants.len();

        loop {
            match &mut self.state {
                SessionState::Broadcast(state) => {
                    if let Some(result) = state.try_emit(self.dialogue) {
                        return Some(result);
                    }

                    match state.pending.join_next().await {
                        Some(Ok((idx, name, result))) => {
                            let participant_name = name;
                            match state.order {
                                BroadcastOrder::Completion => match result {
                                    Ok(content) => {
                                        let turn = DialogueTurn {
                                            participant_name: participant_name.clone(),
                                            content: content.clone(),
                                        };
                                        self.dialogue.history.push(turn.clone());
                                        info!(
                                            target = "llm_toolkit::dialogue",
                                            mode = ?self.model,
                                            participant = %participant_name,
                                            participant_index = idx,
                                            total_participants = participant_total,
                                            event = "dialogue_turn_completed"
                                        );
                                        return Some(Ok(turn));
                                    }
                                    Err(err) => {
                                        error!(
                                            target = "llm_toolkit::dialogue",
                                            mode = ?self.model,
                                            participant = %participant_name,
                                            participant_index = idx,
                                            total_participants = participant_total,
                                            error = %err,
                                            event = "dialogue_turn_failed"
                                        );
                                        return Some(Err(err));
                                    }
                                },
                                BroadcastOrder::ParticipantOrder => {
                                    match &result {
                                        Ok(_) => {
                                            info!(
                                                target = "llm_toolkit::dialogue",
                                                mode = ?self.model,
                                                participant = %participant_name,
                                                participant_index = idx,
                                                total_participants = participant_total,
                                                event = "dialogue_turn_completed"
                                            );
                                        }
                                        Err(err) => {
                                            error!(
                                                target = "llm_toolkit::dialogue",
                                                mode = ?self.model,
                                                participant = %participant_name,
                                                participant_index = idx,
                                                total_participants = participant_total,
                                                error = %err,
                                                event = "dialogue_turn_failed"
                                            );
                                        }
                                    }
                                    state.record_result(idx, result);
                                    continue;
                                }
                            }
                        }
                        Some(Err(join_err)) => {
                            error!(
                                target = "llm_toolkit::dialogue",
                                mode = ?self.model,
                                error = %join_err,
                                event = "dialogue_task_join_failed"
                            );
                            return Some(Err(AgentError::ExecutionFailed(format!(
                                "Broadcast task failed: {}",
                                join_err
                            ))));
                        }
                        None => {
                            if let Some(result) = state.try_emit(self.dialogue) {
                                return Some(result);
                            }
                            self.state = SessionState::Completed;
                            return None;
                        }
                    }
                }
                SessionState::Sequential {
                    next_index,
                    current_input,
                } => {
                    if *next_index >= self.dialogue.participants.len() {
                        self.state = SessionState::Completed;
                        return None;
                    }

                    let idx = *next_index;
                    *next_index += 1;
                    let step_number = idx + 1;

                    let response_result = {
                        let participant = &self.dialogue.participants[idx];
                        let payload: Payload = current_input.clone().into();
                        participant.agent.execute(payload).await
                    };

                    return match response_result {
                        Ok(content) => {
                            *current_input = content.clone();
                            let participant_name =
                                self.dialogue.participants[idx].name().to_string();
                            let turn = DialogueTurn {
                                participant_name: participant_name.clone(),
                                content,
                            };
                            self.dialogue.history.push(turn.clone());
                            info!(
                                target = "llm_toolkit::dialogue",
                                mode = ?self.model,
                                participant = %participant_name,
                                step_index = idx,
                                step_number,
                                total_steps = participant_total,
                                event = "dialogue_turn_completed"
                            );
                            Some(Ok(turn))
                        }
                        Err(err) => {
                            error!(
                                target = "llm_toolkit::dialogue",
                                mode = ?self.model,
                                participant_index = idx,
                                step_number,
                                total_steps = participant_total,
                                error = %err,
                                event = "dialogue_turn_failed"
                            );
                            Some(Err(err))
                        }
                    };
                }
                SessionState::Completed => return None,
            }
        }
    }
}

/// A dialogue manager for multi-agent conversations.
///
/// The dialogue maintains a list of participants and a conversation history,
/// using a turn-taking strategy to determine execution behavior.
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
    participants: Vec<Participant>,
    history: Vec<DialogueTurn>,
    execution_model: ExecutionModel,
}

impl Dialogue {
    /// Creates a new dialogue with the specified execution model.
    ///
    /// This is private - use `broadcast()` or `sequential()` instead.
    fn new(execution_model: ExecutionModel) -> Self {
        Self {
            participants: Vec::new(),
            history: Vec::new(),
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
            history: Vec::new(),
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
            history: Vec::new(),
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
    pub fn partial_session(&mut self, initial_prompt: String) -> DialogueSession<'_> {
        self.partial_session_with_order(initial_prompt, BroadcastOrder::Completion)
    }

    /// Begins a dialogue session with a specified broadcast ordering policy.
    pub fn partial_session_with_order(
        &mut self,
        initial_prompt: String,
        broadcast_order: BroadcastOrder,
    ) -> DialogueSession<'_> {
        self.history.push(DialogueTurn {
            participant_name: "System".to_string(),
            content: initial_prompt.clone(),
        });

        let model = self.execution_model;
        let state = match model {
            ExecutionModel::Broadcast => {
                let history_text = self.format_history();
                let payload: Payload = history_text.into();
                let mut pending = JoinSet::new();

                for (idx, participant) in self.participants.iter().enumerate() {
                    let agent = Arc::clone(&participant.agent);
                    let payload_clone = payload.clone();
                    let name = participant.name().to_string();

                    pending.spawn(async move {
                        let result = agent.execute(payload_clone).await;
                        (idx, name, result)
                    });
                }

                SessionState::Broadcast(BroadcastState::new(
                    pending,
                    broadcast_order,
                    self.participants.len(),
                ))
            }
            ExecutionModel::Sequential => SessionState::Sequential {
                next_index: 0,
                current_input: initial_prompt,
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
    /// // Broadcast mode
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    /// let all_turns = dialogue.run("Discuss AI ethics".to_string()).await?;
    ///
    /// // Sequential mode
    /// let mut dialogue = Dialogue::sequential()
    ///     .add_participant(persona1, summarizer)
    ///     .add_participant(persona2, translator);
    /// let final_turn = dialogue.run("Process this".to_string()).await?;
    /// ```
    pub async fn run(&mut self, initial_prompt: String) -> Result<Vec<DialogueTurn>, AgentError> {
        let model = self.execution_model;
        let mut session =
            self.partial_session_with_order(initial_prompt, BroadcastOrder::Completion);
        let mut turns = Vec::new();

        while let Some(result) = session.next_turn().await {
            let turn = result?;
            turns.push(turn);
        }

        match model {
            ExecutionModel::Broadcast => Ok(turns),
            ExecutionModel::Sequential => Ok(turns.into_iter().last().into_iter().collect()),
        }
    }

    /// Formats the conversation history as a single string.
    ///
    /// This creates a formatted transcript of the dialogue that can be used
    /// as input for the next agent.
    fn format_history(&self) -> String {
        self.history
            .iter()
            .map(|turn| format!("[{}]: {}", turn.participant_name, turn.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns a reference to the conversation history.
    pub fn history(&self) -> &[DialogueTurn] {
        &self.history
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
        assert_eq!(turns[0].participant_name, "Agent1");
        assert_eq!(turns[0].content, "Response 1");
        assert_eq!(turns[1].participant_name, "Agent2");
        assert_eq!(turns[1].content, "Response 2");

        // Check history: System + 2 participant responses
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].participant_name, "System");
        assert_eq!(dialogue.history()[1].participant_name, "Agent1");
        assert_eq!(dialogue.history()[2].participant_name, "Agent2");
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
        assert_eq!(turns[0].participant_name, "Finalizer");
        assert_eq!(turns[0].content, "Final output: all done");

        // Check that history contains the correct number of turns
        // 1 (System) + 3 (participants) = 4 total
        assert_eq!(dialogue.history().len(), 4);

        // Verify the history structure
        assert_eq!(dialogue.history()[0].participant_name, "System");
        assert_eq!(dialogue.history()[0].content, "Initial prompt");

        assert_eq!(dialogue.history()[1].participant_name, "Summarizer");
        assert_eq!(dialogue.history()[1].content, "Summary: input received");

        assert_eq!(dialogue.history()[2].participant_name, "Translator");
        assert_eq!(dialogue.history()[2].content, "Translated: previous output");

        assert_eq!(dialogue.history()[3].participant_name, "Finalizer");
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
        assert_eq!(first.participant_name, "Step1");
        assert_eq!(first.content, "S1 output");

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.participant_name, "Step2");
        assert_eq!(second.content, "S2 output");

        assert!(session.next_turn().await.is_none());

        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].participant_name, "System");
        assert_eq!(dialogue.history()[1].participant_name, "Step1");
        assert_eq!(dialogue.history()[2].participant_name, "Step2");
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
        assert_eq!(first.participant_name, "Fast");
        assert!(first.content.contains("Fast handled"));

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.participant_name, "Slow");
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
        assert_eq!(first.participant_name, "Slow");
        assert!(first.content.contains("Slow handled"));

        let second = session.next_turn().await.unwrap().unwrap();
        assert_eq!(second.participant_name, "Fast");
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
        assert_eq!(turns[0].participant_name, "Agent1");
        assert_eq!(turns[1].participant_name, "Agent3");
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
        assert_eq!(turns[0].participant_name, "Alice");
        assert_eq!(turns[1].participant_name, "Bob");
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
}
