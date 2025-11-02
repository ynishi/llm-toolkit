//! Dialogue session management for incremental turn execution.
//!
//! This module provides the DialogueSession type which allows consuming
//! dialogue turns one at a time as they complete, enabling streaming
//! and responsive UIs.

use super::super::{AgentError, Payload};
use super::message::{DialogueMessage, Speaker};
use super::state::SessionState;
use super::{BroadcastOrder, Dialogue, DialogueTurn, ExecutionModel};
use crate::agent::PayloadMessage;
use tracing::{error, info};

/// Represents an in-flight dialogue execution that can yield turns incrementally.
pub struct DialogueSession<'a> {
    pub(super) dialogue: &'a mut Dialogue,
    pub(super) state: SessionState,
    pub(super) model: ExecutionModel,
}

impl<'a> DialogueSession<'a> {
    /// Returns the execution model backing this session.
    pub fn execution_model(&self) -> ExecutionModel {
        self.model
    }

    /// Starts a new turn in the dialogue with fresh user input.
    ///
    /// This method allows continuing a multi-turn conversation within the same session.
    /// The new input will be combined with context from previous turns (other agents' responses)
    /// automatically.
    ///
    /// # Arguments
    ///
    /// * `new_input` - The new user input/intent for this turn
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the new turn was started successfully, or an error if:
    /// - The session is still processing a previous turn (not completed)
    /// - The dialogue is in sequential mode (not supported for multi-turn within session)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut session = dialogue.partial_session("First question");
    /// while let Some(turn) = session.next_turn().await {
    ///     // Process first round of responses
    /// }
    ///
    /// // Continue with a new turn
    /// session.continue_with("Follow-up question")?;
    /// while let Some(turn) = session.next_turn().await {
    ///     // Process second round of responses
    /// }
    /// ```
    pub fn continue_with(&mut self, new_input: impl Into<Payload>) -> Result<(), AgentError> {
        // Only allow continuing if the current session is completed
        if !matches!(self.state, SessionState::Completed) {
            return Err(AgentError::ExecutionFailed(
                "Cannot start new turn: previous turn is still in progress".to_string(),
            ));
        }

        // Only supported for broadcast mode currently
        match self.model {
            ExecutionModel::Broadcast => {
                let payload: Payload = new_input.into();
                let current_turn = self.dialogue.message_store.current_turn() + 1;

                // Record user input in message store
                let user_message = DialogueMessage::new(
                    current_turn,
                    Speaker::user("User", "Human"),
                    payload.to_text(),
                );
                self.dialogue.message_store.push(user_message);

                // Spawn new broadcast tasks
                let pending = self.dialogue.spawn_broadcast_tasks(current_turn, &payload);

                // Get the broadcast order from the previous state if it was broadcast,
                // otherwise use default (Completion)
                let broadcast_order = BroadcastOrder::Completion;

                self.state = SessionState::Broadcast(super::state::BroadcastState::new(
                    pending,
                    broadcast_order,
                    self.dialogue.participants.len(),
                    current_turn,
                ));

                Ok(())
            }
            ExecutionModel::Sequential => Err(AgentError::ExecutionFailed(
                "Multi-turn within session is not yet supported for sequential mode".to_string(),
            )),
        }
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

                    let current_turn = state.current_turn;
                    match state.pending.join_next().await {
                        Some(Ok((idx, name, result))) => {
                            let participant_name = name;
                            match state.order {
                                BroadcastOrder::Completion => {
                                    match &result {
                                        Ok(content) => {
                                            // Store in MessageStore
                                            let participant = &self.dialogue.participants[idx];
                                            let message = DialogueMessage::new(
                                                current_turn,
                                                Speaker::agent(
                                                    participant_name.clone(),
                                                    participant.persona.role.clone(),
                                                ),
                                                content.clone(),
                                            );
                                            self.dialogue.message_store.push(message);

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
                                    // Record result and continue to collect all responses
                                    state.record_result(idx, participant_name, result);
                                    continue;
                                }
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
                                    state.record_result(idx, participant_name, result);
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
                    current_turn,
                } => {
                    if *next_index >= self.dialogue.participants.len() {
                        self.state = SessionState::Completed;
                        return None;
                    }

                    let idx = *next_index;
                    let turn = *current_turn;
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
                            let participant = &self.dialogue.participants[idx];
                            let participant_name = participant.name().to_string();

                            // Store in MessageStore
                            let message = DialogueMessage::new(
                                turn,
                                Speaker::agent(
                                    participant_name.clone(),
                                    participant.persona.role.clone(),
                                ),
                                content.clone(),
                            );
                            self.dialogue.message_store.push(message);

                            let turn = DialogueTurn {
                                speaker: Speaker::agent(
                                    participant_name.clone(),
                                    participant.persona.role.clone(),
                                ),
                                content,
                            };
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

impl<'a> Drop for DialogueSession<'a> {
    fn drop(&mut self) {
        if let SessionState::Broadcast(state) = &mut self.state {
            match state.order {
                BroadcastOrder::Completion => {
                    for (entry, (idx, participant_name)) in state
                        .buffered
                        .iter()
                        .zip(state.completion_metadata.iter())
                    {
                        if let Some(Ok(content)) = entry {
                            let participant = &self.dialogue.participants[*idx];
                            let pending = PayloadMessage::agent(
                                participant_name.clone(),
                                participant.persona.role.clone(),
                                content.clone(),
                            );
                            self.dialogue.pending_intent_prefix.push(pending);
                        }
                    }
                }
                BroadcastOrder::ParticipantOrder => {
                    for (idx, maybe_result) in state.buffered.iter().enumerate() {
                        if let Some(Ok(content)) = maybe_result {
                            let participant = &self.dialogue.participants[idx];
                            let pending = PayloadMessage::agent(
                                participant.name().to_string(),
                                participant.persona.role.clone(),
                                content.clone(),
                            );
                            self.dialogue.pending_intent_prefix.push(pending);
                        }
                    }
                }
            }
        }
    }
}
