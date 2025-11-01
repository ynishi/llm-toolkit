//! Dialogue session management for incremental turn execution.
//!
//! This module provides the DialogueSession type which allows consuming
//! dialogue turns one at a time as they complete, enabling streaming
//! and responsive UIs.

use super::super::{AgentError, Payload};
use super::message::{DialogueMessage, Speaker};
use super::state::SessionState;
use super::{BroadcastOrder, Dialogue, DialogueTurn, ExecutionModel};
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
                                BroadcastOrder::Completion => match result {
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

                                        let turn = DialogueTurn {
                                            speaker: Speaker::agent(
                                                participant_name.clone(),
                                                participant.persona.role.clone(),
                                            ),
                                            content: content.clone(),
                                        };
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
