//! Dialogue session management for incremental turn execution.
//!
//! This module provides the DialogueSession type which allows consuming
//! dialogue turns one at a time as they complete, enabling streaming
//! and responsive UIs.

use super::super::{Agent, AgentError, Payload, PayloadMessage};
use super::message::{DialogueMessage, MessageMetadata, MessageOrigin, Speaker};
use super::state::SessionState;
use super::{BroadcastOrder, Dialogue, DialogueTurn, ExecutionModel, ParticipantInfo};
use crate::prompt::ToPrompt;
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
        self.model.clone()
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
                            match &state.order {
                                BroadcastOrder::Completion => {
                                    match &result {
                                        Ok(content) => {
                                            // Store in MessageStore
                                            let participant = &self.dialogue.participants[idx];
                                            let metadata = MessageMetadata::new()
                                                .with_origin(MessageOrigin::AgentGenerated);
                                            let message = DialogueMessage::new(
                                                current_turn,
                                                Speaker::agent(
                                                    participant_name.clone(),
                                                    participant.persona.role.clone(),
                                                ),
                                                content.clone(),
                                            )
                                            .with_metadata(&metadata);
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
                                BroadcastOrder::Explicit(_) => {
                                    // For Explicit order, process results based on the specified order
                                    // For now, implement similar to Completion order
                                    match &result {
                                        Ok(content) => {
                                            // Store in MessageStore
                                            let participant = &self.dialogue.participants[idx];
                                            let metadata = MessageMetadata::new()
                                                .with_origin(MessageOrigin::AgentGenerated);
                                            let message = DialogueMessage::new(
                                                current_turn,
                                                Speaker::agent(
                                                    participant_name.clone(),
                                                    participant.persona.role.clone(),
                                                ),
                                                content.clone(),
                                            )
                                            .with_metadata(&metadata);
                                            self.dialogue.message_store.push(message);

                                            info!(
                                                target = "llm_toolkit::dialogue",
                                                mode = ?self.model,
                                                turn = current_turn,
                                                speaker = %participant_name,
                                                role = %participant.persona.role,
                                                participant_index = idx,
                                                total_participants = participant_total,
                                                event = "dialogue_turn_completed"
                                            );
                                        }
                                        Err(err) => {
                                            error!(
                                                target = "llm_toolkit::dialogue",
                                                mode = ?self.model,
                                                turn = current_turn,
                                                speaker = %participant_name,
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
                    current_turn,
                    sequence,
                    payload,
                    prev_agent_outputs,
                    current_turn_outputs,
                    participants_info,
                } => {
                    if sequence.is_empty() || *next_index >= sequence.len() {
                        self.state = SessionState::Completed;
                        return None;
                    }

                    let sequence_position = *next_index;
                    let participant_idx = sequence[sequence_position];
                    let turn = *current_turn;
                    *next_index += 1;
                    let step_number = sequence_position + 1;
                    let step_total = sequence.len();

                    let mut response_payload = build_sequential_payload(
                        payload,
                        prev_agent_outputs.as_slice(),
                        current_turn_outputs.as_slice(),
                        participants_info.as_slice(),
                        sequence_position,
                    );

                    // Attach context if exists
                    if let Some(ref context) = self.dialogue.context {
                        response_payload = response_payload.with_context(context.to_prompt());
                    }

                    // Handle initial join if this participant hasn't sent a message yet
                    let participant = &self.dialogue.participants[participant_idx];
                    let is_initial_join = !participant.has_sent_once;
                    let joining_strategy = participant.joining_strategy;

                    if is_initial_join {
                        if let Some(strategy) = joining_strategy {
                            // Apply joining strategy: filter history messages
                            let all_messages = self.dialogue.message_store.all_messages();
                            let message_refs: Vec<&DialogueMessage> =
                                all_messages.iter().copied().collect();
                            let filtered_history = strategy.filter_messages(&message_refs, turn + 1);

                            // Collect message IDs (for marking as sent)
                            let all_past_message_ids: Vec<_> = all_messages
                                .iter()
                                .filter(|msg| msg.turn < turn)
                                .map(|msg| msg.id)
                                .collect();

                            // Convert filtered history to PayloadMessage
                            let history_messages: Vec<PayloadMessage> = filtered_history
                                .into_iter()
                                .map(|msg| PayloadMessage::from(msg.clone()))
                                .collect();

                            let filtered_count = history_messages.len();

                            // Mark ALL past messages as sent to this participant
                            self.dialogue
                                .message_store
                                .mark_all_as_sent(&all_past_message_ids);

                            // Prepend filtered history to the payload
                            if !history_messages.is_empty() {
                                let mut all_messages_for_payload = history_messages;
                                all_messages_for_payload.extend(response_payload.to_messages());
                                response_payload = Payload::from_messages(all_messages_for_payload);

                                // Re-apply context and participants
                                if let Some(ref context) = self.dialogue.context {
                                    response_payload =
                                        response_payload.with_context(context.to_prompt());
                                }
                                response_payload =
                                    response_payload.with_participants(participants_info.to_vec());
                            }

                            tracing::trace!(
                                target = "llm_toolkit::dialogue",
                                participant = participant.name(),
                                strategy = ?strategy,
                                filtered_count = filtered_count,
                                marked_sent_count = all_past_message_ids.len(),
                                "Applied joining strategy for initial join (sequential partial_session)"
                            );
                        }
                    }

                    let response_result = {
                        let participant = &self.dialogue.participants[participant_idx];
                        participant.agent.execute(response_payload).await
                    };

                    return match response_result {
                        Ok(content) => {
                            let participant = &self.dialogue.participants[participant_idx];
                            let participant_name = participant.name().to_string();
                            let speaker = Speaker::agent(
                                participant_name.clone(),
                                participant.persona.role.clone(),
                            );

                            // Store in MessageStore
                            let metadata =
                                MessageMetadata::new().with_origin(MessageOrigin::AgentGenerated);
                            let message =
                                DialogueMessage::new(turn, speaker.clone(), content.clone())
                                    .with_metadata(&metadata);
                            self.dialogue.message_store.push(message);

                            current_turn_outputs
                                .push(PayloadMessage::new(speaker.clone(), content.clone()));

                            // Mark participant as having sent once (after successful execution)
                            if is_initial_join {
                                self.dialogue.participants[participant_idx].has_sent_once = true;
                            }

                            let turn = DialogueTurn { speaker, content };
                            info!(
                                target = "llm_toolkit::dialogue",
                                mode = ?self.model,
                                participant = %participant_name,
                                step_index = participant_idx,
                                step_number,
                                total_steps = step_total,
                                event = "dialogue_turn_completed"
                            );
                            Some(Ok(turn))
                        }
                        Err(err) => {
                            error!(
                                target = "llm_toolkit::dialogue",
                                mode = ?self.model,
                                participant_index = participant_idx,
                                step_number,
                                total_steps = step_total,
                                error = %err,
                                event = "dialogue_turn_failed"
                            );
                            Some(Err(err))
                        }
                    };
                }
                SessionState::Failed(error) => {
                    if let Some(err) = error.take() {
                        self.state = SessionState::Completed;
                        return Some(Err(err));
                    }
                    self.state = SessionState::Completed;
                    return None;
                }
                SessionState::Completed => return None,
            }
        }
    }
}

fn build_sequential_payload(
    base_payload: &Payload,
    prev_agent_outputs: &[PayloadMessage],
    current_turn_outputs: &[PayloadMessage],
    participants_info: &[ParticipantInfo],
    idx: usize,
) -> Payload {
    if idx == 0 {
        let mut payload = base_payload.clone();

        if !prev_agent_outputs.is_empty() {
            payload = Payload::from_messages(prev_agent_outputs.to_vec()).merge(payload);
        }

        payload.with_participants(participants_info.to_vec())
    } else {
        // For idx > 0:
        // 1. prev_agent_outputs (prior turn's agent outputs)
        // 2. base_payload (new input)
        // 3. current_turn_outputs (current turn's agent outputs so far)
        let mut payload = base_payload.clone();

        // Prepend prev_agent_outputs if present
        if !prev_agent_outputs.is_empty() {
            payload = Payload::from_messages(prev_agent_outputs.to_vec()).merge(payload);
        }

        // Append current_turn_outputs
        if !current_turn_outputs.is_empty() {
            payload = payload.merge(Payload::from_messages(current_turn_outputs.to_vec()));
        }

        payload.with_participants(participants_info.to_vec())
    }
}
