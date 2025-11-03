//! Session state management for dialogue execution.
//!
//! This module contains the state machine for managing dialogue sessions,
//! including broadcast and sequential execution modes.

use super::super::{AgentError, Payload, PayloadMessage};
use super::message::{DialogueMessage, Speaker};
use super::{BroadcastOrder, Dialogue, DialogueTurn, ExecutionModel, ParticipantInfo};
use tokio::task::JoinSet;
use tracing::{info, trace};

/// Internal state for broadcast execution.
pub(super) struct BroadcastState {
    pub(super) pending: JoinSet<(usize, String, Result<String, AgentError>)>,
    pub(super) order: BroadcastOrder,
    pub(super) buffered: Vec<Option<Result<String, AgentError>>>,
    pub(super) next_emit: usize,
    pub(super) current_turn: usize,
    /// For Completion mode: stores (participant_idx, participant_name) for each buffered result
    pub(super) completion_metadata: Vec<(usize, String)>,
}

impl BroadcastState {
    pub(super) fn new(
        pending: JoinSet<(usize, String, Result<String, AgentError>)>,
        order: BroadcastOrder,
        participant_count: usize,
        current_turn: usize,
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
            current_turn,
            completion_metadata: Vec::new(),
        }
    }

    pub(super) fn record_result(
        &mut self,
        idx: usize,
        participant_name: String,
        result: Result<String, AgentError>,
    ) {
        match self.order {
            BroadcastOrder::Completion => {
                // For Completion mode, append to buffered with metadata
                let content_len = result.as_ref().map(|s| s.len()).unwrap_or(0);
                trace!(
                    target = "llm_toolkit::dialogue",
                    participant = %participant_name,
                    participant_index = idx,
                    content_length = content_len,
                    is_error = result.is_err(),
                    turn = self.current_turn,
                    "Recording result to buffered (Completion mode)"
                );
                self.buffered.push(Some(result));
                self.completion_metadata.push((idx, participant_name));
            }
            BroadcastOrder::ParticipantOrder => {
                if idx < self.buffered.len() {
                    let content_len = result.as_ref().map(|s| s.len()).unwrap_or(0);
                    trace!(
                        target = "llm_toolkit::dialogue",
                        participant = %participant_name,
                        participant_index = idx,
                        content_length = content_len,
                        is_error = result.is_err(),
                        turn = self.current_turn,
                        "Recording result to buffered (ParticipantOrder mode)"
                    );
                    self.buffered[idx] = Some(result);
                }
            }
        }
    }

    pub(super) fn try_emit(
        &mut self,
        dialogue: &mut Dialogue,
    ) -> Option<Result<DialogueTurn, AgentError>> {
        match self.order {
            BroadcastOrder::Completion => {
                // For Completion mode, emit results in completion order
                if self.buffered.is_empty() {
                    return None;
                }

                // Pop from front (FIFO - first completed, first emitted)
                let result = self.buffered.remove(0).expect("checked is_empty");
                let (idx, participant_name) = self.completion_metadata.remove(0);

                match result {
                    Ok(content) => {
                        // Content is already stored in MessageStore by session.rs
                        // Just create and return the DialogueTurn
                        let participant = &dialogue.participants[idx];
                        let turn = DialogueTurn {
                            speaker: Speaker::agent(
                                participant_name.clone(),
                                participant.persona.role.clone(),
                            ),
                            content: content.clone(),
                        };
                        info!(
                            target = "llm_toolkit::dialogue",
                            mode = ?ExecutionModel::Broadcast,
                            participant = %participant_name,
                            participant_index = idx,
                            total_participants = dialogue.participants.len(),
                            event = "dialogue_turn_emitted"
                        );
                        Some(Ok(turn))
                    }
                    Err(err) => Some(Err(err)),
                }
            }
            BroadcastOrder::ParticipantOrder => {
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
                        let participant = &dialogue.participants[idx];
                        let participant_name = participant.name().to_string();

                        // Store in MessageStore
                        let message = DialogueMessage::new(
                            self.current_turn,
                            Speaker::agent(
                                participant_name.clone(),
                                participant.persona.role.clone(),
                            ),
                            content.clone(),
                        );
                        dialogue.message_store.push(message);

                        let turn = DialogueTurn {
                            speaker: Speaker::agent(
                                participant_name.clone(),
                                participant.persona.role.clone(),
                            ),
                            content: content.clone(),
                        };
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
    }
}

/// Session state enum for tracking execution progress.
pub(super) enum SessionState {
    Broadcast(BroadcastState),
    Sequential {
        next_index: usize,
        current_turn: usize,
        payload: Payload,
        prev_agent_outputs: Vec<PayloadMessage>,
        current_turn_outputs: Vec<PayloadMessage>,
        participants_info: Vec<ParticipantInfo>,
    },
    Completed,
}
