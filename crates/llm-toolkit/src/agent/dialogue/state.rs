//! Session state management for dialogue execution.
//!
//! This module contains the state machine for managing dialogue sessions,
//! including broadcast and sequential execution modes.

use super::super::AgentError;
use super::message::{DialogueMessage, Speaker};
use super::{BroadcastOrder, Dialogue, DialogueTurn, ExecutionModel};
use tokio::task::JoinSet;
use tracing::info;

/// Internal state for broadcast execution.
pub(super) struct BroadcastState {
    pub(super) pending: JoinSet<(usize, String, Result<String, AgentError>)>,
    pub(super) order: BroadcastOrder,
    pub(super) buffered: Vec<Option<Result<String, AgentError>>>,
    pub(super) next_emit: usize,
    pub(super) current_turn: usize,
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
        }
    }

    pub(super) fn record_result(&mut self, idx: usize, result: Result<String, AgentError>) {
        if matches!(self.order, BroadcastOrder::ParticipantOrder) && idx < self.buffered.len() {
            self.buffered[idx] = Some(result);
        }
    }

    pub(super) fn try_emit(
        &mut self,
        dialogue: &mut Dialogue,
    ) -> Option<Result<DialogueTurn, AgentError>> {
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
                let participant = &dialogue.participants[idx];
                let participant_name = participant.name().to_string();

                // Store in MessageStore
                let message = DialogueMessage::new(
                    self.current_turn,
                    Speaker::participant(
                        participant_name.clone(),
                        participant.persona.role.clone(),
                    ),
                    content.clone(),
                );
                dialogue.message_store.push(message);

                let turn = DialogueTurn {
                    participant_name: participant_name.clone(),
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

/// Session state enum for tracking execution progress.
pub(super) enum SessionState {
    Broadcast(BroadcastState),
    Sequential {
        next_index: usize,
        current_input: String,
        current_turn: usize,
    },
    Completed,
}
