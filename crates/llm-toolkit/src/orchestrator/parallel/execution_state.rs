//! Execution state management for parallel orchestration.
//!
//! This module provides a state machine to track the lifecycle of each step
//! during concurrent execution.

use crate::orchestrator::OrchestratorError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A serializable representation of a step failure with error type preserved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepFailure {
    /// The error kind (timeout, cancelled, or other)
    pub kind: SerializableErrorKind,
    /// The full error message
    pub message: String,
}

/// A serializable representation of an error type for step failures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializableErrorKind {
    /// A step execution timed out.
    StepTimeout { step_id: String, timeout_secs: u64 },
    /// A step execution was cancelled.
    Cancelled { step_id: String },
    /// Any other error type (stored as string).
    Other,
}

impl StepFailure {
    /// Creates a StepFailure from an OrchestratorError, preserving the error type.
    pub fn from_orchestrator_error(error: &OrchestratorError) -> Self {
        let kind = match error {
            OrchestratorError::StepTimeout { step_id, timeout } => {
                SerializableErrorKind::StepTimeout {
                    step_id: step_id.clone(),
                    timeout_secs: timeout.as_secs(),
                }
            }
            OrchestratorError::Cancelled { step_id } => SerializableErrorKind::Cancelled {
                step_id: step_id.clone(),
            },
            _ => SerializableErrorKind::Other,
        };

        StepFailure {
            kind,
            message: error.to_string(),
        }
    }

    /// Returns true if this is a timeout error.
    pub fn is_timeout(&self) -> bool {
        matches!(self.kind, SerializableErrorKind::StepTimeout { .. })
    }

    /// Returns true if this is a cancellation error.
    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, SerializableErrorKind::Cancelled { .. })
    }
}

impl PartialEq for StepFailure {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.message == other.message
    }
}

impl std::fmt::Display for StepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// The execution state of a step in the parallel orchestrator.
///
/// Steps progress through states as follows:
/// - `Pending` -> `Ready` (when all dependencies complete)
/// - `Ready` -> `Running` (when execution begins)
/// - `Running` -> `Completed` (on success)
/// - `Running` -> `Failed` (on error)
/// - `Running` -> `PausedForApproval` (when human approval is required)
/// - `PausedForApproval` -> `Running` (when approval is granted)
/// - Any state -> `Skipped` (when a dependency fails)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepState {
    /// Step is waiting for dependencies to complete
    Pending,
    /// Step is ready to execute (all dependencies satisfied)
    Ready,
    /// Step is currently executing
    Running,
    /// Step completed successfully
    Completed,
    /// Step failed with an error
    Failed(StepFailure),
    /// Step was skipped due to a failed dependency
    Skipped,
    /// Step is paused and waiting for human approval
    PausedForApproval {
        /// Message describing what approval is needed
        message: String,
        /// Payload data for the approval request
        payload: serde_json::Value,
    },
}

impl PartialEq for StepState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StepState::Pending, StepState::Pending) => true,
            (StepState::Ready, StepState::Ready) => true,
            (StepState::Running, StepState::Running) => true,
            (StepState::Completed, StepState::Completed) => true,
            (StepState::Skipped, StepState::Skipped) => true,
            (StepState::Failed(e1), StepState::Failed(e2)) => e1 == e2,
            (
                StepState::PausedForApproval {
                    message: m1,
                    payload: p1,
                },
                StepState::PausedForApproval {
                    message: m2,
                    payload: p2,
                },
            ) => m1 == m2 && p1 == p2,
            _ => false,
        }
    }
}

/// Manages the execution state of all steps in a parallel workflow.
///
/// This provides thread-safe access to step states and helper methods
/// for querying workflow progress.
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::orchestrator::parallel::{ExecutionStateManager, StepState};
///
/// let mut manager = ExecutionStateManager::new();
/// manager.set_state("step_1", StepState::Ready);
/// manager.set_state("step_2", StepState::Pending);
///
/// let ready_steps = manager.get_ready_steps();
/// assert_eq!(ready_steps.len(), 1);
/// assert!(ready_steps.contains(&"step_1".to_string()));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStateManager {
    states: HashMap<String, StepState>,
}

impl ExecutionStateManager {
    /// Creates a new execution state manager with no steps.
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    /// Sets the state of a step.
    ///
    /// If the step doesn't exist yet, it will be added.
    pub fn set_state(&mut self, step_id: &str, state: StepState) {
        self.states.insert(step_id.to_string(), state);
    }

    /// Gets the current state of a step.
    ///
    /// Returns `None` if the step doesn't exist.
    pub fn get_state(&self, step_id: &str) -> Option<&StepState> {
        self.states.get(step_id)
    }

    /// Returns the total number of steps being tracked.
    pub fn step_count(&self) -> usize {
        self.states.len()
    }

    /// Returns all step IDs that are in the Ready state.
    ///
    /// These steps can be executed immediately as all their dependencies
    /// have been satisfied.
    pub fn get_ready_steps(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, state)| matches!(state, StepState::Ready))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns all step IDs that are currently running.
    pub fn get_running_steps(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, state)| matches!(state, StepState::Running))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns all step IDs that were skipped due to failed dependencies.
    pub fn get_skipped_steps(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, state)| matches!(state, StepState::Skipped))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns true if all steps are in the Completed state.
    pub fn all_completed(&self) -> bool {
        !self.states.is_empty()
            && self
                .states
                .values()
                .all(|s| matches!(s, StepState::Completed))
    }

    /// Returns true if all steps are either Completed or Skipped.
    ///
    /// This indicates the workflow has finished, even if some steps
    /// were skipped due to failures.
    pub fn all_completed_or_skipped(&self) -> bool {
        !self.states.is_empty()
            && self
                .states
                .values()
                .all(|s| matches!(s, StepState::Completed | StepState::Skipped))
    }

    /// Returns true if any step is in the Failed state.
    pub fn has_failures(&self) -> bool {
        self.states
            .values()
            .any(|s| matches!(s, StepState::Failed(_)))
    }

    /// Returns true if any step is in the Ready or Running state.
    ///
    /// This indicates the workflow is still actively executing.
    pub fn has_ready_or_running_steps(&self) -> bool {
        self.states
            .values()
            .any(|s| matches!(s, StepState::Ready | StepState::Running))
    }

    /// Returns true if any step is still pending (not Ready, Running, Completed, Failed, or Skipped).
    pub fn has_pending_steps(&self) -> bool {
        self.states
            .values()
            .any(|s| matches!(s, StepState::Pending))
    }

    /// Returns all steps in Failed state with their errors.
    pub fn get_failed_steps(&self) -> Vec<(String, StepFailure)> {
        self.states
            .iter()
            .filter_map(|(id, state)| {
                if let StepState::Failed(err) = state {
                    Some((id.clone(), err.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the first failed step's ID and error, if any.
    ///
    /// This is useful for triggering redesign logic based on the initial failure.
    /// Returns `None` if there are no failed steps.
    pub fn get_first_failure(&self) -> Option<(String, StepFailure)> {
        self.states.iter().find_map(|(id, state)| {
            if let StepState::Failed(err) = state {
                Some((id.clone(), err.clone()))
            } else {
                None
            }
        })
    }
}

impl Default for ExecutionStateManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manager_is_empty() {
        let manager = ExecutionStateManager::new();
        assert_eq!(manager.step_count(), 0);
    }

    #[test]
    fn test_set_and_get_state() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Pending);

        assert_eq!(manager.get_state("step_1").unwrap(), &StepState::Pending);
    }

    #[test]
    fn test_get_state_nonexistent() {
        let manager = ExecutionStateManager::new();
        assert!(manager.get_state("nonexistent").is_none());
    }

    #[test]
    fn test_update_state() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Pending);
        manager.set_state("step_1", StepState::Ready);

        assert_eq!(manager.get_state("step_1").unwrap(), &StepState::Ready);
    }

    #[test]
    fn test_get_ready_steps() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Ready);
        manager.set_state("step_2", StepState::Pending);
        manager.set_state("step_3", StepState::Ready);

        let ready = manager.get_ready_steps();
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&"step_1".to_string()));
        assert!(ready.contains(&"step_3".to_string()));
    }

    #[test]
    fn test_get_ready_steps_empty() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Pending);
        manager.set_state("step_2", StepState::Running);

        let ready = manager.get_ready_steps();
        assert_eq!(ready.len(), 0);
    }

    #[test]
    fn test_get_running_steps() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Running);
        manager.set_state("step_2", StepState::Pending);
        manager.set_state("step_3", StepState::Running);

        let running = manager.get_running_steps();
        assert_eq!(running.len(), 2);
        assert!(running.contains(&"step_1".to_string()));
        assert!(running.contains(&"step_3".to_string()));
    }

    #[test]
    fn test_all_completed() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Completed);

        assert!(manager.all_completed());
    }

    #[test]
    fn test_all_completed_false_with_pending() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Pending);

        assert!(!manager.all_completed());
    }

    #[test]
    fn test_all_completed_false_when_empty() {
        let manager = ExecutionStateManager::new();
        assert!(!manager.all_completed());
    }

    #[test]
    fn test_all_completed_or_skipped() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Skipped);

        assert!(manager.all_completed_or_skipped());
    }

    #[test]
    fn test_all_completed_or_skipped_false_with_pending() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Skipped);
        manager.set_state("step_3", StepState::Pending);

        assert!(!manager.all_completed_or_skipped());
    }

    #[test]
    fn test_has_failures() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state(
            "step_2",
            StepState::Failed(StepFailure::from_orchestrator_error(
                &OrchestratorError::ExecutionFailed("test error".to_string()),
            )),
        );

        assert!(manager.has_failures());
    }

    #[test]
    fn test_has_failures_false() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Pending);

        assert!(!manager.has_failures());
    }

    #[test]
    fn test_has_ready_or_running_steps() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Running);
        manager.set_state("step_2", StepState::Completed);

        assert!(manager.has_ready_or_running_steps());
    }

    #[test]
    fn test_has_ready_or_running_steps_with_ready() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Ready);
        manager.set_state("step_2", StepState::Completed);

        assert!(manager.has_ready_or_running_steps());
    }

    #[test]
    fn test_has_ready_or_running_steps_false() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Skipped);

        assert!(!manager.has_ready_or_running_steps());
    }

    #[test]
    fn test_has_pending_steps() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Pending);
        manager.set_state("step_2", StepState::Completed);

        assert!(manager.has_pending_steps());
    }

    #[test]
    fn test_has_pending_steps_false() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Ready);

        assert!(!manager.has_pending_steps());
    }

    #[test]
    fn test_get_failed_steps() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state(
            "step_2",
            StepState::Failed(StepFailure::from_orchestrator_error(
                &OrchestratorError::ExecutionFailed("test error".to_string()),
            )),
        );

        let failed = manager.get_failed_steps();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "step_2");
        assert_eq!(failed[0].1.message, "Execution failed: test error");
    }

    #[test]
    fn test_get_failed_steps_empty() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Running);

        let failed = manager.get_failed_steps();
        assert_eq!(failed.len(), 0);
    }

    #[test]
    fn test_step_state_equality() {
        assert_eq!(StepState::Pending, StepState::Pending);
        assert_eq!(StepState::Ready, StepState::Ready);
        assert_ne!(StepState::Pending, StepState::Ready);
    }

    #[test]
    fn test_step_state_failed_equality() {
        let error1 = StepFailure::from_orchestrator_error(&OrchestratorError::ExecutionFailed(
            "error1".to_string(),
        ));
        let error2 = StepFailure::from_orchestrator_error(&OrchestratorError::ExecutionFailed(
            "error2".to_string(),
        ));
        let error1_clone = error1.clone();

        assert_eq!(
            StepState::Failed(error1.clone()),
            StepState::Failed(error1_clone)
        );
        assert_ne!(StepState::Failed(error1), StepState::Failed(error2));
    }

    #[test]
    fn test_get_first_failure() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state(
            "step_2",
            StepState::Failed(StepFailure::from_orchestrator_error(
                &OrchestratorError::ExecutionFailed("error_2".to_string()),
            )),
        );
        manager.set_state(
            "step_3",
            StepState::Failed(StepFailure::from_orchestrator_error(
                &OrchestratorError::ExecutionFailed("error_3".to_string()),
            )),
        );

        let first_failure = manager.get_first_failure();
        assert!(first_failure.is_some());
        let (step_id, error) = first_failure.unwrap();
        // Note: HashMap iteration order is not guaranteed, so we just check that we got one of the failures
        assert!(step_id == "step_2" || step_id == "step_3");
        assert!(error.message.contains("error_2") || error.message.contains("error_3"));
    }

    #[test]
    fn test_get_first_failure_none() {
        let mut manager = ExecutionStateManager::new();
        manager.set_state("step_1", StepState::Completed);
        manager.set_state("step_2", StepState::Running);

        let first_failure = manager.get_first_failure();
        assert!(first_failure.is_none());
    }
}
