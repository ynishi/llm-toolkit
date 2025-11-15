use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::time::{SystemTime, UNIX_EPOCH};

use super::strategy::{StrategyMap, StrategyStep};

/// Captures the execution plan and per-step outcomes for a workflow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionJournal {
    /// Strategy snapshot used for the run.
    pub strategy: StrategyMap,
    /// Recorded step outcomes in execution order.
    pub steps: Vec<StepRecord>,
}

impl ExecutionJournal {
    /// Creates a new journal with the given strategy snapshot.
    pub fn new(strategy: StrategyMap) -> Self {
        Self {
            strategy,
            steps: Vec::new(),
        }
    }

    /// Appends a step record to the journal.
    pub fn record_step(&mut self, record: StepRecord) {
        self.steps.push(record);
    }
}

/// Execution status for a strategy step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    PausedForApproval,
}

/// Snapshot of a single step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step_id: String,
    pub title: String,
    pub agent: String,
    pub status: StepStatus,
    pub output_key: Option<String>,
    pub output: Option<JsonValue>,
    pub error: Option<String>,
    pub recorded_at_ms: u64,
}

impl StepRecord {
    /// Builds a new record from a strategy step and status metadata.
    pub fn from_step(
        step: &StrategyStep,
        status: StepStatus,
        output: Option<JsonValue>,
        error: Option<String>,
    ) -> Self {
        Self::with_timestamp(step, status, output, error, current_timestamp_ms())
    }

    /// Same as `from_step` but with explicit timestamp control (useful for deterministic tests).
    pub fn with_timestamp(
        step: &StrategyStep,
        status: StepStatus,
        output: Option<JsonValue>,
        error: Option<String>,
        recorded_at_ms: u64,
    ) -> Self {
        Self {
            step_id: step.step_id.clone(),
            title: step.description.clone(),
            agent: step.assigned_agent.clone(),
            status,
            output_key: step.output_key.clone(),
            output,
            error,
            recorded_at_ms,
        }
    }
}

/// Returns the current system time in milliseconds since UNIX_EPOCH.
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
