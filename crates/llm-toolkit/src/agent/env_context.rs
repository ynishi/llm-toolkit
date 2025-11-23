//! Environment context - Raw runtime information from orchestrator or external sources
//!
//! This module provides `EnvContext` which contains raw execution state
//! information injected by the orchestrator layer or external systems. This is
//! the foundational context used by detectors to infer higher-level context
//! like task health and task types.

use serde::{Deserialize, Serialize};

/// Raw environment context from orchestrator or external sources.
///
/// This struct contains factual runtime information about the current execution
/// state, provided by the orchestrator or external systems. It does not contain
/// inferred/detected information - that belongs in `DetectedContext`.
///
/// # Usage Pattern
///
/// ```rust,ignore
/// // In Orchestrator
/// let env_ctx = EnvContext::new()
///     .with_step_info(StepInfo { ... })
///     .with_journal_summary(JournalSummary { ... });
///
/// let payload = Payload::text(intent)
///     .push_context(ExecutionContext::Env(env_ctx));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EnvContext {
    /// Current step information
    pub step_info: Option<StepInfo>,

    /// Task description from orchestrator
    pub task_description: Option<String>,

    /// Execution journal summary (aggregated statistics)
    pub journal_summary: Option<JournalSummary>,

    /// Number of redesigns/retries so far
    pub redesign_count: usize,

    /// Current strategy phase/stage name
    pub strategy_phase: Option<String>,
}

impl EnvContext {
    /// Creates a new empty environment context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current step information.
    pub fn with_step_info(mut self, info: StepInfo) -> Self {
        self.step_info = Some(info);
        self
    }

    /// Sets the task description.
    pub fn with_task_description(mut self, description: impl Into<String>) -> Self {
        self.task_description = Some(description.into());
        self
    }

    /// Sets the journal summary.
    pub fn with_journal_summary(mut self, summary: JournalSummary) -> Self {
        self.journal_summary = Some(summary);
        self
    }

    /// Sets the redesign count.
    pub fn with_redesign_count(mut self, count: usize) -> Self {
        self.redesign_count = count;
        self
    }

    /// Sets the strategy phase.
    pub fn with_strategy_phase(mut self, phase: impl Into<String>) -> Self {
        self.strategy_phase = Some(phase.into());
        self
    }
}

/// Information about the current execution step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepInfo {
    /// Step identifier (e.g., "step_1", "step_2")
    pub step_id: String,

    /// Human-readable step description
    pub description: String,

    /// Name of the agent assigned to this step
    pub assigned_agent: String,

    /// Expected output description
    pub expected_output: Option<String>,
}

impl StepInfo {
    /// Creates a new step info.
    pub fn new(
        step_id: impl Into<String>,
        description: impl Into<String>,
        assigned_agent: impl Into<String>,
    ) -> Self {
        Self {
            step_id: step_id.into(),
            description: description.into(),
            assigned_agent: assigned_agent.into(),
            expected_output: None,
        }
    }

    /// Sets the expected output description.
    pub fn with_expected_output(mut self, output: impl Into<String>) -> Self {
        self.expected_output = Some(output.into());
        self
    }
}

/// Aggregated statistics from execution journal.
///
/// This provides a high-level view of execution health without
/// requiring the agent to process the full journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalSummary {
    /// Total number of steps executed so far
    pub total_steps: usize,

    /// Number of failed steps
    pub failed_steps: usize,

    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,

    /// Number of consecutive failures (useful for detecting spiraling issues)
    pub consecutive_failures: usize,
}

impl JournalSummary {
    /// Creates a new journal summary.
    pub fn new(total_steps: usize, failed_steps: usize) -> Self {
        let success_rate = if total_steps > 0 {
            1.0 - (failed_steps as f64 / total_steps as f64)
        } else {
            1.0
        };

        Self {
            total_steps,
            failed_steps,
            success_rate,
            consecutive_failures: 0,
        }
    }

    /// Sets consecutive failures count.
    pub fn with_consecutive_failures(mut self, count: usize) -> Self {
        self.consecutive_failures = count;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_context_builder() {
        let ctx = EnvContext::new()
            .with_step_info(StepInfo::new("step_1", "Test step", "TestAgent"))
            .with_task_description("Test task")
            .with_redesign_count(2);

        assert_eq!(ctx.step_info.as_ref().unwrap().step_id, "step_1");
        assert_eq!(ctx.task_description, Some("Test task".to_string()));
        assert_eq!(ctx.redesign_count, 2);
    }

    #[test]
    fn test_journal_summary_success_rate() {
        let summary = JournalSummary::new(10, 3);
        assert_eq!(summary.total_steps, 10);
        assert_eq!(summary.failed_steps, 3);
        assert!((summary.success_rate - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_journal_summary_empty() {
        let summary = JournalSummary::new(0, 0);
        assert_eq!(summary.success_rate, 1.0);
    }
}
