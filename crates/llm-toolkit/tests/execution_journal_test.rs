//! Integration tests for ExecutionJournal
//!
//! These tests verify the journal recording functionality for both
//! Orchestrator and ParallelOrchestrator.

use llm_toolkit::agent::{Agent, AgentError, AgentOutput, DynamicAgent, Payload};
use llm_toolkit::orchestrator::{
    BlueprintWorkflow, ExecutionJournal, ParallelOrchestrator, StepRecord, StepStatus, StrategyMap,
    StrategyStep,
};
use serde_json::{Value as JsonValue, json};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Mock Agents
// ============================================================================

#[derive(Clone)]
struct MockSuccessAgent {
    name: String,
    output: JsonValue,
}

impl MockSuccessAgent {
    fn new(name: impl Into<String>, output: JsonValue) -> Self {
        Self {
            name: name.into(),
            output,
        }
    }
}

#[async_trait::async_trait]
impl Agent for MockSuccessAgent {
    type Output = JsonValue;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock success agent for journal testing";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.output.clone())
    }
}

#[async_trait::async_trait]
impl DynamicAgent for MockSuccessAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn expertise(&self) -> &str {
        Agent::expertise(self)
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<AgentOutput, AgentError> {
        let output = self.execute(input).await?;
        Ok(AgentOutput::Success(output))
    }
}

#[derive(Clone)]
struct MockFailingAgent {
    name: String,
}

impl MockFailingAgent {
    fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait::async_trait]
impl Agent for MockFailingAgent {
    type Output = JsonValue;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock failing agent for journal testing";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Err(AgentError::ExecutionFailed(
            "Intentional failure".to_string(),
        ))
    }
}

#[async_trait::async_trait]
impl DynamicAgent for MockFailingAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn expertise(&self) -> &str {
        Agent::expertise(self)
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<AgentOutput, AgentError> {
        self.execute(input).await?;
        unreachable!()
    }
}

// ============================================================================
// Tests for ExecutionJournal basic functionality
// ============================================================================

#[test]
fn test_execution_journal_creation() {
    let strategy = StrategyMap::new("Test Strategy".to_string());
    let journal = ExecutionJournal::new(strategy.clone());

    assert_eq!(journal.strategy.goal, "Test Strategy");
    assert_eq!(journal.steps.len(), 0);
}

#[test]
fn test_execution_journal_record_step() {
    let mut strategy = StrategyMap::new("Test Strategy".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "First step".to_string(),
        "TestAgent".to_string(),
        "Do something".to_string(),
        "Step 1".to_string(),
    ));

    let mut journal = ExecutionJournal::new(strategy.clone());
    let step = &strategy.steps[0];

    let record = StepRecord::from_step(
        step,
        StepStatus::Completed,
        Some(json!({"result": "success"})),
        None,
    );

    journal.record_step(record);

    assert_eq!(journal.steps.len(), 1);
    assert_eq!(journal.steps[0].step_id, "step_1");
    assert_eq!(journal.steps[0].title, "First step");
    assert!(matches!(journal.steps[0].status, StepStatus::Completed));
}

#[test]
fn test_step_record_with_timestamp() {
    let step = StrategyStep::new(
        "step_1".to_string(),
        "Test step".to_string(),
        "TestAgent".to_string(),
        "Do something".to_string(),
        "Step 1".to_string(),
    );

    let fixed_timestamp = 1234567890u64;
    let record =
        StepRecord::with_timestamp(&step, StepStatus::Running, None, None, fixed_timestamp);

    assert_eq!(record.recorded_at_ms, fixed_timestamp);
    assert!(matches!(record.status, StepStatus::Running));
}

// ============================================================================
// Tests for ParallelOrchestrator journal integration
// ============================================================================

#[tokio::test]
async fn test_parallel_orchestrator_journal_on_success() {
    let mut strategy = StrategyMap::new("Test Journal".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "First step".to_string(),
        "Agent1".to_string(),
        "Process task".to_string(),
        "Step 1".to_string(),
    ));
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Second step".to_string(),
        "Agent2".to_string(),
        "Process task".to_string(),
        "Step 2".to_string(),
    ));

    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = ParallelOrchestrator::new(blueprint);
    orchestrator.set_strategy_map(strategy);

    let agent1 = Arc::new(MockSuccessAgent::new("Agent1", json!({"data": "step1"})));
    let agent2 = Arc::new(MockSuccessAgent::new("Agent2", json!({"data": "step2"})));
    orchestrator.add_agent("Agent1", agent1);
    orchestrator.add_agent("Agent2", agent2);

    let result = orchestrator
        .execute("test task", CancellationToken::new(), None, None)
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 2);

    // Verify journal was captured
    let journal = result.journal.expect("Journal should be captured");
    assert_eq!(journal.strategy.goal, "Test Journal");
    assert_eq!(journal.steps.len(), 2);

    // Verify step statuses
    let step1_record = journal
        .steps
        .iter()
        .find(|s| s.step_id == "step_1")
        .expect("step_1 should be in journal");
    let step2_record = journal
        .steps
        .iter()
        .find(|s| s.step_id == "step_2")
        .expect("step_2 should be in journal");

    assert!(matches!(step1_record.status, StepStatus::Completed));
    assert!(matches!(step2_record.status, StepStatus::Completed));
}

#[derive(Clone)]
struct NoRedesignAgent;

#[async_trait::async_trait]
impl Agent for NoRedesignAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock agent that never triggers redesign";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok("FAIL".to_string())
    }
}

#[derive(Clone)]
struct DummyStrategyGenerator;

#[async_trait::async_trait]
impl Agent for DummyStrategyGenerator {
    type Output = StrategyMap;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Dummy strategy generator (should not be called)";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        panic!("DummyStrategyGenerator should never be called");
    }
}

#[tokio::test]
async fn test_parallel_orchestrator_journal_on_failure() {
    let mut strategy = StrategyMap::new("Test Failure Journal".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Failing step".to_string(),
        "FailAgent".to_string(),
        "Process task".to_string(),
        "Step 1".to_string(),
    ));
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Dependent step".to_string(),
        "Agent2".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Step 2".to_string(),
    ));

    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    // Use with_internal_agents to inject NoRedesignAgent to prevent redesign
    let internal_agent = Box::new(NoRedesignAgent);
    let internal_json_agent = Box::new(DummyStrategyGenerator);
    let mut orchestrator =
        ParallelOrchestrator::with_internal_agents(blueprint, internal_agent, internal_json_agent);
    orchestrator.set_strategy_map(strategy);

    let fail_agent = Arc::new(MockFailingAgent::new("FailAgent"));
    let agent2 = Arc::new(MockSuccessAgent::new("Agent2", json!({"data": "step2"})));
    orchestrator.add_agent("FailAgent", fail_agent);
    orchestrator.add_agent("Agent2", agent2);

    let result = orchestrator
        .execute("test task", CancellationToken::new(), None, None)
        .await
        .unwrap();

    assert!(!result.success);

    // Verify journal captures failure
    let journal = result
        .journal
        .expect("Journal should be captured on failure");
    assert_eq!(journal.steps.len(), 2);

    let step1_record = journal
        .steps
        .iter()
        .find(|s| s.step_id == "step_1")
        .expect("step_1 should be in journal");
    let step2_record = journal
        .steps
        .iter()
        .find(|s| s.step_id == "step_2")
        .expect("step_2 should be in journal");

    assert!(matches!(step1_record.status, StepStatus::Failed));
    assert!(step1_record.error.is_some());

    // step_2 should be skipped due to failed dependency
    assert!(matches!(step2_record.status, StepStatus::Skipped));
}

#[tokio::test]
async fn test_parallel_orchestrator_execution_journal_accessor() {
    let mut strategy = StrategyMap::new("Test Journal Accessor".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "Process".to_string(),
        "Step 1".to_string(),
    ));

    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = ParallelOrchestrator::new(blueprint);
    orchestrator.set_strategy_map(strategy);

    let agent = Arc::new(MockSuccessAgent::new("Agent1", json!({"result": "ok"})));
    orchestrator.add_agent("Agent1", agent);

    // Before execution, journal should be None
    assert!(orchestrator.execution_journal().is_none());

    let _ = orchestrator
        .execute("test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    // After execution, journal should be available
    let journal = orchestrator
        .execution_journal()
        .expect("Journal should be available after execution");

    assert_eq!(journal.strategy.goal, "Test Journal Accessor");
    assert_eq!(journal.steps.len(), 1);
}
