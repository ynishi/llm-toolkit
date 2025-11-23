//! Integration tests for ParallelOrchestrator Human-in-the-Loop (HIL) functionality.
//!
//! These tests verify the orchestrator's ability to pause for human approval
//! and resume execution after approval is granted.

use llm_toolkit::agent::{Agent, AgentError, AgentOutput, DynamicAgent, Payload};
use llm_toolkit::orchestrator::{
    BlueprintWorkflow, OrchestrationState, ParallelOrchestrator, StrategyMap, StrategyStep,
    parallel::StepState,
};
use serde_json::{Value as JsonValue, json};
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// Mock Agents
// ============================================================================

/// Mock agent that always requests approval
#[derive(Clone)]
struct ApprovalAgent {
    agent_name: String,
    approval_message: String,
    payload: JsonValue,
}

impl ApprovalAgent {
    fn new(name: impl Into<String>, message: impl Into<String>, payload: JsonValue) -> Self {
        Self {
            agent_name: name.into(),
            approval_message: message.into(),
            payload,
        }
    }
}

#[async_trait::async_trait]
impl Agent for ApprovalAgent {
    type Output = JsonValue;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock agent that requests human approval";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        // This should never be called directly - execute_dynamic is used instead
        unreachable!("ApprovalAgent should use execute_dynamic")
    }

    fn name(&self) -> String {
        "ApprovalAgent".to_string()
    }
}

#[async_trait::async_trait]
impl DynamicAgent for ApprovalAgent {
    fn name(&self) -> String {
        self.agent_name.clone()
    }

    fn description(&self) -> &str {
        "Mock agent that requests human approval"
    }

    async fn execute_dynamic(&self, _input: Payload) -> Result<AgentOutput, AgentError> {
        Ok(AgentOutput::RequiresApproval {
            message_for_human: self.approval_message.clone(),
            current_payload: self.payload.clone(),
        })
    }
}

/// Simple mock agent that returns a JSON value
#[derive(Clone)]
struct MockAgent {
    agent_name: String,
    output: JsonValue,
}

impl MockAgent {
    fn new(name: impl Into<String>, output: JsonValue) -> Self {
        Self {
            agent_name: name.into(),
            output,
        }
    }
}

#[async_trait::async_trait]
impl Agent for MockAgent {
    type Output = JsonValue;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock agent for testing";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.output.clone())
    }

    fn name(&self) -> String {
        self.agent_name.clone()
    }
}

#[async_trait::async_trait]
impl DynamicAgent for MockAgent {
    fn name(&self) -> String {
        self.agent_name.clone()
    }

    fn description(&self) -> &str {
        "Mock agent for testing"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<AgentOutput, AgentError> {
        let output = self.execute(input).await?;
        Ok(AgentOutput::Success(output))
    }
}

// ============================================================================
// HIL Tests
// ============================================================================

/// Test that the orchestrator pauses when an agent requests approval
#[tokio::test]
async fn test_orchestrator_pauses_on_approval_request() {
    let mut strategy = StrategyMap::new("Approval Test".to_string());

    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Request approval".to_string(),
        "ApprovalAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Approval requested".to_string(),
    ));

    let blueprint = BlueprintWorkflow::new("Test Blueprint".to_string());
    let mut orchestrator = ParallelOrchestrator::new(blueprint);
    orchestrator.set_strategy(strategy);

    // Register approval agent
    orchestrator.add_agent(
        "ApprovalAgent",
        Arc::new(ApprovalAgent::new(
            "ApprovalAgent",
            "Please approve this action",
            json!({"action": "test_action", "details": "needs review"}),
        )),
    );

    let result = orchestrator
        .execute(
            "test task",
            tokio_util::sync::CancellationToken::new(),
            None,
            None,
        )
        .await
        .unwrap();

    // Verify execution paused
    assert!(result.paused, "Execution should be paused");
    assert!(result.success, "Paused execution is still successful");
    assert_eq!(
        result.steps_executed, 0,
        "No steps should complete when paused"
    );
    assert!(
        result.pause_reason.is_some(),
        "Pause reason should be present"
    );
    assert_eq!(
        result.pause_reason.unwrap(),
        "Please approve this action",
        "Pause reason should match approval message"
    );
}

/// Test comprehensive HIL workflow: pause -> manual approval -> resume
#[tokio::test]
async fn test_orchestrator_resumes_after_approval() {
    // Create temporary directory for state file
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let state_file_path = temp_dir.path().join("hil_state.json");

    // ========================================================================
    // Setup: Two-step workflow (step_A -> step_B)
    // ========================================================================

    let mut strategy = StrategyMap::new("HIL Resume Test".to_string());

    strategy.add_step(StrategyStep::new(
        "step_A".to_string(),
        "Request approval".to_string(),
        "ApprovalAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Approval requested".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "step_B".to_string(),
        "Process after approval".to_string(),
        "MockAgent".to_string(),
        "Use approved data: {{ step_A_output }}".to_string(),
        "Final result".to_string(),
    ));

    // ========================================================================
    // First Run: Execute until pause
    // ========================================================================

    let blueprint = BlueprintWorkflow::new("Test Blueprint".to_string());
    let mut orchestrator_first = ParallelOrchestrator::new(blueprint);
    orchestrator_first.set_strategy(strategy.clone());

    orchestrator_first.add_agent(
        "ApprovalAgent",
        Arc::new(ApprovalAgent::new(
            "ApprovalAgent",
            "Please approve: create user account",
            json!({"user": "alice", "role": "admin"}),
        )),
    );

    orchestrator_first.add_agent(
        "MockAgent",
        Arc::new(MockAgent::new(
            "MockAgent",
            json!({"result": "user_created", "status": "success"}),
        )),
    );

    // Execute with save_state_to
    let result_first = orchestrator_first
        .execute(
            "create user account",
            tokio_util::sync::CancellationToken::new(),
            None,
            Some(&state_file_path),
        )
        .await
        .unwrap();

    // Verify execution paused
    assert!(result_first.paused, "First run should pause");
    assert_eq!(result_first.steps_executed, 0);

    // ========================================================================
    // Manual Approval: Modify saved state
    // ========================================================================

    assert!(state_file_path.exists(), "State file should be created");

    // Read the saved state
    let state_json = std::fs::read_to_string(&state_file_path).expect("Failed to read state file");
    let mut saved_state: OrchestrationState =
        serde_json::from_str(&state_json).expect("Failed to deserialize state");

    // Find step_A which should be PausedForApproval
    let step_a_state = saved_state
        .execution_manager
        .get_state("step_A")
        .expect("step_A state should exist");

    // Verify it's paused
    match step_a_state {
        StepState::PausedForApproval { message, payload } => {
            assert_eq!(message, "Please approve: create user account");
            assert_eq!(payload["user"], "alice");
            assert_eq!(payload["role"], "admin");

            // Extract payload for injection
            let approved_payload = payload.clone();

            // Change state from PausedForApproval to Completed
            // This simulates the human approving and the step completing successfully
            saved_state
                .execution_manager
                .set_state("step_A", StepState::Completed);

            // Inject the approved payload into the shared context
            // so step_B can access it via {{ step_A_output }}
            saved_state
                .context
                .insert("step_A_output".to_string(), approved_payload);
        }
        other => panic!("Expected step_A to be PausedForApproval, got {:?}", other),
    }

    // Verify step_B is still in Pending state (waiting for step_A to complete)
    let step_b_state = saved_state
        .execution_manager
        .get_state("step_B")
        .expect("step_B state should exist");
    assert!(
        matches!(step_b_state, StepState::Pending),
        "step_B should still be Pending, waiting for step_A to complete"
    );

    // Write the modified state back to the file
    let modified_state_json =
        serde_json::to_string_pretty(&saved_state).expect("Failed to serialize state");
    std::fs::write(&state_file_path, modified_state_json).expect("Failed to write state file");

    // ========================================================================
    // Second Run: Resume from modified state
    // ========================================================================

    let blueprint = BlueprintWorkflow::new("Test Blueprint".to_string());
    let mut orchestrator_second = ParallelOrchestrator::new(blueprint);
    orchestrator_second.set_strategy(strategy.clone());

    orchestrator_second.add_agent(
        "ApprovalAgent",
        Arc::new(ApprovalAgent::new(
            "ApprovalAgent",
            "Should not be called",
            json!({}),
        )),
    );

    orchestrator_second.add_agent(
        "MockAgent",
        Arc::new(MockAgent::new(
            "MockAgent",
            json!({"result": "user_created", "status": "success"}),
        )),
    );

    // Execute with resume_from
    let result_second = orchestrator_second
        .execute(
            "create user account",
            tokio_util::sync::CancellationToken::new(),
            Some(&state_file_path),
            None,
        )
        .await
        .unwrap();

    // ========================================================================
    // Final Verification
    // ========================================================================

    assert!(!result_second.paused, "Second run should not pause");
    assert!(result_second.success, "Second run should succeed");

    // Only step_B should be executed (step_A was already completed in the paused state)
    assert_eq!(
        result_second.steps_executed, 1,
        "Only step_B should be executed (step_A was already completed)"
    );
    assert_eq!(result_second.steps_skipped, 0, "No steps should be skipped");

    // Verify both outputs are present in final context
    assert!(
        result_second.context.contains_key("step_A_output"),
        "step_A output should be restored from approved state"
    );
    assert!(
        result_second.context.contains_key("step_B_output"),
        "step_B output should be present"
    );

    // Verify output values
    let step_a_output = &result_second.context["step_A_output"];
    assert_eq!(step_a_output["user"], "alice");
    assert_eq!(step_a_output["role"], "admin");

    let step_b_output = &result_second.context["step_B_output"];
    assert_eq!(step_b_output["result"], "user_created");
    assert_eq!(step_b_output["status"], "success");
}
