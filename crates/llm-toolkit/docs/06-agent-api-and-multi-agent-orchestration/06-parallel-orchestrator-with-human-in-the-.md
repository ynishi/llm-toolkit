### Parallel Orchestrator with Human-in-the-Loop (HIL)


The `ParallelOrchestrator` supports a Human-in-the-Loop (HIL) capability, allowing agents to pause execution and explicitly request human approval before proceeding with critical, ambiguous, or safety-sensitive tasks. This feature transforms the orchestrator from a purely automated workflow engine into a collaborative partner that can safely wait for human guidance at key decision points.

#### Overview

The `ParallelOrchestrator` executes workflows based on dependency graphs, running independent steps concurrently in "waves" to maximize performance. The HIL feature builds upon the orchestrator's existing interrupt and resume (save/load state) functionality to provide a robust, auditable approval workflow.

Human-in-the-Loop is essential for scenarios where:
- **Safety-critical operations** require explicit confirmation (e.g., deploying to production, deleting data)
- **Ambiguous decisions** need human judgment (e.g., selecting the best approach from multiple options)
- **Compliance requirements** mandate human oversight for certain actions
- **Trust boundaries** exist between automated and manual processes

#### HIL Workflow

The Human-in-the-Loop workflow follows these steps:

1. **Agent Requests Approval**: An agent reaches a point requiring human input and returns `AgentOutput::RequiresApproval` instead of a standard result.

2. **Orchestrator Pauses**: The orchestrator receives the approval request, transitions the corresponding step into a `PausedForApproval` state, and gracefully stops execution.

3. **State Persistence**: Before stopping, the orchestrator automatically saves the complete `OrchestrationState` (including the paused step, approval message, and context) to a file using the existing `save_state_to` mechanism.

4. **Human Review**: The application notifies the user that approval is needed. The user inspects the saved state file, which contains:
   - The approval message explaining what needs review
   - The current payload/context from the agent
   - The complete workflow state

5. **Approval & State Modification**: To approve, the user (or an external tool) modifies the saved state file:
   - Changes the step's status from `PausedForApproval` to `Completed`
   - Optionally injects approved data into the shared context for downstream steps

6. **Orchestrator Resumes**: The application re-invokes the orchestrator using the `resume_from` parameter, pointing to the modified state file. The orchestrator loads the state and seamlessly continues execution from the now-approved step.

#### Implementing an Agent with Approval Requests

To enable an agent to request approval, implement the `DynamicAgent` trait and return `AgentOutput::RequiresApproval`:

```rust
use llm_toolkit::agent::{Agent, AgentError, AgentOutput, DynamicAgent, Payload};
use serde_json::{json, Value as JsonValue};

#[derive(Clone)]
struct DeploymentAgent;

#[async_trait::async_trait]
impl Agent for DeploymentAgent {
    type Output = JsonValue;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Handles production deployments with human approval";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        unreachable!("DeploymentAgent uses execute_dynamic")
    }
}

#[async_trait::async_trait]
impl DynamicAgent for DeploymentAgent {
    fn name(&self) -> String {
        "DeploymentAgent".to_string()
    }

    fn description(&self) -> &str {
        "Handles production deployments with human approval"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<AgentOutput, AgentError> {
        // Prepare deployment plan
        let deployment_plan = json!({
            "target": "production",
            "service": "user-api",
            "version": "v2.1.0",
            "estimated_downtime": "30 seconds"
        });

        // Request human approval before proceeding
        Ok(AgentOutput::RequiresApproval {
            message_for_human: "Please review and approve deployment to production: user-api v2.1.0".to_string(),
            current_payload: deployment_plan,
        })
    }
}
```

#### Using HIL in Workflows

Here's a complete example showing how to handle the pause-approve-resume cycle:

```rust
use llm_toolkit::orchestrator::{
    ParallelOrchestrator, StrategyMap, StrategyStep, OrchestrationState,
    parallel::StepState
};
use std::sync::Arc;
use std::path::Path;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    // Define workflow with approval step
    let mut strategy = StrategyMap::new("Production Deployment");

    strategy.add_step(StrategyStep::new(
        "validate_changes",
        "Validate code changes",
        "ValidationAgent",
        "Validate changes for {{ service }}",
        "validation_result",
    ));

    strategy.add_step(StrategyStep::new(
        "deploy",
        "Deploy to production",
        "DeploymentAgent",
        "Deploy {{ service }} with validation: {{ validation_result }}",
        "deployment_result",
    ));

    // Create orchestrator and register agents
    let mut orchestrator = ParallelOrchestrator::new(strategy.clone());
    orchestrator.add_agent("ValidationAgent", Arc::new(ValidationAgent));
    orchestrator.add_agent("DeploymentAgent", Arc::new(DeploymentAgent));

    let state_file = Path::new("/tmp/deployment_state.json");

    // First execution: run until pause
    let result = orchestrator
        .execute(
            "Deploy user-api service",
            CancellationToken::new(),
            None,                    // No resume (fresh start)
            Some(state_file),        // Save state on pause
        )
        .await
        .unwrap();

    if result.paused {
        println!("Workflow paused for approval:");
        println!("Reason: {}", result.pause_reason.unwrap());
        println!("State saved to: {:?}", state_file);

        // ================================================================
        // Human intervention: Review and approve
        // ================================================================

        // Read the saved state
        let state_json = std::fs::read_to_string(state_file)
            .expect("Failed to read state file");
        let mut saved_state: OrchestrationState =
            serde_json::from_str(&state_json)
                .expect("Failed to deserialize state");

        // Find the paused step
        let step_state = saved_state
            .execution_manager
            .get_state("deploy")
            .expect("Deploy step not found");

        // Inspect the approval request
        if let StepState::PausedForApproval { message, payload } = step_state {
            println!("Approval message: {}", message);
            println!("Deployment plan: {}", serde_json::to_string_pretty(&payload).unwrap());

            // User reviews and approves...
            // Modify the state: mark step as completed
            saved_state
                .execution_manager
                .set_state("deploy", StepState::Completed);

            // Inject approved deployment result into context
            saved_state.context.insert(
                "deployment_result".to_string(),
                json!({
                    "status": "approved_and_deployed",
                    "approved_by": "user@example.com",
                    "timestamp": "2024-01-15T10:30:00Z"
                })
            );
        }

        // Write modified state back
        let modified_json = serde_json::to_string_pretty(&saved_state)
            .expect("Failed to serialize state");
        std::fs::write(state_file, modified_json)
            .expect("Failed to write state");

        println!("Approval granted. Resuming workflow...");

        // ================================================================
        // Resume execution with approved state
        // ================================================================

        let mut orchestrator_resumed = ParallelOrchestrator::new(strategy);
        orchestrator_resumed.add_agent("ValidationAgent", Arc::new(ValidationAgent));
        orchestrator_resumed.add_agent("DeploymentAgent", Arc::new(DeploymentAgent));

        let final_result = orchestrator_resumed
            .execute(
                "Deploy user-api service",
                CancellationToken::new(),
                Some(state_file),    // Resume from modified state
                None,                // No need to save again
            )
            .await
            .unwrap();

        assert!(final_result.success, "Workflow should complete successfully");
        assert!(!final_result.paused, "Workflow should not pause again");

        println!("Deployment completed successfully!");
        println!("Final result: {:?}", final_result.context.get("deployment_result"));
    }
}
```

#### Key Features

- **Explicit Approval Contract**: Agents use `AgentOutput::RequiresApproval` to clearly signal when human input is needed.
- **State Transparency**: The saved state file contains all information needed for the user to make an informed decision.
- **Flexible Approval Process**: Users can approve by simply editing the JSON state file, or build custom approval workflows (web UIs, CLI tools, etc.) that modify the state programmatically.
- **Seamless Resumption**: The orchestrator resumes exactly where it left off, with no duplicate work or lost context.
- **Audit Trail**: The state file serves as a complete record of what was requested, what was approved, and when.

#### Return Values

When an agent requests approval, the orchestrator returns a `ParallelOrchestrationResult` with:
- `paused = true`: Indicates execution was paused
- `success = true`: The pause is intentional and successful, not an error
- `pause_reason = Some(message)`: Contains the approval message from the agent
- `steps_executed = 0` (typically): No steps complete when pausing for approval
- The state is saved to the file specified in `save_state_to`

After resuming with an approved state:
- `paused = false`: Execution completed normally
- `success = true`: Workflow completed successfully
- `steps_executed`: Count of steps executed during resume (excludes already-completed steps)
- `context`: Contains all outputs, including injected approval data

