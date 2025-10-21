//! Integration tests for ParallelOrchestrator
//!
//! These tests verify the orchestrator's ability to execute workflows
//! with various dependency patterns in parallel.

use llm_toolkit::agent::{Agent, AgentError, DynamicAgent, Payload};
use llm_toolkit::orchestrator::{
    LoopBlock, ParallelOrchestrator, StrategyInstruction, StrategyMap, StrategyStep,
    TerminateInstruction,
};
use serde_json::{Value as JsonValue, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Mock Agents
// ============================================================================

/// Simple mock agent that returns a JSON value after a delay
#[derive(Clone)]
struct MockAgent {
    agent_name: String,
    output: JsonValue,
    delay: Duration,
    execution_log: Arc<Mutex<Vec<(String, Instant)>>>,
}

impl MockAgent {
    fn new(name: impl Into<String>, output: JsonValue) -> Self {
        Self {
            agent_name: name.into(),
            output,
            delay: Duration::from_millis(50),
            execution_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    #[allow(dead_code)]
    async fn get_execution_log(&self) -> Vec<(String, Instant)> {
        self.execution_log.lock().await.clone()
    }
}

#[async_trait::async_trait]
impl Agent for MockAgent {
    type Output = JsonValue;

    fn expertise(&self) -> &str {
        "Mock agent for testing"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        let start = Instant::now();
        self.execution_log
            .lock()
            .await
            .push((self.agent_name.clone(), start));

        tokio::time::sleep(self.delay).await;
        Ok(self.output.clone())
    }
}

#[async_trait::async_trait]
impl DynamicAgent for MockAgent {
    fn name(&self) -> String {
        self.agent_name.clone()
    }

    fn expertise(&self) -> &str {
        "Mock agent for testing"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
        self.execute(input).await
    }
}

/// Agent that always fails
#[derive(Clone)]
struct FailingAgent {
    agent_name: String,
}

impl FailingAgent {
    fn new(name: impl Into<String>) -> Self {
        Self {
            agent_name: name.into(),
        }
    }
}

#[async_trait::async_trait]
impl Agent for FailingAgent {
    type Output = JsonValue;

    fn expertise(&self) -> &str {
        "Failing agent for testing"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Err(AgentError::ExecutionFailed(format!(
            "{} failed",
            self.agent_name
        )))
    }
}

#[async_trait::async_trait]
impl DynamicAgent for FailingAgent {
    fn name(&self) -> String {
        self.agent_name.clone()
    }

    fn expertise(&self) -> &str {
        "Failing agent for testing"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
        self.execute(input).await
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

/// Test simple sequential DAG: step1 -> step2 -> step3
#[tokio::test]
async fn test_simple_sequential_dag() {
    let mut strategy = StrategyMap::new("Simple Sequential".to_string());

    // step1: no dependencies
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "First step".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Step 1 complete".to_string(),
    ));

    // step2: depends on step1
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Second step".to_string(),
        "Agent2".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Step 2 complete".to_string(),
    ));

    // step3: depends on step2
    strategy.add_step(StrategyStep::new(
        "step_3".to_string(),
        "Third step".to_string(),
        "Agent3".to_string(),
        "Process {{ step_2_output }}".to_string(),
        "Step 3 complete".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    // Register mock agents
    orchestrator.add_agent(
        "Agent1",
        Arc::new(MockAgent::new("Agent1", json!({"result": "step1"}))),
    );
    orchestrator.add_agent(
        "Agent2",
        Arc::new(MockAgent::new("Agent2", json!({"result": "step2"}))),
    );
    orchestrator.add_agent(
        "Agent3",
        Arc::new(MockAgent::new("Agent3", json!({"result": "step3"}))),
    );

    let result = orchestrator.execute("test task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 3);
    assert_eq!(result.steps_skipped, 0);
    assert!(result.context.contains_key("step_1_output"));
    assert!(result.context.contains_key("step_2_output"));
    assert!(result.context.contains_key("step_3_output"));
}

/// Test diamond DAG:
///     step1
///    /     \
///  step2   step3
///    \     /
///     step4
#[tokio::test]
async fn test_diamond_dag() {
    let mut strategy = StrategyMap::new("Diamond DAG".to_string());

    // Root step
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Root".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Root".to_string(),
    ));

    // Left branch
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Left".to_string(),
        "Agent2".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Left".to_string(),
    ));

    // Right branch
    strategy.add_step(StrategyStep::new(
        "step_3".to_string(),
        "Right".to_string(),
        "Agent3".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Right".to_string(),
    ));

    // Merge step
    strategy.add_step(StrategyStep::new(
        "step_4".to_string(),
        "Merge".to_string(),
        "Agent4".to_string(),
        "Merge {{ step_2_output }} and {{ step_3_output }}".to_string(),
        "Merge".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    let agent1 = Arc::new(MockAgent::new("Agent1", json!({"result": "root"})));
    let agent2 = Arc::new(MockAgent::new("Agent2", json!({"result": "left"})));
    let agent3 = Arc::new(MockAgent::new("Agent3", json!({"result": "right"})));
    let agent4 = Arc::new(MockAgent::new("Agent4", json!({"result": "merged"})));

    orchestrator.add_agent("Agent1", agent1);
    orchestrator.add_agent("Agent2", agent2);
    orchestrator.add_agent("Agent3", agent3);
    orchestrator.add_agent("Agent4", agent4);

    let result = orchestrator.execute("test task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 4);
    assert_eq!(result.steps_skipped, 0);
}

/// Test independent steps executing in parallel
#[tokio::test]
async fn test_independent_steps_parallel_execution() {
    let mut strategy = StrategyMap::new("Independent Steps".to_string());

    // Three independent steps
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Independent 1".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Independent 2".to_string(),
        "Agent2".to_string(),
        "Process {{ task }}".to_string(),
        "Output 2".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "step_3".to_string(),
        "Independent 3".to_string(),
        "Agent3".to_string(),
        "Process {{ task }}".to_string(),
        "Output 3".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    // Each agent takes 100ms
    let delay = Duration::from_millis(100);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(MockAgent::new("Agent1", json!({"result": "1"})).with_delay(delay)),
    );
    orchestrator.add_agent(
        "Agent2",
        Arc::new(MockAgent::new("Agent2", json!({"result": "2"})).with_delay(delay)),
    );
    orchestrator.add_agent(
        "Agent3",
        Arc::new(MockAgent::new("Agent3", json!({"result": "3"})).with_delay(delay)),
    );

    let start = Instant::now();
    let result = orchestrator.execute("test task", CancellationToken::new()).await.unwrap();
    let duration = start.elapsed();

    assert!(result.success);
    assert_eq!(result.steps_executed, 3);

    // Should complete in ~100ms (parallel) rather than ~300ms (sequential)
    assert!(
        duration < Duration::from_millis(250),
        "Expected parallel execution (~100ms) but took {:?}",
        duration
    );
}

/// Test error handling and failure propagation
#[tokio::test]
async fn test_error_handling_and_cascade() {
    let mut strategy = StrategyMap::new("Error Cascade".to_string());

    // step1: will succeed
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Success".to_string(),
        "SuccessAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Success".to_string(),
    ));

    // step2: will fail, depends on step1
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Fail".to_string(),
        "FailAgent".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Fail".to_string(),
    ));

    // step3: depends on step2, should be skipped
    strategy.add_step(StrategyStep::new(
        "step_3".to_string(),
        "Should Skip".to_string(),
        "NeverRunAgent".to_string(),
        "Process {{ step_2_output }}".to_string(),
        "Never runs".to_string(),
    ));

    // step4: independent, should succeed
    strategy.add_step(StrategyStep::new(
        "step_4".to_string(),
        "Independent".to_string(),
        "IndependentAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Independent".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    orchestrator.add_agent(
        "SuccessAgent",
        Arc::new(MockAgent::new("Success", json!({"result": "ok"}))),
    );
    orchestrator.add_agent("FailAgent", Arc::new(FailingAgent::new("Fail")));
    orchestrator.add_agent(
        "NeverRunAgent",
        Arc::new(MockAgent::new("NeverRun", json!({"result": "never"}))),
    );
    orchestrator.add_agent(
        "IndependentAgent",
        Arc::new(MockAgent::new(
            "Independent",
            json!({"result": "independent"}),
        )),
    );

    let result = orchestrator.execute("test task", CancellationToken::new()).await.unwrap();

    // Workflow should fail
    assert!(!result.success);

    // step1 and step4 should execute
    assert_eq!(result.steps_executed, 2);

    // step3 should be skipped (step2 failed)
    // Note: steps_skipped includes failed steps in current implementation
    assert!(result.steps_skipped > 0 || result.error.is_some());

    // Verify context has step1 and step4 outputs
    assert!(result.context.contains_key("step_1_output"));
    assert!(result.context.contains_key("step_4_output"));

    // step3 should not have output (it was skipped)
    assert!(!result.context.contains_key("step_3_output"));
}

/// Test with custom output keys
#[tokio::test]
async fn test_custom_output_keys() {
    let mut strategy = StrategyMap::new("Custom Keys".to_string());

    let mut step1 = StrategyStep::new(
        "step_1".to_string(),
        "First".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Output 1".to_string(),
    );
    step1.output_key = Some("custom_key".to_string());
    strategy.add_step(step1);

    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Second".to_string(),
        "Agent2".to_string(),
        "Process {{ custom_key }}".to_string(),
        "Output 2".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    orchestrator.add_agent(
        "Agent1",
        Arc::new(MockAgent::new("Agent1", json!({"result": "custom"}))),
    );
    orchestrator.add_agent(
        "Agent2",
        Arc::new(MockAgent::new("Agent2", json!({"result": "used_custom"}))),
    );

    let result = orchestrator.execute("test task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 2);

    // Verify custom key is used
    assert!(result.context.contains_key("custom_key"));
    assert_eq!(result.context["custom_key"], json!({"result": "custom"}));
}

/// Ensure the parallel executor stops before encountering a Loop instruction.
#[tokio::test]
async fn test_parallel_execution_stops_before_loop() {
    let mut strategy = StrategyMap::new("Loop Boundary".to_string());

    let step1 = StrategyStep::new(
        "step_1".to_string(),
        "First".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Output 1".to_string(),
    );

    let step2 = StrategyStep::new(
        "step_2".to_string(),
        "Second".to_string(),
        "Agent2".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Output 2".to_string(),
    );

    let loop_step = StrategyStep::new(
        "loop_step".to_string(),
        "Loop Body".to_string(),
        "LoopAgent".to_string(),
        "Process {{ step_2_output }}".to_string(),
        "Loop Output".to_string(),
    );

    strategy.add_instruction(StrategyInstruction::Step(step1.clone()));
    strategy.add_instruction(StrategyInstruction::Step(step2.clone()));
    strategy.add_instruction(StrategyInstruction::Loop(LoopBlock {
        loop_id: "loop_1".to_string(),
        description: None,
        loop_type: None,
        max_iterations: 1,
        condition_template: None,
        body: vec![StrategyInstruction::Step(loop_step.clone())],
        aggregation: None,
    }));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(MockAgent::new("Agent1", json!({"result": "s1"}))),
    );
    orchestrator.add_agent(
        "Agent2",
        Arc::new(MockAgent::new("Agent2", json!({"result": "s2"}))),
    );
    orchestrator.add_agent(
        "LoopAgent",
        Arc::new(MockAgent::new("LoopAgent", json!({"result": "loop"}))),
    );

    let result = orchestrator.execute("loop task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert!(!result.terminated);
    assert_eq!(result.steps_executed, 2);
    assert_eq!(result.steps_skipped, 0);
    assert!(result.context.contains_key("step_1_output"));
    assert!(result.context.contains_key("step_2_output"));
    assert!(!result.context.contains_key("loop_step_output"));
}

/// Verify that Terminate instructions trigger early shutdown and report skipped steps.
#[tokio::test]
async fn test_parallel_execution_with_terminate_instruction() {
    let mut strategy = StrategyMap::new("Terminate Early".to_string());

    let mut step1 = StrategyStep::new(
        "step_1".to_string(),
        "First".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Output 1".to_string(),
    );
    step1.output_key = Some("termination_flag".to_string());

    let step2 = StrategyStep::new(
        "step_2".to_string(),
        "Second".to_string(),
        "Agent2".to_string(),
        "Process {{ step_1_output }}".to_string(),
        "Output 2".to_string(),
    );

    strategy.add_instruction(StrategyInstruction::Step(step1.clone()));
    strategy.add_instruction(StrategyInstruction::Terminate(TerminateInstruction {
        terminate_id: "early_stop".to_string(),
        description: None,
        condition_template: Some("{{ termination_flag }}".to_string()),
        final_output_template: None,
    }));
    strategy.add_instruction(StrategyInstruction::Step(step2.clone()));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent("Agent1", Arc::new(MockAgent::new("Agent1", json!("true"))));
    orchestrator.add_agent(
        "Agent2",
        Arc::new(MockAgent::new("Agent2", json!({"result": "unused"}))),
    );

    let result = orchestrator.execute("terminate task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert!(result.terminated);
    assert_eq!(result.steps_executed, 1);
    assert_eq!(result.steps_skipped, 1);
    assert!(result.context.contains_key("termination_flag"));
    assert!(!result.context.contains_key("step_2_output"));
    assert!(result.termination_reason.is_none());
}

/// Test complex DAG with multiple levels
#[tokio::test]
async fn test_complex_multi_level_dag() {
    let mut strategy = StrategyMap::new("Complex DAG".to_string());

    // Level 0
    strategy.add_step(StrategyStep::new(
        "root".to_string(),
        "Root".to_string(),
        "RootAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Root".to_string(),
    ));

    // Level 1 - depends on root
    strategy.add_step(StrategyStep::new(
        "level1_a".to_string(),
        "L1A".to_string(),
        "L1A".to_string(),
        "Process {{ root_output }}".to_string(),
        "L1A".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "level1_b".to_string(),
        "L1B".to_string(),
        "L1B".to_string(),
        "Process {{ root_output }}".to_string(),
        "L1B".to_string(),
    ));

    // Level 2 - depends on level1_a
    strategy.add_step(StrategyStep::new(
        "level2_a".to_string(),
        "L2A".to_string(),
        "L2A".to_string(),
        "Process {{ level1_a_output }}".to_string(),
        "L2A".to_string(),
    ));

    // Level 2 - depends on level1_b
    strategy.add_step(StrategyStep::new(
        "level2_b".to_string(),
        "L2B".to_string(),
        "L2B".to_string(),
        "Process {{ level1_b_output }}".to_string(),
        "L2B".to_string(),
    ));

    // Final merge - depends on both level2
    strategy.add_step(StrategyStep::new(
        "final".to_string(),
        "Final".to_string(),
        "FinalAgent".to_string(),
        "Merge {{ level2_a_output }} and {{ level2_b_output }}".to_string(),
        "Final".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    for agent_name in &["RootAgent", "L1A", "L1B", "L2A", "L2B", "FinalAgent"] {
        orchestrator.add_agent(
            *agent_name,
            Arc::new(MockAgent::new(*agent_name, json!({"result": *agent_name}))),
        );
    }

    let result = orchestrator.execute("complex task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 6);
    assert!(result.context.contains_key("final_output"));
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

/// Test that agents with shared state are properly synchronized
#[tokio::test]
async fn test_thread_safe_agent_with_shared_state() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Agent with shared atomic counter
    #[derive(Clone)]
    struct CountingAgent {
        counter: Arc<AtomicUsize>,
    }

    impl CountingAgent {
        fn new(counter: Arc<AtomicUsize>) -> Self {
            Self { counter }
        }
    }

    #[async_trait::async_trait]
    impl Agent for CountingAgent {
        type Output = JsonValue;

        fn expertise(&self) -> &str {
            "Counting agent with shared state"
        }

        async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
            // Increment counter atomically
            let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(json!({"count": count}))
        }
    }

    #[async_trait::async_trait]
    impl DynamicAgent for CountingAgent {
        fn name(&self) -> String {
            "CountingAgent".to_string()
        }

        fn expertise(&self) -> &str {
            "Counting agent with shared state"
        }

        async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
            self.execute(input).await
        }
    }

    let mut strategy = StrategyMap::new("Thread Safety Test".to_string());

    // Create 10 independent steps that all use the same counting agent
    for i in 1..=10 {
        strategy.add_step(StrategyStep::new(
            format!("step_{}", i),
            format!("Step {}", i),
            "CountingAgent".to_string(),
            "Process {{ task }}".to_string(),
            format!("Output {}", i),
        ));
    }

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    let counter = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(CountingAgent::new(Arc::clone(&counter)));
    orchestrator.add_agent("CountingAgent", agent);

    let result = orchestrator.execute("count task", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 10);

    // All 10 steps should have incremented the counter exactly once
    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

/// Test that ParallelOrchestrator can be safely shared across threads
#[tokio::test]
async fn test_orchestrator_is_send_sync() {
    // This test verifies that ParallelOrchestrator implements Send + Sync
    // by spawning it in a separate tokio task

    let mut strategy = StrategyMap::new("Send/Sync Test".to_string());

    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "Process {{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(MockAgent::new("Agent1", json!({"result": "ok"}))),
    );

    // Spawn the orchestrator in a separate task to verify Send + Sync
    let handle = tokio::spawn(async move { orchestrator.execute("test task", CancellationToken::new()).await });

    let result = handle.await.unwrap().unwrap();
    assert!(result.success);
}

/// Test concurrent access to shared context
#[tokio::test]
async fn test_concurrent_context_access() {
    let mut strategy = StrategyMap::new("Concurrent Context".to_string());

    // Create a wide DAG where many steps depend on the same root
    strategy.add_step(StrategyStep::new(
        "root".to_string(),
        "Root".to_string(),
        "RootAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Root".to_string(),
    ));

    // 20 steps all depending on root, all reading from shared context
    for i in 1..=20 {
        strategy.add_step(StrategyStep::new(
            format!("child_{}", i),
            format!("Child {}", i),
            format!("ChildAgent{}", i),
            "Process {{ root_output }}".to_string(),
            format!("Child {}", i),
        ));
    }

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    orchestrator.add_agent(
        "RootAgent",
        Arc::new(MockAgent::new("Root", json!({"shared": "data"}))),
    );

    for i in 1..=20 {
        orchestrator.add_agent(
            format!("ChildAgent{}", i),
            Arc::new(MockAgent::new(format!("Child{}", i), json!({"child": i}))),
        );
    }

    let result = orchestrator.execute("concurrent test", CancellationToken::new()).await.unwrap();

    assert!(result.success);
    assert_eq!(result.steps_executed, 21); // root + 20 children

    // Verify all children executed (no race conditions in context access)
    for i in 1..=20 {
        assert!(result.context.contains_key(&format!("child_{}_output", i)));
    }
}
