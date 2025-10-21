//! Timeout tests for ParallelOrchestrator
//!
//! These tests verify that per-step timeout enforcement works correctly
//! and that timed-out steps properly propagate to dependents.

use llm_toolkit::agent::{Agent, AgentError, DynamicAgent, Payload};
use llm_toolkit::orchestrator::parallel::ParallelOrchestratorConfig;
use llm_toolkit::orchestrator::{ParallelOrchestrator, StrategyMap, StrategyStep};
use serde_json::{Value as JsonValue, json};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Mock Agents
// ============================================================================

#[derive(Clone)]
struct FastAgent {
    name: String,
    output: JsonValue,
}

impl FastAgent {
    fn new(name: impl Into<String>, output: JsonValue) -> Self {
        Self {
            name: name.into(),
            output,
        }
    }
}

#[async_trait::async_trait]
impl Agent for FastAgent {
    type Output = JsonValue;

    fn expertise(&self) -> &str {
        "Fast test agent"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.output.clone())
    }
}

#[async_trait::async_trait]
impl DynamicAgent for FastAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn expertise(&self) -> &str {
        "Fast test agent"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
        self.execute(input).await
    }
}

#[derive(Clone)]
struct SlowAgent {
    name: String,
    sleep_duration: Duration,
}

impl SlowAgent {
    fn new(name: impl Into<String>, sleep_duration: Duration) -> Self {
        Self {
            name: name.into(),
            sleep_duration,
        }
    }
}

#[async_trait::async_trait]
impl Agent for SlowAgent {
    type Output = JsonValue;

    fn expertise(&self) -> &str {
        "Slow test agent"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        tokio::time::sleep(self.sleep_duration).await;
        Ok(json!({"result": "slow"}))
    }
}

#[async_trait::async_trait]
impl DynamicAgent for SlowAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn expertise(&self) -> &str {
        "Slow test agent"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
        self.execute(input).await
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_config_default_has_no_timeout() {
    let config = ParallelOrchestratorConfig::default();
    assert!(config.step_timeout.is_none());
}

#[test]
fn test_config_with_step_timeout() {
    let config =
        ParallelOrchestratorConfig::default().with_step_timeout(Duration::from_millis(100));

    assert!(config.step_timeout.is_some());
    assert_eq!(config.step_timeout.unwrap(), Duration::from_millis(100));
}

#[test]
fn test_config_builder_chaining() {
    let config = ParallelOrchestratorConfig::default().with_step_timeout(Duration::from_secs(5));

    assert_eq!(config.step_timeout, Some(Duration::from_secs(5)));
}

// ============================================================================
// Timeout Enforcement Tests
// ============================================================================

#[tokio::test]
async fn test_step_timeout_enforcement() {
    // Create strategy with one slow step
    let mut strategy = StrategyMap::new("Timeout Test".to_string());
    strategy.add_step(StrategyStep::new(
        "slow_step".to_string(),
        "Slow Step".to_string(),
        "SlowAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    // Create orchestrator with 100ms timeout
    let config =
        ParallelOrchestratorConfig::default().with_step_timeout(Duration::from_millis(100));

    let mut orchestrator = ParallelOrchestrator::with_config(strategy, config);

    // Add agent that sleeps for 2 seconds (will timeout)
    orchestrator.add_agent(
        "SlowAgent",
        Arc::new(SlowAgent::new("SlowAgent", Duration::from_secs(2))),
    );

    let result = orchestrator
        .execute("timeout test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    // Should fail due to timeout
    assert!(!result.success, "Workflow should fail due to timeout");
    assert!(result.error.is_some(), "Error should be present");
    let error_msg = result.error.as_ref().unwrap();
    assert!(
        error_msg.contains("timed out"),
        "Error message should mention timeout"
    );
    assert_eq!(result.steps_executed, 0, "No steps should complete");
}

#[tokio::test]
async fn test_timeout_propagates_to_dependents() {
    // Create strategy: slow_step (times out) -> dependent_step (should be skipped)
    let mut strategy = StrategyMap::new("Timeout Propagation Test".to_string());

    strategy.add_step(StrategyStep::new(
        "slow_step".to_string(),
        "Slow Step".to_string(),
        "SlowAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "dependent_step".to_string(),
        "Dependent Step".to_string(),
        "FastAgent".to_string(),
        "{{ slow_step_output }}".to_string(),
        "Output 2".to_string(),
    ));

    // Create orchestrator with 100ms timeout
    let config =
        ParallelOrchestratorConfig::default().with_step_timeout(Duration::from_millis(100));

    let mut orchestrator = ParallelOrchestrator::with_config(strategy, config);

    orchestrator.add_agent(
        "SlowAgent",
        Arc::new(SlowAgent::new("SlowAgent", Duration::from_secs(2))),
    );
    orchestrator.add_agent(
        "FastAgent",
        Arc::new(FastAgent::new("FastAgent", json!({"ok": true}))),
    );

    let result = orchestrator
        .execute("propagation test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    // Should fail due to timeout
    assert!(!result.success, "Workflow should fail");
    assert_eq!(
        result.steps_executed, 0,
        "Slow step should timeout before completing"
    );
    assert!(result.steps_skipped > 0, "Dependent step should be skipped");
}

#[tokio::test]
async fn test_no_timeout_when_step_completes_quickly() {
    // Create strategy with one fast step
    let mut strategy = StrategyMap::new("No Timeout Test".to_string());
    strategy.add_step(StrategyStep::new(
        "fast_step".to_string(),
        "Fast Step".to_string(),
        "FastAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    // Create orchestrator with 1 second timeout (plenty of time)
    let config = ParallelOrchestratorConfig::default().with_step_timeout(Duration::from_secs(1));

    let mut orchestrator = ParallelOrchestrator::with_config(strategy, config);

    orchestrator.add_agent(
        "FastAgent",
        Arc::new(FastAgent::new("FastAgent", json!({"ok": true}))),
    );

    let result = orchestrator
        .execute("no timeout test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    // Should succeed
    assert!(result.success, "Workflow should succeed");
    assert_eq!(result.steps_executed, 1, "Step should complete");
    assert_eq!(result.steps_skipped, 0, "No steps should be skipped");
}

#[tokio::test]
async fn test_timeout_with_multiple_independent_steps() {
    // Create strategy with 3 independent steps, one slow
    let mut strategy = StrategyMap::new("Multiple Steps Timeout Test".to_string());

    strategy.add_step(StrategyStep::new(
        "fast_step_1".to_string(),
        "Fast Step 1".to_string(),
        "FastAgent1".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "slow_step".to_string(),
        "Slow Step".to_string(),
        "SlowAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 2".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "fast_step_2".to_string(),
        "Fast Step 2".to_string(),
        "FastAgent2".to_string(),
        "{{ task }}".to_string(),
        "Output 3".to_string(),
    ));

    // Create orchestrator with 100ms timeout
    let config =
        ParallelOrchestratorConfig::default().with_step_timeout(Duration::from_millis(100));

    let mut orchestrator = ParallelOrchestrator::with_config(strategy, config);

    orchestrator.add_agent(
        "FastAgent1",
        Arc::new(FastAgent::new("FastAgent1", json!({"ok": 1}))),
    );
    orchestrator.add_agent(
        "SlowAgent",
        Arc::new(SlowAgent::new("SlowAgent", Duration::from_secs(2))),
    );
    orchestrator.add_agent(
        "FastAgent2",
        Arc::new(FastAgent::new("FastAgent2", json!({"ok": 2}))),
    );

    let result = orchestrator
        .execute("multiple steps test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    // Should fail overall, but fast steps should complete
    assert!(!result.success, "Workflow should fail due to one timeout");
    assert_eq!(result.steps_executed, 2, "Two fast steps should complete");
}

#[tokio::test]
async fn test_no_timeout_when_config_has_none() {
    // Create strategy with slow step
    let mut strategy = StrategyMap::new("No Config Timeout Test".to_string());
    strategy.add_step(StrategyStep::new(
        "slow_step".to_string(),
        "Slow Step".to_string(),
        "SlowAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    // Create orchestrator with NO timeout configured
    let config = ParallelOrchestratorConfig::default();
    assert!(config.step_timeout.is_none());

    let mut orchestrator = ParallelOrchestrator::with_config(strategy, config);

    // Add agent that sleeps for 200ms
    orchestrator.add_agent(
        "SlowAgent",
        Arc::new(SlowAgent::new("SlowAgent", Duration::from_millis(200))),
    );

    let result = orchestrator
        .execute(
            "no config timeout test",
            CancellationToken::new(),
            None,
            None,
        )
        .await
        .unwrap();

    // Should succeed because no timeout is configured
    assert!(
        result.success,
        "Workflow should succeed without timeout config"
    );
    assert_eq!(result.steps_executed, 1, "Step should complete");
}

// ============================================================================
// Cancellation Tests
// ============================================================================

#[tokio::test]
async fn test_workflow_cancellation() {
    // Create strategy: step1 -> step2
    let mut strategy = StrategyMap::new("Cancellation Test".to_string());

    strategy.add_step(StrategyStep::new(
        "step1".to_string(),
        "Step 1".to_string(),
        "SlowAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "step2".to_string(),
        "Step 2".to_string(),
        "FastAgent".to_string(),
        "{{ step1_output }}".to_string(),
        "Output 2".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);

    // Add agents
    orchestrator.add_agent(
        "SlowAgent",
        Arc::new(SlowAgent::new("SlowAgent", Duration::from_millis(200))),
    );
    orchestrator.add_agent(
        "FastAgent",
        Arc::new(FastAgent::new("FastAgent", json!({"ok": true}))),
    );

    // Create cancellation token
    let cancellation_token = CancellationToken::new();
    let cancel_token_clone = cancellation_token.clone();

    // Spawn orchestrator execution
    let execution_handle = tokio::spawn(async move {
        orchestrator
            .execute("cancellation test", cancel_token_clone, None, None)
            .await
    });

    // Wait a bit to ensure step1 starts
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Trigger cancellation
    cancellation_token.cancel();

    // Await the result
    let result = execution_handle.await.unwrap().unwrap();

    // Verify cancellation behavior
    assert!(!result.success, "Workflow should fail due to cancellation");
    assert!(result.error.is_some(), "Error should be present");

    let error_msg = result.error.as_ref().unwrap();
    assert!(
        error_msg.contains("failed") || error_msg.contains("skipped"),
        "Error message should indicate failure, got: {}",
        error_msg
    );

    // The cancellation should cause step1 to be interrupted
    // Since step1 is slow (200ms) and we cancel after 50ms, it should be cancelled
    // before completion, resulting in 0 executed steps and 2 skipped steps
    assert!(
        result.steps_executed <= 1,
        "At most 1 step should execute before cancellation"
    );
    assert!(
        result.steps_skipped >= 1,
        "At least 1 step should be skipped due to cancellation"
    );
}
