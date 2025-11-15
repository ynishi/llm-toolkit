//! Integration tests for StrategyLifecycle trait
//!
//! These tests verify that both Orchestrator and ParallelOrchestrator
//! implement the StrategyLifecycle trait correctly.

use llm_toolkit::agent::{Agent, AgentError, AgentOutput, DynamicAgent, Payload};
use llm_toolkit::orchestrator::{
    BlueprintWorkflow, Orchestrator, ParallelOrchestrator, StrategyLifecycle, StrategyMap,
    StrategyStep,
};
use serde_json::{Value as JsonValue, json};
use std::sync::Arc;

// ============================================================================
// Mock Agents
// ============================================================================

#[derive(Clone)]
struct MockAgent {
    name: String,
    output: JsonValue,
}

impl MockAgent {
    fn new(name: impl Into<String>, output: JsonValue) -> Self {
        Self {
            name: name.into(),
            output,
        }
    }
}

#[async_trait::async_trait]
impl Agent for MockAgent {
    type Output = JsonValue;

    fn expertise(&self) -> &str {
        "Mock agent for lifecycle testing"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.output.clone())
    }
}

#[async_trait::async_trait]
impl DynamicAgent for MockAgent {
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

/// Mock strategy generator that returns a predefined strategy
#[derive(Clone)]
struct MockStrategyGenerator {
    strategy: StrategyMap,
}

impl MockStrategyGenerator {
    fn new(strategy: StrategyMap) -> Self {
        Self { strategy }
    }
}

#[async_trait::async_trait]
impl Agent for MockStrategyGenerator {
    type Output = StrategyMap;

    fn expertise(&self) -> &str {
        "Mock strategy generator"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.strategy.clone())
    }
}

/// Mock String agent for ParallelOrchestrator internal agent
#[derive(Clone)]
struct MockStringAgent;

impl MockStringAgent {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Agent for MockStringAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "Mock string agent"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok("FAIL".to_string())
    }
}

// ============================================================================
// Tests for StrategyLifecycle on ParallelOrchestrator
// ============================================================================

#[test]
fn test_parallel_orchestrator_set_and_get_strategy_map() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = ParallelOrchestrator::new(blueprint);

    // Initially, no strategy should be set
    assert!(orchestrator.strategy_map().is_none());

    // Create and set a strategy
    let mut strategy = StrategyMap::new("Test Strategy".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "First step".to_string(),
        "Agent1".to_string(),
        "Do something".to_string(),
        "Step 1".to_string(),
    ));

    orchestrator.set_strategy_map(strategy.clone());

    // Verify strategy is set
    let retrieved = orchestrator.strategy_map().expect("Strategy should be set");
    assert_eq!(retrieved.goal, "Test Strategy");
    assert_eq!(retrieved.steps.len(), 1);
}

#[tokio::test]
async fn test_parallel_orchestrator_generate_strategy_only() {
    // Create a predefined strategy
    let mut expected_strategy = StrategyMap::new("Generated Strategy".to_string());
    expected_strategy.add_step(StrategyStep::new(
        "gen_step_1".to_string(),
        "Generated step".to_string(),
        "GenAgent".to_string(),
        "Process {{ task }}".to_string(),
        "Gen Step 1".to_string(),
    ));

    // Create orchestrator with mock strategy generator
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let strategy_gen = Box::new(MockStrategyGenerator::new(expected_strategy.clone()));
    let string_gen = Box::new(MockStringAgent::new());

    let mut orchestrator =
        ParallelOrchestrator::with_internal_agents(blueprint, string_gen, strategy_gen);

    // Add a dummy agent so that strategy generation doesn't fail with "No agents available"
    let dummy_agent = Arc::new(MockAgent::new("GenAgent", json!({"result": "ok"})));
    orchestrator.add_agent("GenAgent", dummy_agent);

    // Generate strategy without execution
    let generated = orchestrator
        .generate_strategy_only("test task")
        .await
        .expect("Should generate strategy");

    assert_eq!(generated.goal, "Generated Strategy");
    assert_eq!(generated.steps.len(), 1);

    // Verify the strategy is also set in the orchestrator
    let stored = orchestrator
        .strategy_map()
        .expect("Strategy should be stored");
    assert_eq!(stored.goal, "Generated Strategy");
}

// ============================================================================
// Tests for StrategyLifecycle on Orchestrator
// ============================================================================

#[test]
fn test_orchestrator_set_and_get_strategy_map() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = Orchestrator::new(blueprint);

    // Initially, no strategy should be set
    assert!(orchestrator.strategy_map().is_none());

    // Create and set a strategy
    let mut strategy = StrategyMap::new("Test Strategy".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "First step".to_string(),
        "Agent1".to_string(),
        "Do something".to_string(),
        "Step 1".to_string(),
    ));

    orchestrator.set_strategy_map(strategy.clone());

    // Verify strategy is set
    let retrieved = orchestrator.strategy_map().expect("Strategy should be set");
    assert_eq!(retrieved.goal, "Test Strategy");
    assert_eq!(retrieved.steps.len(), 1);
}

#[tokio::test]
async fn test_orchestrator_generate_strategy_only() {
    // Create a predefined strategy
    let mut expected_strategy = StrategyMap::new("Generated Strategy".to_string());
    expected_strategy.add_step(StrategyStep::new(
        "gen_step_1".to_string(),
        "Generated step".to_string(),
        "GenAgent".to_string(),
        "Process task".to_string(),
        "Gen Step 1".to_string(),
    ));

    // Create orchestrator with mock strategy generator
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let string_gen = Box::new(MockStringAgent::new());
    let strategy_gen = Box::new(MockStrategyGenerator::new(expected_strategy.clone()));

    let mut orchestrator = Orchestrator::with_internal_agents(blueprint, string_gen, strategy_gen);

    // Generate strategy without execution
    let generated = orchestrator
        .generate_strategy_only("test task")
        .await
        .expect("Should generate strategy");

    assert_eq!(generated.goal, "Generated Strategy");
    assert_eq!(generated.steps.len(), 1);

    // Verify the strategy is also set in the orchestrator
    let stored = orchestrator
        .strategy_map()
        .expect("Strategy should be stored");
    assert_eq!(stored.goal, "Generated Strategy");
}

// ============================================================================
// Tests for trait polymorphism
// ============================================================================

#[tokio::test]
async fn test_strategy_lifecycle_trait_object() {
    // This test verifies that StrategyLifecycle can be used as a trait object
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator: Box<dyn StrategyLifecycle> =
        Box::new(ParallelOrchestrator::new(blueprint));

    let mut strategy = StrategyMap::new("Trait Object Test".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Test step".to_string(),
        "Agent1".to_string(),
        "Process".to_string(),
        "Step 1".to_string(),
    ));

    // Use trait methods through trait object
    orchestrator.set_strategy_map(strategy.clone());

    let retrieved = orchestrator
        .strategy_map()
        .expect("Strategy should be retrievable");
    assert_eq!(retrieved.goal, "Trait Object Test");
}

#[test]
fn test_strategy_lifecycle_consistency() {
    // Verify that set_strategy_map and strategy_map are consistent
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = ParallelOrchestrator::new(blueprint);

    let mut strategy1 = StrategyMap::new("First Strategy".to_string());
    strategy1.add_step(StrategyStep::new(
        "step_1".to_string(),
        "First".to_string(),
        "Agent1".to_string(),
        "Task 1".to_string(),
        "Step 1".to_string(),
    ));

    orchestrator.set_strategy_map(strategy1);
    assert_eq!(orchestrator.strategy_map().unwrap().goal, "First Strategy");

    // Replace with second strategy
    let mut strategy2 = StrategyMap::new("Second Strategy".to_string());
    strategy2.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Second".to_string(),
        "Agent2".to_string(),
        "Task 2".to_string(),
        "Step 2".to_string(),
    ));

    orchestrator.set_strategy_map(strategy2);
    assert_eq!(orchestrator.strategy_map().unwrap().goal, "Second Strategy");
}
