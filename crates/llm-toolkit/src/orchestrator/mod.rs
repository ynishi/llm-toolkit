//! Orchestrator - Agent swarm coordination for complex LLM workflows.
//!
//! This module provides a policy-driven orchestrator that coordinates multiple agents
//! to accomplish complex tasks. Unlike traditional workflow engines with rigid type systems,
//! the orchestrator uses natural language blueprints and LLM-based strategy generation
//! for maximum flexibility.
//!
//! # Design Philosophy
//!
//! The orchestrator avoids "photorealistic complexity" - the temptation to model
//! agent interactions with strict Actor-like message passing and complex state management.
//! Instead, it leverages LLM flexibility to:
//!
//! - Generate execution strategies ad-hoc from blueprints
//! - Adapt to errors with tactical or full redesign
//! - Inject context naturally into agent intents
//!
//! # Example
//!
//! ```rust,ignore
//! use llm_toolkit::orchestrator::{Orchestrator, BlueprintWorkflow};
//! use llm_toolkit::agent::ClaudeCodeAgent;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let blueprint = BlueprintWorkflow::new(r#"
//!         Technical Article Workflow:
//!         1. Analyze topic and create outline
//!         2. Research each section
//!         3. Write full article
//!         4. Generate title and summary
//!     "#.to_string());
//!
//!     let mut orchestrator = Orchestrator::new(blueprint);
//!     orchestrator.add_agent(Box::new(ClaudeCodeAgent::new()));
//!
//!     let result = orchestrator.execute(
//!         "Write an article about Rust async programming"
//!     ).await?;
//!
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```

pub mod blueprint;
pub mod error;
pub mod strategy;

pub use blueprint::BlueprintWorkflow;
pub use error::OrchestratorError;
pub use strategy::{RedesignStrategy, StrategyMap, StrategyStep};

use crate::agent::Agent;
use std::collections::HashMap;

/// The orchestrator coordinates multiple agents to execute complex workflows.
///
/// The orchestrator maintains:
/// - A `BlueprintWorkflow` describing the intended process
/// - A registry of available agents
/// - Dynamically generated execution strategies
/// - Runtime context for inter-agent communication
pub struct Orchestrator {
    /// The workflow blueprint (reference material for strategy generation).
    blueprint: BlueprintWorkflow,

    /// Available agents, keyed by their name.
    agents: HashMap<String, Box<dyn Agent<Output = String>>>,

    /// The currently active execution strategy.
    strategy_map: Option<StrategyMap>,

    /// Runtime context storing intermediate results.
    context: HashMap<String, String>,
}

impl Orchestrator {
    /// Creates a new Orchestrator with a given blueprint.
    pub fn new(blueprint: BlueprintWorkflow) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            strategy_map: None,
            context: HashMap::new(),
        }
    }

    /// Adds an agent to the orchestrator's registry.
    ///
    /// Note: Currently only supports agents with `Output = String`.
    /// Future versions may support heterogeneous agent types.
    pub fn add_agent(&mut self, agent: Box<dyn Agent<Output = String>>) {
        let name = agent.name();
        self.agents.insert(name, agent);
    }

    /// Returns a reference to an agent by name.
    pub fn get_agent(&self, name: &str) -> Option<&dyn Agent<Output = String>> {
        self.agents.get(name).map(|boxed| &**boxed)
    }

    /// Returns a list of all available agent names.
    pub fn list_agents(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Returns a formatted string describing all available agents and their expertise.
    pub fn format_agent_list(&self) -> String {
        self.agents
            .iter()
            .map(|(name, agent)| format!("- {}: {}", name, agent.expertise()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Executes the workflow with the given task description.
    ///
    /// This is the main entry point for orchestration. The orchestrator will:
    /// 1. Generate a strategy map from the blueprint and available agents
    /// 2. Execute each step in sequence
    /// 3. Handle errors with adaptive redesign strategies
    /// 4. Return the final result
    ///
    /// # Arguments
    ///
    /// * `task` - A natural language description of what needs to be accomplished
    ///
    /// # Returns
    ///
    /// The final output as a string. Future versions may support typed outputs.
    pub async fn execute(&mut self, task: &str) -> Result<String, OrchestratorError> {
        log::info!("Starting orchestrator execution for task: {}", task);

        // Phase 1: Generate strategy (stub for now)
        self.generate_strategy(task).await?;

        // Phase 2: Execute strategy (stub for now)
        let result = self.execute_strategy().await?;

        log::info!("Orchestrator execution completed");
        Ok(result)
    }

    /// Generates an execution strategy from the blueprint, agents, and task.
    ///
    /// This is a stub implementation. Future versions will use LLM calls
    /// to generate strategies dynamically.
    async fn generate_strategy(&mut self, task: &str) -> Result<(), OrchestratorError> {
        log::debug!("Generating strategy for task: {}", task);

        // Stub: Create a simple single-step strategy
        let mut strategy = StrategyMap::new(task.to_string());

        // If we have agents, assign the first one
        if let Some((agent_name, _agent)) = self.agents.iter().next() {
            let step = StrategyStep::new(
                "step_1".to_string(),
                "Execute the task".to_string(),
                agent_name.clone(),
                task.to_string(),
                "Result of the task".to_string(),
            );
            strategy.add_step(step);
        } else {
            return Err(OrchestratorError::StrategyGenerationFailed(
                "No agents available".to_string(),
            ));
        }

        self.strategy_map = Some(strategy);
        Ok(())
    }

    /// Executes the current strategy step by step.
    ///
    /// This is a stub implementation. Future versions will include:
    /// - Context injection
    /// - Error handling and redesign
    /// - Progress tracking
    async fn execute_strategy(&mut self) -> Result<String, OrchestratorError> {
        let strategy = self
            .strategy_map
            .as_ref()
            .ok_or_else(OrchestratorError::no_strategy)?;

        let mut final_result = String::new();

        for (i, step) in strategy.steps.iter().enumerate() {
            log::info!("Executing step {}: {}", i + 1, step.description);

            let agent = self
                .agents
                .get(&step.assigned_agent)
                .ok_or_else(|| OrchestratorError::AgentNotFound(step.assigned_agent.clone()))?;

            // Build intent (stub - just use template as-is)
            let intent = step.intent_template.clone();

            // Execute agent
            let output = agent.execute(intent).await?;

            // Store result in context
            self.context
                .insert(format!("step_{}_output", step.step_id), output.clone());

            final_result = output;
        }

        Ok(final_result)
    }

    /// Clears the current strategy and context (useful for re-execution).
    pub fn reset(&mut self) {
        self.strategy_map = None;
        self.context.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::impls::ClaudeCodeAgent;

    #[test]
    fn test_orchestrator_creation() {
        let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
        let orch = Orchestrator::new(blueprint);
        assert_eq!(orch.list_agents().len(), 0);
    }

    #[test]
    fn test_add_agent() {
        let blueprint = BlueprintWorkflow::new("Test".to_string());
        let mut orch = Orchestrator::new(blueprint);

        orch.add_agent(Box::new(ClaudeCodeAgent::new()));
        assert_eq!(orch.list_agents().len(), 1);
        assert!(orch.get_agent("ClaudeCodeAgent").is_some());
    }

    #[test]
    fn test_format_agent_list() {
        let blueprint = BlueprintWorkflow::new("Test".to_string());
        let mut orch = Orchestrator::new(blueprint);

        orch.add_agent(Box::new(ClaudeCodeAgent::new()));
        let list = orch.format_agent_list();
        assert!(list.contains("ClaudeCodeAgent"));
    }
}
