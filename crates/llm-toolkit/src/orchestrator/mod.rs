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

// Prompt definitions require both derive (ToPrompt macro) and agent (for usage)
#[cfg(all(feature = "derive", feature = "agent"))]
pub mod prompts;

pub use blueprint::BlueprintWorkflow;
pub use error::OrchestratorError;
pub use strategy::{RedesignStrategy, StrategyMap, StrategyStep};

use crate::agent::Agent;
use std::collections::HashMap;

#[cfg(feature = "agent")]
use crate::agent::impls::{ClaudeCodeAgent, ClaudeCodeJsonAgent};

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

    /// Internal JSON agent for structured strategy generation.
    #[cfg(feature = "agent")]
    internal_json_agent: ClaudeCodeJsonAgent<StrategyMap>,

    /// Internal string agent for intent generation and redesign decisions.
    #[cfg(feature = "agent")]
    internal_agent: ClaudeCodeAgent,

    /// The currently active execution strategy.
    strategy_map: Option<StrategyMap>,

    /// Runtime context storing intermediate results.
    context: HashMap<String, String>,

    /// The original task description (stored for regeneration).
    current_task: Option<String>,
}

impl Orchestrator {
    /// Creates a new Orchestrator with a given blueprint.
    #[cfg(feature = "agent")]
    pub fn new(blueprint: BlueprintWorkflow) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            internal_json_agent: ClaudeCodeJsonAgent::new(),
            internal_agent: ClaudeCodeAgent::new(),
            strategy_map: None,
            context: HashMap::new(),
            current_task: None,
        }
    }

    /// Creates a new Orchestrator without the internal agent (for testing).
    #[cfg(not(feature = "agent"))]
    pub fn new(blueprint: BlueprintWorkflow) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            strategy_map: None,
            context: HashMap::new(),
            current_task: None,
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

        // Store task for potential regeneration
        self.current_task = Some(task.to_string());

        // Phase 1: Generate strategy
        self.generate_strategy(task).await?;

        // Phase 2: Execute strategy
        let result = self.execute_strategy().await?;

        log::info!("Orchestrator execution completed");
        Ok(result)
    }

    /// Generates an execution strategy from the blueprint, agents, and task.
    ///
    /// Uses the internal LLM agent to analyze the task, available agents,
    /// and blueprint to generate an optimal execution strategy.
    #[cfg(all(feature = "agent", feature = "derive"))]
    async fn generate_strategy(&mut self, task: &str) -> Result<(), OrchestratorError> {
        use crate::prompt::ToPrompt;
        use prompts::StrategyGenerationRequest;

        log::debug!("Generating strategy for task: {}", task);

        if self.agents.is_empty() {
            return Err(OrchestratorError::StrategyGenerationFailed(
                "No agents available".to_string(),
            ));
        }

        // Build the prompt using llm-toolkit's ToPrompt
        let request = StrategyGenerationRequest::new(
            task.to_string(),
            self.format_agent_list(),
            self.blueprint.description.clone(),
            self.blueprint.graph.clone(),
        );

        let prompt = request.to_prompt();

        log::debug!("Strategy generation prompt:\n{}", prompt);

        // Call internal JSON agent to generate strategy
        let strategy_map = self
            .internal_json_agent
            .execute(prompt)
            .await
            .map_err(|e| OrchestratorError::StrategyGenerationFailed(e.to_string()))?;

        log::info!("Generated strategy with {} steps", strategy_map.steps.len());

        self.strategy_map = Some(strategy_map);
        Ok(())
    }

    /// Generates an execution strategy (stub for non-derive feature).
    #[cfg(not(all(feature = "agent", feature = "derive")))]
    async fn generate_strategy(&mut self, task: &str) -> Result<(), OrchestratorError> {
        log::debug!("Generating strategy for task (stub mode): {}", task);

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

    /// Collects context information for a given step.
    ///
    /// Gathers previous step outputs and other relevant context data.
    fn collect_context(&self, step_index: usize) -> HashMap<String, String> {
        let mut context = HashMap::new();

        // Add all previous step outputs
        if let Some(strategy) = &self.strategy_map {
            for i in 0..step_index {
                if i < strategy.steps.len() {
                    let prev_step = &strategy.steps[i];
                    let key = format!("step_{}_output", prev_step.step_id);
                    if let Some(output) = self.context.get(&key) {
                        context.insert(key, output.clone());
                        // Also provide as "previous_output" for convenience
                        if i == step_index - 1 {
                            context.insert("previous_output".to_string(), output.clone());
                        }
                    }
                }
            }
        }

        // Add any other context from self.context
        for (key, value) in &self.context {
            if !key.starts_with("step_") {
                context.insert(key.clone(), value.clone());
            }
        }

        context
    }

    /// Formats context as a readable string for prompts.
    fn format_context(&self, context: &HashMap<String, String>) -> String {
        if context.is_empty() {
            return "No context available yet.".to_string();
        }

        context
            .iter()
            .map(|(key, value)| {
                // Truncate long values for readability
                let display_value = if value.len() > 200 {
                    format!("{}... (truncated)", &value[..200])
                } else {
                    value.clone()
                };
                format!("- {}: {}", key, display_value)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Formats completed steps for redesign prompts.
    fn format_completed_steps(&self, up_to_index: usize) -> String {
        if let Some(strategy) = &self.strategy_map {
            strategy
                .steps
                .iter()
                .take(up_to_index)
                .enumerate()
                .map(|(i, step)| {
                    let output = self
                        .context
                        .get(&format!("step_{}_output", step.step_id))
                        .cloned()
                        .unwrap_or_else(|| "(no output)".to_string());

                    format!(
                        "Step {}: {}\n  Agent: {}\n  Output: {}\n",
                        i + 1,
                        step.description,
                        step.assigned_agent,
                        if output.len() > 100 {
                            format!("{}...", &output[..100])
                        } else {
                            output
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            "No completed steps".to_string()
        }
    }

    /// Builds an optimized intent prompt using LLM.
    ///
    /// Takes the intent template and current context, and generates a high-quality
    /// prompt specifically tailored for the assigned agent.
    #[cfg(all(feature = "agent", feature = "derive"))]
    async fn build_intent(
        &self,
        step: &StrategyStep,
        context: &HashMap<String, String>,
    ) -> Result<String, OrchestratorError> {
        use crate::prompt::ToPrompt;
        use prompts::IntentGenerationRequest;

        // Get agent expertise
        let agent = self
            .agents
            .get(&step.assigned_agent)
            .ok_or_else(|| OrchestratorError::AgentNotFound(step.assigned_agent.clone()))?;

        let request = IntentGenerationRequest::new(
            step.description.clone(),
            step.expected_output.clone(),
            agent.expertise().to_string(),
            step.intent_template.clone(),
            self.format_context(context),
        );

        let prompt = request.to_prompt();

        log::debug!("Generating intent for step: {}", step.step_id);

        // Use internal agent to generate the intent
        let intent = self.internal_agent.execute(prompt).await?;

        Ok(intent)
    }

    /// Builds intent (stub for non-derive feature).
    #[cfg(not(all(feature = "agent", feature = "derive")))]
    async fn build_intent(
        &self,
        step: &StrategyStep,
        context: &HashMap<String, String>,
    ) -> Result<String, OrchestratorError> {
        // Simple template substitution as fallback
        let mut intent = step.intent_template.clone();

        for (key, value) in context {
            let placeholder = format!("{{{}}}", key);
            intent = intent.replace(&placeholder, value);
        }

        Ok(intent)
    }

    /// Executes the current strategy step by step.
    ///
    /// Includes context injection, intent generation, and error handling with redesign.
    async fn execute_strategy(&mut self) -> Result<String, OrchestratorError> {
        // Check strategy exists
        if self.strategy_map.is_none() {
            return Err(OrchestratorError::no_strategy());
        }

        let mut final_result = String::new();
        let mut step_index = 0;

        loop {
            // Get current strategy info
            let (step_count, goal, step) = {
                let strategy = self.strategy_map.as_ref().unwrap();
                if step_index >= strategy.steps.len() {
                    break; // All steps completed
                }
                (
                    strategy.steps.len(),
                    strategy.goal.clone(),
                    strategy.steps[step_index].clone(),
                )
            };

            log::info!(
                "Executing step {}/{}: {}",
                step_index + 1,
                step_count,
                step.description
            );

            // Collect context
            let context = self.collect_context(step_index);

            // Build intent using LLM
            let intent = self.build_intent(&step, &context).await?;

            log::debug!("Generated intent:\n{}", intent);

            // Execute agent
            let agent = self
                .agents
                .get(&step.assigned_agent)
                .ok_or_else(|| OrchestratorError::AgentNotFound(step.assigned_agent.clone()))?;

            match agent.execute(intent).await {
                Ok(output) => {
                    log::info!("Step {} completed successfully", step_index + 1);

                    // Store result in context
                    self.context
                        .insert(format!("step_{}_output", step.step_id), output.clone());

                    final_result = output;
                    step_index += 1;
                }
                Err(e) => {
                    log::warn!("Step {} failed: {}", step_index + 1, e);

                    // Determine redesign strategy
                    match self
                        .determine_redesign_strategy(&e, step_index, &goal)
                        .await?
                    {
                        RedesignStrategy::Retry => {
                            log::info!("Retrying step {}", step_index + 1);
                            // Loop will retry the same step
                            continue;
                        }
                        RedesignStrategy::TacticalRedesign => {
                            log::info!("Performing tactical redesign from step {}", step_index + 1);
                            self.tactical_redesign(&e, step_index).await?;
                            // Retry from the same index with new strategy
                            continue;
                        }
                        RedesignStrategy::FullRegenerate => {
                            log::info!("Performing full strategy regeneration");
                            self.full_regenerate(&e, step_index).await?;
                            // Reset to beginning with new strategy
                            step_index = 0;
                            // Clear context to start fresh
                            self.context.clear();
                            continue;
                        }
                    }
                }
            }
        }

        Ok(final_result)
    }

    /// Determines the appropriate redesign strategy after an error.
    #[cfg(all(feature = "agent", feature = "derive"))]
    async fn determine_redesign_strategy(
        &self,
        error: &crate::agent::AgentError,
        failed_step_index: usize,
        goal: &str,
    ) -> Result<RedesignStrategy, OrchestratorError> {
        use crate::prompt::ToPrompt;
        use prompts::RedesignDecisionRequest;

        // Check if error is transient
        if error.is_transient() {
            return Ok(RedesignStrategy::Retry);
        }

        let strategy = self
            .strategy_map
            .as_ref()
            .ok_or_else(OrchestratorError::no_strategy)?;

        let request = RedesignDecisionRequest::new(
            goal.to_string(),
            failed_step_index,
            strategy.steps.len(),
            strategy.steps[failed_step_index].description.clone(),
            error.to_string(),
            self.format_completed_steps(failed_step_index),
        );

        let prompt = request.to_prompt();

        log::debug!("Redesign decision prompt:\n{}", prompt);

        // Ask internal agent for decision
        let decision = self.internal_agent.execute(prompt).await?;

        let decision_upper = decision.trim().to_uppercase();

        if decision_upper.contains("RETRY") {
            Ok(RedesignStrategy::Retry)
        } else if decision_upper.contains("TACTICAL") {
            Ok(RedesignStrategy::TacticalRedesign)
        } else if decision_upper.contains("FULL") {
            Ok(RedesignStrategy::FullRegenerate)
        } else {
            log::warn!(
                "Unexpected redesign decision: {}. Defaulting to FULL",
                decision
            );
            Ok(RedesignStrategy::FullRegenerate)
        }
    }

    /// Determines the appropriate redesign strategy (stub).
    #[cfg(not(all(feature = "agent", feature = "derive")))]
    async fn determine_redesign_strategy(
        &self,
        error: &crate::agent::AgentError,
        _failed_step_index: usize,
        _goal: &str,
    ) -> Result<RedesignStrategy, OrchestratorError> {
        if error.is_transient() {
            Ok(RedesignStrategy::Retry)
        } else {
            Ok(RedesignStrategy::FullRegenerate)
        }
    }

    /// Performs tactical redesign of remaining steps.
    #[cfg(all(feature = "agent", feature = "derive"))]
    async fn tactical_redesign(
        &mut self,
        error: &crate::agent::AgentError,
        failed_step_index: usize,
    ) -> Result<(), OrchestratorError> {
        use crate::prompt::ToPrompt;
        use prompts::TacticalRedesignRequest;

        let strategy = self
            .strategy_map
            .as_ref()
            .ok_or_else(OrchestratorError::no_strategy)?
            .clone();

        let request = TacticalRedesignRequest::new(
            strategy.goal.clone(),
            serde_json::to_string_pretty(&strategy)
                .map_err(|e| OrchestratorError::StrategyGenerationFailed(e.to_string()))?,
            failed_step_index,
            strategy.steps[failed_step_index].description.clone(),
            error.to_string(),
            self.format_completed_steps(failed_step_index),
            self.format_agent_list(),
        );

        let prompt = request.to_prompt();

        log::debug!("Tactical redesign prompt:\n{}", prompt);

        // Get new steps from LLM
        let response = self.internal_agent.execute(prompt).await?;

        // Parse JSON array of StrategyStep
        let new_steps: Vec<StrategyStep> = serde_json::from_str(&response).map_err(|e| {
            OrchestratorError::StrategyGenerationFailed(format!(
                "Failed to parse redesigned steps: {}",
                e
            ))
        })?;

        log::info!("Tactical redesign generated {} new steps", new_steps.len());

        // Update strategy: keep completed steps, replace from failed_step_index onwards
        if let Some(ref mut strategy) = self.strategy_map {
            strategy.steps.truncate(failed_step_index);
            strategy.steps.extend(new_steps);
        }

        Ok(())
    }

    /// Performs tactical redesign (stub).
    #[cfg(not(all(feature = "agent", feature = "derive")))]
    async fn tactical_redesign(
        &mut self,
        _error: &crate::agent::AgentError,
        _failed_step_index: usize,
    ) -> Result<(), OrchestratorError> {
        Err(OrchestratorError::ExecutionFailed(
            "Tactical redesign not available without agent and derive features".to_string(),
        ))
    }

    /// Performs full strategy regeneration from scratch.
    #[cfg(all(feature = "agent", feature = "derive"))]
    async fn full_regenerate(
        &mut self,
        error: &crate::agent::AgentError,
        failed_step_index: usize,
    ) -> Result<(), OrchestratorError> {
        use crate::prompt::ToPrompt;
        use prompts::FullRegenerateRequest;

        let task = self
            .current_task
            .as_ref()
            .ok_or_else(|| {
                OrchestratorError::Other("No current task available for regeneration".to_string())
            })?
            .clone();

        let strategy = self
            .strategy_map
            .as_ref()
            .ok_or_else(OrchestratorError::no_strategy)?
            .clone();

        let request = FullRegenerateRequest::new(
            task,
            self.format_agent_list(),
            self.blueprint.description.clone(),
            self.blueprint.graph.clone(),
            serde_json::to_string_pretty(&strategy)
                .map_err(|e| OrchestratorError::StrategyGenerationFailed(e.to_string()))?,
            format!(
                "Step {} ({}) failed with error: {}",
                failed_step_index + 1,
                strategy.steps[failed_step_index].description,
                error
            ),
            self.format_completed_steps(failed_step_index),
        );

        let prompt = request.to_prompt();

        log::debug!("Full regeneration prompt:\n{}", prompt);

        // Generate completely new strategy
        let new_strategy = self
            .internal_json_agent
            .execute(prompt)
            .await
            .map_err(|e| OrchestratorError::StrategyGenerationFailed(e.to_string()))?;

        log::info!(
            "Full regeneration completed with {} steps",
            new_strategy.steps.len()
        );

        self.strategy_map = Some(new_strategy);
        Ok(())
    }

    /// Performs full regeneration (stub).
    #[cfg(not(all(feature = "agent", feature = "derive")))]
    async fn full_regenerate(
        &mut self,
        _error: &crate::agent::AgentError,
        _failed_step_index: usize,
    ) -> Result<(), OrchestratorError> {
        Err(OrchestratorError::ExecutionFailed(
            "Full regeneration not available without agent and derive features".to_string(),
        ))
    }

    /// Clears the current strategy and context (useful for re-execution).
    pub fn reset(&mut self) {
        self.strategy_map = None;
        self.context.clear();
        self.current_task = None;
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
