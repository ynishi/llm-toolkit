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
//! use llm_toolkit::orchestrator::{Orchestrator, BlueprintWorkflow, OrchestrationStatus};
//! use llm_toolkit::agent::ClaudeCodeAgent;
//!
//! #[tokio::main]
//! async fn main() {
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
//!     ).await;
//!
//!     match result.status {
//!         OrchestrationStatus::Success => {
//!             println!("Success! Steps: {}, Redesigns: {}",
//!                 result.steps_executed, result.redesigns_triggered);
//!             if let Some(output) = result.final_output {
//!                 println!("{}", output);
//!             }
//!         }
//!         OrchestrationStatus::Failure => {
//!             eprintln!("Failed: {:?}", result.error_message);
//!         }
//!     }
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

use crate::agent::{Agent, AgentAdapter, DynamicAgent};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Status of the orchestration execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrchestrationStatus {
    Success,
    Failure,
}

/// Structured result returned by the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResult {
    pub status: OrchestrationStatus,
    pub final_output: Option<JsonValue>,
    pub steps_executed: usize,
    pub redesigns_triggered: usize,
    pub error_message: Option<String>,
}

#[cfg(feature = "agent")]
use crate::agent::impls::{ClaudeCodeAgent, ClaudeCodeJsonAgent, InnerValidatorAgent};

/// Represents the result of a validation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidationResult {
    status: String,
    reason: String,
}

/// The orchestrator coordinates multiple agents to execute complex workflows.
///
/// The orchestrator maintains:
/// - A `BlueprintWorkflow` describing the intended process
/// - A registry of available agents
/// - Dynamically generated execution strategies
/// - Runtime context for inter-agent communication
pub struct Orchestrator {
    /// The workflow blueprint (reference material for strategy generation).
    #[cfg(any(not(feature = "agent"), feature = "derive"))]
    blueprint: BlueprintWorkflow,

    /// Available agents, keyed by their name.
    /// Uses DynamicAgent for type erasure, allowing heterogeneous agent types.
    agents: HashMap<String, Box<dyn DynamicAgent>>,

    /// Internal JSON agent for structured strategy generation.
    /// Output type is StrategyMap for generating execution strategies.
    #[cfg(feature = "agent")]
    internal_json_agent: Box<dyn Agent<Output = StrategyMap>>,

    /// Internal string agent for intent generation and redesign decisions.
    /// Output type is String for generating prompts and making decisions.
    #[cfg(feature = "agent")]
    internal_agent: Box<dyn Agent<Output = String>>,

    /// Built-in validation agent for validating step outputs.
    /// Always available for the strategy generation LLM to use.
    #[cfg(feature = "agent")]
    inner_validator_agent: Box<dyn Agent<Output = String>>,

    /// The currently active execution strategy.
    strategy_map: Option<StrategyMap>,

    /// Runtime context storing intermediate results as JSON values.
    context: HashMap<String, JsonValue>,

    /// The original task description (stored for regeneration).
    current_task: Option<String>,
}

impl Orchestrator {
    /// Creates a new Orchestrator with a given blueprint.
    ///
    /// Uses default internal agents (ClaudeCodeAgent and ClaudeCodeJsonAgent).
    #[cfg(feature = "agent")]
    pub fn new(blueprint: BlueprintWorkflow) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            internal_json_agent: Box::new(ClaudeCodeJsonAgent::new()),
            internal_agent: Box::new(ClaudeCodeAgent::new()),
            inner_validator_agent: Box::new(InnerValidatorAgent::new()),
            strategy_map: None,
            context: HashMap::new(),
            current_task: None,
        }
    }

    /// Creates a new Orchestrator with custom internal agents.
    ///
    /// This allows you to inject mock or alternative agents for testing or custom LLM backends.
    ///
    /// # Arguments
    ///
    /// * `blueprint` - The workflow blueprint
    /// * `internal_agent` - Agent for string outputs (intent generation, redesign decisions)
    /// * `internal_json_agent` - Agent for StrategyMap generation
    #[cfg(feature = "agent")]
    pub fn with_internal_agents(
        blueprint: BlueprintWorkflow,
        internal_agent: Box<dyn Agent<Output = String>>,
        internal_json_agent: Box<dyn Agent<Output = StrategyMap>>,
    ) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            internal_json_agent,
            internal_agent,
            inner_validator_agent: Box::new(InnerValidatorAgent::new()),
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
    /// Accepts any agent with any output type. The agent will be automatically
    /// wrapped in an `AgentAdapter` for type erasure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[derive(Agent)]
    /// #[agent(expertise = "...", output = "MyCustomType")]
    /// struct MyAgent;
    ///
    /// orchestrator.add_agent(MyAgent);
    /// ```
    pub fn add_agent<T>(&mut self, agent: impl Agent<Output = T> + 'static)
    where
        T: Serialize + serde::de::DeserializeOwned + 'static,
    {
        let adapter = AgentAdapter::new(agent);
        let name = adapter.name();
        self.agents.insert(name, Box::new(adapter));
    }

    /// Adds an agent with ToPrompt support to the orchestrator's registry.
    ///
    /// When the output type implements `ToPrompt`, the orchestrator will automatically
    /// use the prompt representation for context management instead of plain JSON.
    /// This provides better LLM understanding of complex types like enums with descriptions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use llm_toolkit::{ToPrompt, Agent};
    ///
    /// #[derive(ToPrompt, Serialize, Deserialize)]
    /// pub enum AnalysisResult {
    ///     /// The topic is technically sound
    ///     Approved,
    ///     /// Needs revision
    ///     NeedsRevision,
    /// }
    ///
    /// #[derive(Agent)]
    /// #[agent(expertise = "...", output = "AnalysisResult")]
    /// struct AnalyzerAgent;
    ///
    /// orchestrator.add_agent_with_to_prompt(AnalyzerAgent);
    /// ```
    #[cfg(feature = "agent")]
    pub fn add_agent_with_to_prompt<T>(&mut self, agent: impl Agent<Output = T> + 'static)
    where
        T: Serialize + serde::de::DeserializeOwned + crate::prompt::ToPrompt + 'static,
    {
        let adapter = AgentAdapter::with_to_prompt(agent, |output: &T| output.to_prompt());
        let name = adapter.name();
        self.agents.insert(name, Box::new(adapter));
    }

    /// Returns a reference to a dynamic agent by name.
    pub fn get_agent(&self, name: &str) -> Option<&dyn DynamicAgent> {
        self.agents.get(name).map(|boxed| &**boxed)
    }

    /// Returns a list of all available agent names.
    pub fn list_agents(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Returns a formatted string describing all available agents and their expertise.
    #[cfg(feature = "agent")]
    pub fn format_agent_list(&self) -> String {
        let mut agent_list: Vec<String> = self
            .agents
            .iter()
            .map(|(name, agent)| format!("- {}: {}", name, agent.expertise()))
            .collect();

        // Always include the InnerValidatorAgent
        agent_list.push(format!(
            "- {}: {}",
            self.inner_validator_agent.name(),
            self.inner_validator_agent.expertise()
        ));

        agent_list.join("\n")
    }

    /// Returns a formatted string describing all available agents and their expertise.
    #[cfg(not(feature = "agent"))]
    pub fn format_agent_list(&self) -> String {
        self.agents
            .iter()
            .map(|(name, agent)| format!("- {}: {}", name, agent.expertise()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns a reference to the orchestrator's runtime context.
    ///
    /// The context contains intermediate results from executed steps, stored as JSON values.
    /// Keys follow the pattern:
    /// - `step_{step_id}_output`: JSON representation of step output
    /// - `step_{step_id}_output_prompt`: ToPrompt representation (if available)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let context = orchestrator.context();
    /// if let Some(output) = context.get("step_1_output") {
    ///     println!("Step 1 output: {}", output);
    /// }
    /// ```
    pub fn context(&self) -> &HashMap<String, JsonValue> {
        &self.context
    }

    /// Returns the output of a specific step by step_id.
    ///
    /// This is a convenience method for accessing step outputs from the context.
    ///
    /// # Arguments
    ///
    /// * `step_id` - The step ID (e.g., "step_1", "step_2")
    ///
    /// # Returns
    ///
    /// The JSON output of the step, or `None` if the step hasn't been executed yet.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // After executing a workflow
    /// if let Some(concept) = orchestrator.get_step_output("step_1") {
    ///     // Deserialize if you know the type
    ///     let concept: HighConceptResponse = serde_json::from_value(concept.clone())?;
    /// }
    /// ```
    pub fn get_step_output(&self, step_id: &str) -> Option<&JsonValue> {
        self.context.get(&format!("step_{}_output", step_id))
    }

    /// Returns the ToPrompt representation of a step's output, if available.
    ///
    /// This returns the human-readable prompt version of the output, which is only
    /// available if the output type implements `ToPrompt` and the agent was registered
    /// with `add_agent_with_to_prompt()`.
    ///
    /// # Arguments
    ///
    /// * `step_id` - The step ID (e.g., "step_1", "step_2")
    ///
    /// # Returns
    ///
    /// The prompt representation as a string, or `None` if not available.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Get human-readable version of step output
    /// if let Some(prompt) = orchestrator.get_step_output_prompt("step_1") {
    ///     println!("Step 1 output (human-readable):\n{}", prompt);
    /// }
    /// ```
    pub fn get_step_output_prompt(&self, step_id: &str) -> Option<&str> {
        self.context
            .get(&format!("step_{}_output_prompt", step_id))
            .and_then(|v| v.as_str())
    }

    /// Returns all step outputs as a map of step_id to JSON value.
    ///
    /// This filters the context to only include step outputs (not prompt versions).
    ///
    /// # Returns
    ///
    /// A HashMap where keys are step IDs and values are the JSON outputs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let all_outputs = orchestrator.get_all_step_outputs();
    /// for (step_id, output) in all_outputs {
    ///     println!("{}: {:?}", step_id, output);
    /// }
    /// ```
    pub fn get_all_step_outputs(&self) -> HashMap<String, &JsonValue> {
        self.context
            .iter()
            .filter_map(|(key, value)| {
                if key.starts_with("step_") && key.ends_with("_output") && !key.ends_with("_output_prompt") {
                    // Extract step_id from "step_{step_id}_output"
                    key.strip_prefix("step_")
                        .and_then(|s| s.strip_suffix("_output"))
                        .map(|step_id| (step_id.to_string(), value))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Executes the workflow with the given task description.
    ///
    /// This is the main entry point for orchestration. The orchestrator will:
    /// 1. Generate a strategy map from the blueprint and available agents
    /// 2. Execute each step in sequence
    /// 3. Handle errors with adaptive redesign strategies
    /// 4. Return a structured result with execution details
    ///
    /// # Arguments
    ///
    /// * `task` - A natural language description of what needs to be accomplished
    ///
    /// # Returns
    ///
    /// An `OrchestrationResult` containing:
    /// - Status (Success/Failure)
    /// - Final output (if successful)
    /// - Number of steps executed
    /// - Number of redesigns triggered
    /// - Error message (if failed)
    pub async fn execute(&mut self, task: &str) -> OrchestrationResult {
        log::info!("Starting orchestrator execution for task: {}", task);

        // Store task for potential regeneration
        self.current_task = Some(task.to_string());

        // Phase 1: Generate strategy
        if let Err(e) = self.generate_strategy(task).await {
            log::error!("Strategy generation failed: {}", e);
            return OrchestrationResult {
                status: OrchestrationStatus::Failure,
                final_output: None,
                steps_executed: 0,
                redesigns_triggered: 0,
                error_message: Some(e.to_string()),
            };
        }

        // Phase 2: Execute strategy
        match self.execute_strategy().await {
            Ok((final_output, steps_executed, redesigns_triggered)) => {
                log::info!("Orchestrator execution completed successfully");
                OrchestrationResult {
                    status: OrchestrationStatus::Success,
                    final_output: Some(final_output),
                    steps_executed,
                    redesigns_triggered,
                    error_message: None,
                }
            }
            Err(e) => {
                log::error!("Orchestrator execution failed: {}", e);
                OrchestrationResult {
                    status: OrchestrationStatus::Failure,
                    final_output: None,
                    steps_executed: 0,
                    redesigns_triggered: 0,
                    error_message: Some(e.to_string()),
                }
            }
        }
    }

    /// Generates an execution strategy from the blueprint, agents, and task.
    ///
    /// Uses the internal LLM agent to analyze the task, available agents,
    /// and blueprint to generate an optimal execution strategy.
    #[cfg(feature = "agent")]
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
            .execute(prompt.into())
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
    fn collect_context(&self, step_index: usize) -> HashMap<String, JsonValue> {
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

    /// Extracts placeholders from intent template (e.g., {{ concept_content }}).
    fn extract_placeholders(template: &str) -> Vec<String> {
        let mut placeholders = Vec::new();
        let mut chars = template.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'

                // Extract content between {{ and }}
                let mut placeholder = String::new();
                let mut found_end = false;

                while let Some(inner_c) = chars.next() {
                    if inner_c == '}' && chars.peek() == Some(&'}') {
                        chars.next(); // consume second '}'
                        found_end = true;
                        break;
                    }
                    placeholder.push(inner_c);
                }

                if found_end {
                    let placeholder = placeholder.trim().to_string();
                    if !placeholder.is_empty() && !placeholders.contains(&placeholder) {
                        placeholders.push(placeholder);
                    }
                }
            }
        }

        placeholders
    }

    /// Finds the best matching step for a placeholder using keyword matching.
    fn find_matching_step_by_keyword(
        &self,
        placeholder: &str,
        up_to_step_index: usize,
    ) -> Option<String> {
        let strategy = self.strategy_map.as_ref()?;

        // Convert placeholder to searchable keywords
        // "concept_content" -> ["concept", "content"]
        let keywords: Vec<String> = placeholder.split('_').map(|s| s.to_lowercase()).collect();

        let mut best_match: Option<(usize, usize)> = None; // (step_index, match_score)

        for i in 0..up_to_step_index.min(strategy.steps.len()) {
            let step = &strategy.steps[i];
            let searchable_text = format!(
                "{} {} {}",
                step.description.to_lowercase(),
                step.expected_output.to_lowercase(),
                step.step_id.to_lowercase()
            );

            // Count how many keywords match
            let match_count = keywords
                .iter()
                .filter(|keyword| searchable_text.contains(*keyword))
                .count();

            if match_count > 0 {
                if let Some((_, current_best_score)) = best_match {
                    if match_count > current_best_score {
                        best_match = Some((i, match_count));
                    }
                } else {
                    best_match = Some((i, match_count));
                }
            }
        }

        best_match.map(|(step_idx, _)| strategy.steps[step_idx].step_id.clone())
    }

    /// Finds the best matching step for a placeholder using LLM-based semantic matching.
    #[cfg(feature = "agent")]
    async fn find_matching_step_by_llm(
        &self,
        placeholder: &str,
        up_to_step_index: usize,
    ) -> Option<String> {
        use crate::prompt::ToPrompt;
        use prompts::SemanticMatchRequest;

        let strategy = self.strategy_map.as_ref()?;

        // Build steps info for LLM
        let steps_info = (0..up_to_step_index.min(strategy.steps.len()))
            .map(|i| {
                let step = &strategy.steps[i];
                format!(
                    "- step_id: {}\n  description: {}\n  expected_output: {}",
                    step.step_id, step.description, step.expected_output
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let request = SemanticMatchRequest::new(placeholder.to_string(), steps_info);
        let prompt = request.to_prompt();

        log::debug!("Using LLM to match placeholder '{}'", placeholder);

        // Use internal agent to find the match
        match self.internal_agent.execute(prompt.into()).await {
            Ok(step_id) => {
                let step_id = step_id.trim().to_string();
                // Validate that the returned step_id exists
                if strategy.steps.iter().any(|s| s.step_id == step_id) {
                    log::debug!(
                        "LLM matched placeholder '{}' to step '{}'",
                        placeholder,
                        step_id
                    );
                    Some(step_id)
                } else {
                    log::warn!(
                        "LLM returned invalid step_id '{}' for placeholder '{}'",
                        step_id,
                        placeholder
                    );
                    None
                }
            }
            Err(e) => {
                log::error!("LLM semantic matching failed for '{}': {}", placeholder, e);
                None
            }
        }
    }

    /// Builds semantic context by mapping placeholders to appropriate step outputs.
    #[cfg(feature = "agent")]
    async fn build_semantic_context_async(
        &self,
        step_index: usize,
        intent_template: &str,
    ) -> HashMap<String, JsonValue> {
        let mut context = HashMap::new();

        // Extract placeholders from intent template
        let placeholders = Self::extract_placeholders(intent_template);

        log::debug!("Extracted placeholders: {:?}", placeholders);

        for placeholder in placeholders {
            // Try to find matching step using keyword matching first
            let matched_step_id = if let Some(step_id) =
                self.find_matching_step_by_keyword(&placeholder, step_index)
            {
                log::debug!("Keyword match found for '{}': {}", placeholder, step_id);
                Some(step_id)
            } else {
                // Fallback to LLM-based semantic matching
                log::debug!(
                    "No keyword match for '{}', trying LLM semantic matching",
                    placeholder
                );
                self.find_matching_step_by_llm(&placeholder, step_index)
                    .await
            };

            if let Some(matched_step_id) = matched_step_id {
                let key = format!("step_{}_output", matched_step_id);

                // Try to get prompt version first (ToPrompt), fallback to JSON
                let prompt_key = format!("step_{}_output_prompt", matched_step_id);
                if let Some(JsonValue::String(prompt_str)) = self.context.get(&prompt_key) {
                    log::debug!(
                        "Mapped placeholder '{}' to step '{}' (using ToPrompt version)",
                        placeholder,
                        matched_step_id
                    );
                    context.insert(placeholder.clone(), JsonValue::String(prompt_str.clone()));
                } else if let Some(output) = self.context.get(&key) {
                    log::debug!(
                        "Mapped placeholder '{}' to step '{}' (using JSON version)",
                        placeholder,
                        matched_step_id
                    );
                    context.insert(placeholder.clone(), output.clone());
                } else {
                    log::warn!(
                        "Placeholder '{}' matched to step '{}' but output not found in context",
                        placeholder,
                        matched_step_id
                    );
                }
            } else {
                log::warn!(
                    "Could not find matching step for placeholder '{}' using any method",
                    placeholder
                );
            }
        }

        // Always include previous_output for convenience
        if step_index > 0
            && let Some(strategy) = &self.strategy_map
            && let Some(prev_step) = strategy.steps.get(step_index - 1)
        {
            let key = format!("step_{}_output", prev_step.step_id);
            if let Some(output) = self.context.get(&key) {
                context.insert("previous_output".to_string(), output.clone());
            }
        }

        context
    }

    /// Formats context as a readable string for prompts.
    #[cfg(feature = "agent")]
    fn format_context(&self, context: &HashMap<String, JsonValue>) -> String {
        if context.is_empty() {
            return "No context available yet.".to_string();
        }

        context
            .iter()
            .filter_map(|(key, value)| {
                // Skip _prompt versions in the listing (they'll be used as replacements)
                if key.ends_with("_output_prompt") {
                    return None;
                }

                // If there's a _prompt version, prefer it
                let display_value = if let Some(prompt_key) = key.strip_suffix("_output") {
                    let prompt_key_full = format!("{}_output_prompt", prompt_key);
                    if let Some(JsonValue::String(prompt_str)) = context.get(&prompt_key_full) {
                        // Use prompt representation
                        if prompt_str.len() > 500 {
                            format!("{}... (truncated)", &prompt_str[..500])
                        } else {
                            prompt_str.clone()
                        }
                    } else {
                        // Fallback to JSON representation
                        let value_str =
                            serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
                        if value_str.len() > 200 {
                            format!("{}... (truncated)", &value_str[..200])
                        } else {
                            value_str
                        }
                    }
                } else {
                    // Not a step output, use as-is
                    let value_str =
                        serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
                    if value_str.len() > 200 {
                        format!("{}... (truncated)", &value_str[..200])
                    } else {
                        value_str
                    }
                };

                Some(format!("- {}: {}", key, display_value))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Formats completed steps for redesign prompts.
    #[cfg(feature = "agent")]
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
                        .unwrap_or_else(|| JsonValue::String("(no output)".to_string()));

                    // Convert JSON value to string
                    let output_str =
                        serde_json::to_string(&output).unwrap_or_else(|_| "null".to_string());

                    format!(
                        "Step {}: {}\n  Agent: {}\n  Output: {}\n",
                        i + 1,
                        step.description,
                        step.assigned_agent,
                        if output_str.len() > 100 {
                            format!("{}...", &output_str[..100])
                        } else {
                            output_str
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
    #[cfg(feature = "agent")]
    async fn build_intent(
        &self,
        step: &StrategyStep,
        context: &HashMap<String, JsonValue>,
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
        let intent = self.internal_agent.execute(prompt.into()).await?;

        Ok(intent)
    }

    /// Builds intent (stub for non-derive feature).
    #[cfg(not(all(feature = "agent", feature = "derive")))]
    async fn build_intent(
        &self,
        step: &StrategyStep,
        context: &HashMap<String, JsonValue>,
    ) -> Result<String, OrchestratorError> {
        // Simple template substitution as fallback
        let mut intent = step.intent_template.clone();

        for (key, value) in context {
            let placeholder = format!("{{{}}}", key);
            let value_str = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
            intent = intent.replace(&placeholder, &value_str);
        }

        Ok(intent)
    }

    /// Executes the current strategy step by step.
    ///
    /// Includes context injection, intent generation, and error handling with redesign.
    /// Returns (final_output, steps_executed, redesigns_triggered).
    async fn execute_strategy(&mut self) -> Result<(JsonValue, usize, usize), OrchestratorError> {
        // Check strategy exists
        if self.strategy_map.is_none() {
            return Err(OrchestratorError::no_strategy());
        }

        let mut final_result = JsonValue::Null;
        let mut step_index = 0;
        let mut steps_executed = 0;
        let mut redesigns_triggered = 0;

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

            // Build semantic context by mapping placeholders to appropriate step outputs
            let context = self
                .build_semantic_context_async(step_index, &step.intent_template)
                .await;

            // Build intent using LLM
            let intent = self.build_intent(&step, &context).await?;

            log::debug!("Generated intent:\n{}", intent);

            // Execute agent
            let agent = self
                .agents
                .get(&step.assigned_agent)
                .ok_or_else(|| OrchestratorError::AgentNotFound(step.assigned_agent.clone()))?;

            match agent.execute_dynamic(intent.into()).await {
                Ok(output) => {
                    log::info!("Step {} completed successfully", step_index + 1);

                    // Store JSON result in context
                    self.context
                        .insert(format!("step_{}_output", step.step_id), output.clone());

                    // Store prompt version if available (for ToPrompt implementations)
                    if let Some(prompt_str) = agent.try_to_prompt(&output) {
                        log::debug!("Storing prompt representation for step {}", step.step_id);
                        self.context.insert(
                            format!("step_{}_output_prompt", step.step_id),
                            JsonValue::String(prompt_str),
                        );
                    }

                    final_result = output;
                    steps_executed += 1;

                    // Check if this step requires validation
                    let requires_validation = step.requires_validation;
                    step_index += 1;

                    // If validation is required, the next step should be the validation step
                    if requires_validation {
                        if step_index >= self.strategy_map.as_ref().unwrap().steps.len() {
                            log::warn!("Step requires validation but no validation step found");
                            continue;
                        }

                        // Execute validation step
                        let validation_result = self.execute_validation_step(step_index).await?;

                        // Parse validation result
                        if let Some(validation_status) =
                            self.parse_validation_result(&validation_result)
                        {
                            if validation_status.status == "FAIL" {
                                log::warn!(
                                    "Validation failed for step {}: {}",
                                    step_index,
                                    validation_status.reason
                                );

                                // Treat validation failure as if the original step failed
                                let validation_error = crate::agent::AgentError::ExecutionFailed(
                                    format!("Validation failed: {}", validation_status.reason),
                                );

                                // Trigger redesign for the original step (step_index - 1)
                                match self
                                    .determine_redesign_strategy(
                                        &validation_error,
                                        step_index - 1,
                                        &goal,
                                    )
                                    .await?
                                {
                                    RedesignStrategy::Retry => {
                                        log::info!(
                                            "Retrying step {} after validation failure",
                                            step_index
                                        );
                                        redesigns_triggered += 1;
                                        step_index -= 1; // Go back to the original step
                                        continue;
                                    }
                                    RedesignStrategy::TacticalRedesign => {
                                        log::info!(
                                            "Performing tactical redesign from step {} after validation failure",
                                            step_index
                                        );
                                        redesigns_triggered += 1;
                                        self.tactical_redesign(&validation_error, step_index - 1)
                                            .await?;
                                        step_index -= 1; // Go back to the original step
                                        continue;
                                    }
                                    RedesignStrategy::FullRegenerate => {
                                        log::info!(
                                            "Performing full strategy regeneration after validation failure"
                                        );
                                        redesigns_triggered += 1;
                                        self.full_regenerate(&validation_error, step_index - 1)
                                            .await?;
                                        step_index = 0;
                                        self.context.clear();
                                        continue;
                                    }
                                }
                            } else {
                                log::info!("Validation passed for step {}", step_index);
                                // Store validation result as JSON string
                                let validation_step =
                                    &self.strategy_map.as_ref().unwrap().steps[step_index];
                                self.context.insert(
                                    format!("step_{}_output", validation_step.step_id),
                                    JsonValue::String(validation_result),
                                );
                                step_index += 1; // Move to next step
                            }
                        } else {
                            log::warn!("Failed to parse validation result, continuing anyway");
                            step_index += 1;
                        }
                    }
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
                            redesigns_triggered += 1;
                            // Loop will retry the same step
                            continue;
                        }
                        RedesignStrategy::TacticalRedesign => {
                            log::info!("Performing tactical redesign from step {}", step_index + 1);
                            redesigns_triggered += 1;
                            self.tactical_redesign(&e, step_index).await?;
                            // Retry from the same index with new strategy
                            continue;
                        }
                        RedesignStrategy::FullRegenerate => {
                            log::info!("Performing full strategy regeneration");
                            redesigns_triggered += 1;
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

        Ok((final_result, steps_executed, redesigns_triggered))
    }

    /// Executes a validation step.
    #[cfg(feature = "agent")]
    async fn execute_validation_step(
        &mut self,
        validation_step_index: usize,
    ) -> Result<String, OrchestratorError> {
        let step = {
            let strategy = self
                .strategy_map
                .as_ref()
                .ok_or_else(OrchestratorError::no_strategy)?;
            if validation_step_index >= strategy.steps.len() {
                return Err(OrchestratorError::ExecutionFailed(
                    "Validation step index out of bounds".to_string(),
                ));
            }
            strategy.steps[validation_step_index].clone()
        };

        log::info!(
            "Executing validation step {}: {}",
            validation_step_index + 1,
            step.description
        );

        // Collect context
        let context = self.collect_context(validation_step_index);

        // Build intent for validation
        let intent = self.build_intent(&step, &context).await?;

        log::debug!("Validation intent:\n{}", intent);

        // Execute the inner validator agent
        let result = self.inner_validator_agent.execute(intent.into()).await?;

        log::debug!("Validation result:\n{}", result);

        Ok(result)
    }

    /// Executes a validation step (stub for non-agent feature).
    #[cfg(not(feature = "agent"))]
    async fn execute_validation_step(
        &mut self,
        _validation_step_index: usize,
    ) -> Result<String, OrchestratorError> {
        Err(OrchestratorError::ExecutionFailed(
            "Validation not available without agent feature".to_string(),
        ))
    }

    /// Parses validation result JSON.
    fn parse_validation_result(&self, result: &str) -> Option<ValidationResult> {
        serde_json::from_str(result).ok()
    }

    /// Determines the appropriate redesign strategy after an error.
    #[cfg(feature = "agent")]
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
        let decision = self.internal_agent.execute(prompt.into()).await?;

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
    #[cfg(feature = "agent")]
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
        let response = self.internal_agent.execute(prompt.into()).await?;

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
    #[cfg(feature = "agent")]
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
            .execute(prompt.into())
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

#[cfg(all(test, feature = "agent"))]
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

        orch.add_agent(ClaudeCodeAgent::new());
        assert_eq!(orch.list_agents().len(), 1);
        assert!(orch.get_agent("ClaudeCodeAgent").is_some());
    }

    #[test]
    fn test_format_agent_list() {
        let blueprint = BlueprintWorkflow::new("Test".to_string());
        let mut orch = Orchestrator::new(blueprint);

        orch.add_agent(ClaudeCodeAgent::new());
        let list = orch.format_agent_list();
        assert!(list.contains("ClaudeCodeAgent"));
    }

    #[test]
    fn test_context_initially_empty() {
        let blueprint = BlueprintWorkflow::new("Test".to_string());
        let orch = Orchestrator::new(blueprint);

        // Initially context should be empty
        assert!(orch.context().is_empty());
        assert!(orch.get_step_output("step_1").is_none());
        assert!(orch.get_all_step_outputs().is_empty());
    }

    #[test]
    fn test_context_accessors_available() {
        let blueprint = BlueprintWorkflow::new("Test".to_string());
        let orch = Orchestrator::new(blueprint);

        // Verify all accessor methods are callable (main fix - these were not accessible before)
        let _context = orch.context();
        let _step_output = orch.get_step_output("step_1");
        let _step_prompt = orch.get_step_output_prompt("step_1");
        let _all_outputs = orch.get_all_step_outputs();

        // Before this fix, the above lines would not compile because context was private
        assert!(true, "Context accessor methods are now public and accessible");
    }
}
