//! Parallel orchestrator for concurrent workflow execution.
//!
//! This module provides the main `ParallelOrchestrator` implementation that executes
//! workflow steps concurrently based on their dependencies.

use crate::agent::DynamicAgent;
use crate::orchestrator::{OrchestratorError, StrategyMap};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::parallel::{
    DependencyGraph, ExecutionStateManager, ParallelOrchestratorConfig, StepState,
    build_dependency_graph,
};

/// Result of orchestrator execution.
#[derive(Debug, Clone)]
pub struct ParallelOrchestrationResult {
    /// Final execution status
    pub success: bool,
    /// Number of steps executed
    pub steps_executed: usize,
    /// Number of steps skipped due to failures
    pub steps_skipped: usize,
    /// Execution context with all step outputs
    pub context: HashMap<String, JsonValue>,
    /// Error message if failed
    pub error: Option<String>,
}

impl ParallelOrchestrationResult {
    /// Creates a successful result
    pub fn success(steps_executed: usize, context: HashMap<String, JsonValue>) -> Self {
        Self {
            success: true,
            steps_executed,
            steps_skipped: 0,
            context,
            error: None,
        }
    }

    /// Creates a failed result
    pub fn failure(
        steps_executed: usize,
        steps_skipped: usize,
        context: HashMap<String, JsonValue>,
        error: String,
    ) -> Self {
        Self {
            success: false,
            steps_executed,
            steps_skipped,
            context,
            error: Some(error),
        }
    }
}

/// Parallel orchestrator for concurrent workflow execution.
///
/// This orchestrator analyzes workflow dependencies and executes independent steps
/// concurrently in "waves" for optimal performance.
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::orchestrator::ParallelOrchestrator;
///
/// let mut orchestrator = ParallelOrchestrator::new(strategy_map);
/// orchestrator.add_agent("Agent1", my_agent);
///
/// let result = orchestrator.execute("Process customer data").await?;
/// assert!(result.success);
/// ```
pub struct ParallelOrchestrator {
    /// Strategy map defining the workflow
    strategy: StrategyMap,
    /// Agent registry
    agents: HashMap<String, Arc<dyn DynamicAgent + Send + Sync>>,
    /// Configuration
    #[allow(dead_code)] // TODO: Will be used for concurrency limiting and timeouts in Phase 5
    config: ParallelOrchestratorConfig,
}

impl ParallelOrchestrator {
    /// Creates a new parallel orchestrator with the given strategy.
    pub fn new(strategy: StrategyMap) -> Self {
        Self {
            strategy,
            agents: HashMap::new(),
            config: ParallelOrchestratorConfig::default(),
        }
    }

    /// Creates a new parallel orchestrator with custom configuration.
    pub fn with_config(strategy: StrategyMap, config: ParallelOrchestratorConfig) -> Self {
        Self {
            strategy,
            agents: HashMap::new(),
            config,
        }
    }

    /// Registers an agent with the orchestrator.
    ///
    /// The agent must be Send + Sync for concurrent execution.
    pub fn add_agent(
        &mut self,
        name: impl Into<String>,
        agent: Arc<dyn DynamicAgent + Send + Sync>,
    ) {
        self.agents.insert(name.into(), agent);
    }

    /// Executes the workflow with the given task.
    ///
    /// This analyzes dependencies, executes steps in parallel waves, and handles
    /// failures by propagating skipped state to dependent steps.
    ///
    /// # Arguments
    ///
    /// * `task` - The task description to process
    ///
    /// # Returns
    ///
    /// An `ParallelOrchestrationResult` containing execution status and context
    pub async fn execute(
        &mut self,
        task: &str,
    ) -> Result<ParallelOrchestrationResult, OrchestratorError> {
        info!("Starting parallel orchestration for task: {}", task);

        // 1. Build dependency graph
        let dep_graph = build_dependency_graph(&self.strategy)?;
        debug!(
            "Built dependency graph with {} nodes",
            dep_graph.node_count()
        );

        // 2. Initialize execution state
        let mut exec_state = ExecutionStateManager::new();
        let shared_context = Arc::new(Mutex::new(HashMap::new()));

        // Populate initial context
        {
            let mut ctx = shared_context.lock().await;
            ctx.insert("task".to_string(), JsonValue::String(task.to_string()));
        }

        // Initialize all steps as Pending
        for step in &self.strategy.steps {
            exec_state.set_state(&step.step_id, StepState::Pending);
        }

        // Mark zero-dependency steps as Ready
        for step_id in dep_graph.get_zero_dependency_steps() {
            exec_state.set_state(&step_id, StepState::Ready);
            debug!("Step {} is ready (no dependencies)", step_id);
        }

        let mut steps_executed = 0;
        let mut wave_number = 0;

        // 3. Main execution loop
        while exec_state.has_ready_or_running_steps() || exec_state.has_pending_steps() {
            let ready_steps = exec_state.get_ready_steps();

            if ready_steps.is_empty() {
                // Wait for running tasks or check if we're stuck
                if !exec_state.has_ready_or_running_steps() {
                    warn!(
                        "No ready or running steps, but pending steps remain - potential deadlock"
                    );
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                continue;
            }

            wave_number += 1;
            info!(
                "Executing wave {} with {} steps",
                wave_number,
                ready_steps.len()
            );

            // Execute wave
            for step_id in &ready_steps {
                exec_state.set_state(step_id, StepState::Running);
            }

            let results = self
                .execute_wave(
                    ready_steps,
                    &dep_graph,
                    &exec_state,
                    Arc::clone(&shared_context),
                )
                .await;

            // Process results
            for (step_id, result) in results {
                match result {
                    Ok(output) => {
                        exec_state.set_state(&step_id, StepState::Completed);
                        steps_executed += 1;

                        // Store output in context
                        {
                            let mut ctx = shared_context.lock().await;

                            // Find the step to get its output_key
                            if let Some(step) =
                                self.strategy.steps.iter().find(|s| s.step_id == step_id)
                            {
                                let output_key = step
                                    .output_key
                                    .clone()
                                    .unwrap_or_else(|| format!("{}_output", step_id));
                                ctx.insert(output_key, output);
                            }
                        }

                        // Unlock dependent steps
                        self.unlock_dependents(&step_id, &dep_graph, &mut exec_state);
                    }
                    Err(e) => {
                        warn!("Step {} failed: {}", step_id, e);
                        exec_state.set_state(&step_id, StepState::Failed(Arc::new(e)));

                        // Cascade skipped to dependents
                        self.cascade_skipped(&step_id, &dep_graph, &mut exec_state);
                    }
                }
            }
        }

        // 4. Build final result
        let final_context = shared_context.lock().await.clone();
        let steps_skipped =
            exec_state.get_ready_steps().len() + exec_state.get_running_steps().len();

        if exec_state.has_failures() {
            let failed_steps = exec_state.get_failed_steps();
            let error_msg = format!(
                "Workflow failed: {} step(s) failed, {} skipped",
                failed_steps.len(),
                steps_skipped
            );
            Ok(ParallelOrchestrationResult::failure(
                steps_executed,
                steps_skipped,
                final_context,
                error_msg,
            ))
        } else {
            Ok(ParallelOrchestrationResult::success(
                steps_executed,
                final_context,
            ))
        }
    }

    /// Executes a wave of independent steps concurrently.
    async fn execute_wave(
        &self,
        step_ids: Vec<String>,
        _dep_graph: &DependencyGraph,
        _exec_state: &ExecutionStateManager,
        shared_context: Arc<Mutex<HashMap<String, JsonValue>>>,
    ) -> Vec<(String, Result<JsonValue, OrchestratorError>)> {
        let mut tasks = Vec::new();

        for step_id in step_ids {
            // Find the step definition
            let step = match self.strategy.steps.iter().find(|s| s.step_id == step_id) {
                Some(s) => s.clone(),
                None => {
                    tasks.push(tokio::spawn(async move {
                        (
                            step_id.clone(),
                            Err(OrchestratorError::ExecutionFailed(format!(
                                "Step {} not found in strategy",
                                step_id
                            ))),
                        )
                    }));
                    continue;
                }
            };

            // Find the agent
            let agent = match self.agents.get(&step.assigned_agent) {
                Some(a) => Arc::clone(a),
                None => {
                    tasks.push(tokio::spawn(async move {
                        (
                            step_id.clone(),
                            Err(OrchestratorError::ExecutionFailed(format!(
                                "Agent {} not found",
                                step.assigned_agent
                            ))),
                        )
                    }));
                    continue;
                }
            };

            let context = Arc::clone(&shared_context);

            // Spawn task
            let task = tokio::spawn(async move {
                // Render intent template
                let intent = match Self::render_template(&step.intent_template, &context).await {
                    Ok(i) => i,
                    Err(e) => return (step_id.clone(), Err(e)),
                };

                // Execute agent
                let result = agent
                    .execute_dynamic(intent.into())
                    .await
                    .map_err(|e| e.into());

                (step_id.clone(), result)
            });

            tasks.push(task);
        }

        // Wait for all tasks
        let mut results = Vec::new();
        for task in tasks {
            if let Ok(result) = task.await {
                results.push(result);
            }
        }

        results
    }

    /// Renders a Jinja2 template with the current context.
    async fn render_template(
        template: &str,
        context: &Arc<Mutex<HashMap<String, JsonValue>>>,
    ) -> Result<String, OrchestratorError> {
        use minijinja::Environment;

        let ctx = context.lock().await;
        let env = Environment::new();
        let tmpl = env
            .template_from_str(template)
            .map_err(|e| OrchestratorError::TemplateRenderError(e.to_string()))?;

        tmpl.render(&*ctx)
            .map_err(|e| OrchestratorError::TemplateRenderError(e.to_string()))
    }

    /// Unlocks dependent steps when a step completes successfully.
    fn unlock_dependents(
        &self,
        step_id: &str,
        dep_graph: &DependencyGraph,
        exec_state: &mut ExecutionStateManager,
    ) {
        for dependent_id in dep_graph.get_dependents(step_id) {
            // Check if all dependencies are completed
            let all_deps_completed = dep_graph
                .get_dependencies(&dependent_id)
                .iter()
                .all(|dep| matches!(exec_state.get_state(dep), Some(StepState::Completed)));

            if all_deps_completed
                && matches!(
                    exec_state.get_state(&dependent_id),
                    Some(StepState::Pending)
                )
            {
                exec_state.set_state(&dependent_id, StepState::Ready);
                debug!("Step {} is now ready", dependent_id);
            }
        }
    }

    /// Cascades skipped state to all dependent steps when a step fails.
    fn cascade_skipped(
        &self,
        failed_step_id: &str,
        dep_graph: &DependencyGraph,
        exec_state: &mut ExecutionStateManager,
    ) {
        let mut to_skip = vec![failed_step_id.to_string()];
        let mut visited = std::collections::HashSet::new();

        while let Some(step_id) = to_skip.pop() {
            if visited.contains(&step_id) {
                continue;
            }
            visited.insert(step_id.clone());

            for dependent in dep_graph.get_dependents(&step_id) {
                if !matches!(exec_state.get_state(&dependent), Some(StepState::Completed)) {
                    exec_state.set_state(&dependent, StepState::Skipped);
                    debug!("Step {} skipped due to failed dependency", dependent);
                    to_skip.push(dependent.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // We'll add tests incrementally
    #[test]
    fn test_orchestrator_creation() {
        let strategy = StrategyMap::new("Test".to_string());
        let orchestrator = ParallelOrchestrator::new(strategy);

        assert_eq!(orchestrator.agents.len(), 0);
    }
}
