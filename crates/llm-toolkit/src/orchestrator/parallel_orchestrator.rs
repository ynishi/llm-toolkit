//! Parallel orchestrator for concurrent workflow execution.
//!
//! This module provides the main `ParallelOrchestrator` implementation that executes
//! workflow steps concurrently based on their dependencies.

use crate::agent::DynamicAgent;
use crate::orchestrator::{
    OrchestratorError, StrategyInstruction, StrategyMap, StrategyStep, TerminateInstruction,
};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, info, info_span, warn};

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
    /// Indicates whether execution ended early via a Terminate instruction
    pub terminated: bool,
    /// Optional termination payload rendered from the instruction
    pub termination_reason: Option<String>,
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
            terminated: false,
            termination_reason: None,
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
            terminated: false,
            termination_reason: None,
        }
    }

    /// Creates a result representing an early termination.
    pub fn terminated(
        steps_executed: usize,
        steps_skipped: usize,
        context: HashMap<String, JsonValue>,
        termination_reason: Option<String>,
    ) -> Self {
        Self {
            success: true,
            steps_executed,
            steps_skipped,
            context,
            error: None,
            terminated: true,
            termination_reason,
        }
    }
}

#[derive(Debug, Clone)]
struct ExecutionSegment {
    steps: Vec<StrategyStep>,
    terminate: Option<TerminateInstruction>,
}

#[derive(Debug)]
struct SegmentOutcome {
    exec_state: ExecutionStateManager,
    steps_executed: usize,
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
    /// * `cancellation_token` - Token to cancel execution
    ///
    /// # Returns
    ///
    /// An `ParallelOrchestrationResult` containing execution status and context
    pub async fn execute(
        &mut self,
        task: &str,
        cancellation_token: CancellationToken,
    ) -> Result<ParallelOrchestrationResult, OrchestratorError> {
        let total_steps = self.strategy.steps.len();

        async {
            info!("Starting parallel orchestration for task: {}", task);

            let (prefix_instructions, truncated_due_to_loop) =
                Self::collect_parallel_prefix(&self.strategy);

            if truncated_due_to_loop {
                debug!(
                    "Loop boundary encountered; limiting parallel execution to {} instruction(s)",
                    prefix_instructions.len()
                );
            }

            let segments = Self::build_segments(&prefix_instructions);
            let shared_context = Arc::new(Mutex::new(HashMap::new()));

            {
                let mut ctx = shared_context.lock().await;
                ctx.insert("task".to_string(), JsonValue::String(task.to_string()));
            }

            let mut steps_executed_total = 0usize;
            let mut steps_skipped_total = 0usize;

            for (segment_index, segment) in segments.iter().enumerate() {
                if !segment.steps.is_empty() {
                    let segment_result = self
                        .execute_segment(segment, Arc::clone(&shared_context), cancellation_token.clone())
                        .await?;

                    steps_executed_total += segment_result.steps_executed;
                    steps_skipped_total += segment_result.exec_state.get_skipped_steps().len();

                    if segment_result.exec_state.has_failures() {
                        let final_context = shared_context.lock().await.clone();
                        steps_skipped_total +=
                            Self::count_steps_in_segments(&segments, segment_index + 1);

                        let failed_steps = segment_result.exec_state.get_failed_steps();
                        let mut has_timeout = false;
                        let mut error_details = Vec::new();

                        for (step_id, err) in &failed_steps {
                            if matches!(**err, OrchestratorError::StepTimeout { .. }) {
                                has_timeout = true;
                            }
                            error_details.push(format!("{}: {}", step_id, err));
                        }

                        let error_msg = if has_timeout {
                            format!(
                                "Workflow failed: {} step(s) timed out, {} skipped. Details: {}",
                                failed_steps.len(),
                                steps_skipped_total,
                                error_details.join("; ")
                            )
                        } else {
                            format!(
                                "Workflow failed: {} step(s) failed, {} skipped",
                                failed_steps.len(),
                                steps_skipped_total
                            )
                        };

                        return Ok(ParallelOrchestrationResult::failure(
                            steps_executed_total,
                            steps_skipped_total,
                            final_context,
                            error_msg,
                        ));
                    }
                }

                if let Some(terminate) = &segment.terminate {
                    let should_terminate = self
                        .evaluate_termination_condition(terminate, &shared_context)
                        .await?;

                    if should_terminate {
                        info!("Termination triggered: {}", terminate.terminate_id);

                        steps_skipped_total +=
                            Self::count_steps_in_segments(&segments, segment_index + 1);

                        let termination_reason = self
                            .render_termination_payload(terminate, &shared_context)
                            .await?;

                        let final_context = shared_context.lock().await.clone();

                        return Ok(ParallelOrchestrationResult::terminated(
                            steps_executed_total,
                            steps_skipped_total,
                            final_context,
                            termination_reason,
                        ));
                    }
                }
            }

            let final_context = shared_context.lock().await.clone();
            Ok(ParallelOrchestrationResult::success(
                steps_executed_total,
                final_context,
            ))
        }
        .instrument(info_span!(
            "parallel_orchestrator_execute",
            task = %task,
            total_steps = total_steps,
        ))
        .await
    }

    /// Executes a wave of independent steps concurrently.
    async fn execute_wave(
        &self,
        step_ids: Vec<String>,
        step_lookup: &HashMap<String, StrategyStep>,
        shared_context: Arc<Mutex<HashMap<String, JsonValue>>>,
        cancellation_token: CancellationToken,
    ) -> Vec<(String, Result<JsonValue, OrchestratorError>)> {
        let mut tasks = Vec::new();
        let step_timeout = self.config.step_timeout;

        for step_id in step_ids {
            // Find the step definition
            let step = match step_lookup.get(&step_id) {
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
            let cancel_token = cancellation_token.clone();

            // Create span for this step
            let step_span = info_span!(
                "parallel_step",
                step_id = %step.step_id,
                agent_name = %step.assigned_agent,
            );

            // Spawn task with span
            let task = tokio::spawn(
                async move {
                    // Render intent template
                    let intent = match Self::render_template(&step.intent_template, &context).await
                    {
                        Ok(i) => i,
                        Err(e) => return (step_id.clone(), Err(e)),
                    };

                    // Execute agent with optional timeout and cancellation
                    let result = if let Some(timeout_duration) = step_timeout {
                        tokio::select! {
                            _ = cancel_token.cancelled() => {
                                warn!(step_id = %step_id, "Step cancelled");
                                Err(OrchestratorError::Cancelled {
                                    step_id: step_id.clone(),
                                })
                            }
                            timeout_result = tokio::time::timeout(
                                timeout_duration,
                                agent.execute_dynamic(intent.into()),
                            ) => {
                                match timeout_result {
                                    Ok(Ok(output)) => Ok(output),
                                    Ok(Err(e)) => Err(e.into()),
                                    Err(_) => {
                                        warn!(
                                            step_id = %step_id,
                                            timeout = ?timeout_duration,
                                            "Step execution timed out"
                                        );
                                        Err(OrchestratorError::StepTimeout {
                                            step_id: step_id.clone(),
                                            timeout: timeout_duration,
                                        })
                                    }
                                }
                            }
                        }
                    } else {
                        tokio::select! {
                            _ = cancel_token.cancelled() => {
                                warn!(step_id = %step_id, "Step cancelled");
                                Err(OrchestratorError::Cancelled {
                                    step_id: step_id.clone(),
                                })
                            }
                            agent_result = agent.execute_dynamic(intent.into()) => {
                                agent_result.map_err(|e| e.into())
                            }
                        }
                    };

                    (step_id.clone(), result)
                }
                .instrument(step_span),
            );

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

    async fn execute_segment(
        &self,
        segment: &ExecutionSegment,
        shared_context: Arc<Mutex<HashMap<String, JsonValue>>>,
        cancellation_token: CancellationToken,
    ) -> Result<SegmentOutcome, OrchestratorError> {
        if segment.steps.is_empty() {
            return Ok(SegmentOutcome {
                exec_state: ExecutionStateManager::new(),
                steps_executed: 0,
            });
        }

        let mut subset_strategy = StrategyMap::new(self.strategy.goal.clone());
        subset_strategy.steps = segment.steps.clone();

        let dep_graph = build_dependency_graph(&subset_strategy)?;
        let mut exec_state = ExecutionStateManager::new();

        for step in &segment.steps {
            exec_state.set_state(&step.step_id, StepState::Pending);
        }

        let step_lookup = Self::create_step_lookup(&segment.steps);

        for step_id in dep_graph.get_zero_dependency_steps() {
            exec_state.set_state(&step_id, StepState::Ready);
            debug!(step_id = %step_id, "Step marked as Ready (no dependencies)");
        }

        let mut steps_executed = 0usize;
        let mut wave_number = 0usize;

        while exec_state.has_ready_or_running_steps() || exec_state.has_pending_steps() {
            let ready_steps = exec_state.get_ready_steps();

            if ready_steps.is_empty() {
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
            let wave_span = info_span!(
                "wave",
                wave_number = wave_number,
                ready_steps = ready_steps.len()
            );

            let _wave_guard = wave_span.enter();

            info!(
                "Executing wave {} with {} steps",
                wave_number,
                ready_steps.len()
            );

            for step_id in &ready_steps {
                exec_state.set_state(step_id, StepState::Running);
                debug!(step_id = %step_id, "Step execution started");
            }

            let results = self
                .execute_wave(ready_steps, &step_lookup, Arc::clone(&shared_context), cancellation_token.clone())
                .await;

            // Process results
            for (step_id, result) in results {
                match result {
                    Ok(output) => {
                        exec_state.set_state(&step_id, StepState::Completed);
                        info!(step_id = %step_id, "Step completed successfully");
                        steps_executed += 1;

                        {
                            let mut ctx = shared_context.lock().await;

                            if let Some(step) = step_lookup.get(&step_id) {
                                let output_key = step
                                    .output_key
                                    .clone()
                                    .unwrap_or_else(|| format!("{}_output", step_id));
                                ctx.insert(output_key, output);
                            }
                        }

                        self.unlock_dependents(&step_id, &dep_graph, &mut exec_state);
                    }
                    Err(e) => {
                        warn!(step_id = %step_id, error = %e, "Step failed");
                        exec_state.set_state(&step_id, StepState::Failed(Arc::new(e)));
                        self.cascade_skipped(&step_id, &dep_graph, &mut exec_state);
                    }
                }
            }
        }

        Ok(SegmentOutcome {
            exec_state,
            steps_executed,
        })
    }

    fn collect_parallel_prefix(strategy: &StrategyMap) -> (Vec<StrategyInstruction>, bool) {
        if !strategy.elements.is_empty() {
            let mut prefix = Vec::new();
            let mut truncated = false;

            for instruction in &strategy.elements {
                match instruction {
                    StrategyInstruction::Loop(_) => {
                        truncated = true;
                        break;
                    }
                    other => prefix.push(other.clone()),
                }
            }

            (prefix, truncated)
        } else {
            (
                strategy
                    .steps
                    .iter()
                    .cloned()
                    .map(StrategyInstruction::Step)
                    .collect(),
                false,
            )
        }
    }

    fn build_segments(instructions: &[StrategyInstruction]) -> Vec<ExecutionSegment> {
        let mut segments = Vec::new();
        let mut current_steps = Vec::new();

        for instruction in instructions {
            match instruction {
                StrategyInstruction::Step(step) => current_steps.push(step.clone()),
                StrategyInstruction::Terminate(term) => {
                    segments.push(ExecutionSegment {
                        steps: mem::take(&mut current_steps),
                        terminate: Some(term.clone()),
                    });
                }
                StrategyInstruction::Loop(_) => {
                    // Loop instructions should have been truncated already.
                }
            }
        }

        if !current_steps.is_empty() || segments.is_empty() {
            segments.push(ExecutionSegment {
                steps: current_steps,
                terminate: None,
            });
        }

        segments
    }

    fn count_steps_in_segments(segments: &[ExecutionSegment], start_index: usize) -> usize {
        segments
            .iter()
            .skip(start_index)
            .map(|segment| segment.steps.len())
            .sum()
    }

    fn create_step_lookup(steps: &[StrategyStep]) -> HashMap<String, StrategyStep> {
        let mut map = HashMap::with_capacity(steps.len());
        for step in steps {
            map.insert(step.step_id.clone(), step.clone());
        }
        map
    }

    async fn evaluate_termination_condition(
        &self,
        terminate: &TerminateInstruction,
        context: &Arc<Mutex<HashMap<String, JsonValue>>>,
    ) -> Result<bool, OrchestratorError> {
        match &terminate.condition_template {
            None => Ok(false),
            Some(template) => {
                let rendered = Self::render_template(template, context).await?;
                Ok(rendered.trim().eq_ignore_ascii_case("true"))
            }
        }
    }

    async fn render_termination_payload(
        &self,
        terminate: &TerminateInstruction,
        context: &Arc<Mutex<HashMap<String, JsonValue>>>,
    ) -> Result<Option<String>, OrchestratorError> {
        if let Some(template) = &terminate.final_output_template {
            let rendered = Self::render_template(template, context).await?;
            Ok(Some(rendered))
        } else {
            Ok(None)
        }
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
                debug!(step_id = %dependent_id, "Step marked as Ready (dependencies completed)");
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
                    debug!(step_id = %dependent, failed_dependency = %step_id, "Step skipped due to failed dependency");
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
