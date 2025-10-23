//! Parallel orchestrator for concurrent workflow execution.
//!
//! This module provides the main `ParallelOrchestrator` implementation that executes
//! workflow steps concurrently based on their dependencies.

use crate::agent::DynamicAgent;
use crate::orchestrator::{
    OrchestratorError, StrategyInstruction, StrategyMap, StrategyStep, TerminateInstruction,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::mem;
use std::path::Path;
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
    /// Indicates whether execution was paused for human approval
    pub paused: bool,
    /// Reason for pause (human-readable message from agent)
    pub pause_reason: Option<String>,
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
            paused: false,
            pause_reason: None,
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
            paused: false,
            pause_reason: None,
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
            paused: false,
            pause_reason: None,
        }
    }

    /// Creates a result representing a paused execution awaiting approval.
    pub fn paused(
        steps_executed: usize,
        steps_skipped: usize,
        context: HashMap<String, JsonValue>,
        pause_reason: String,
    ) -> Self {
        Self {
            success: true,
            steps_executed,
            steps_skipped,
            context,
            error: None,
            terminated: false,
            termination_reason: None,
            paused: true,
            pause_reason: Some(pause_reason),
        }
    }
}

/// Serializable state for resuming orchestration.
///
/// This struct captures the complete state of an orchestration at a point in time,
/// allowing execution to be paused and resumed later.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OrchestrationState {
    /// Shared context containing all step outputs and task information
    pub context: HashMap<String, JsonValue>,
    /// Execution state manager tracking the status of all steps
    pub execution_manager: ExecutionStateManager,
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
/// use llm_toolkit::orchestrator::{ParallelOrchestrator, BlueprintWorkflow};
///
/// let blueprint = BlueprintWorkflow::new("My workflow");
/// let mut orchestrator = ParallelOrchestrator::new(blueprint);
/// orchestrator.add_agent("Agent1", my_agent);
///
/// let result = orchestrator.execute("Process customer data", token, None, None).await?;
/// assert!(result.success);
/// ```
pub struct ParallelOrchestrator {
    /// The workflow blueprint (reference material for strategy generation).
    #[cfg(feature = "agent")]
    blueprint: crate::orchestrator::BlueprintWorkflow,

    /// Internal JSON agent for structured strategy generation.
    /// Output type is StrategyMap for generating execution strategies.
    #[cfg(feature = "agent")]
    internal_json_agent: Box<dyn crate::agent::Agent<Output = StrategyMap>>,

    /// Internal string agent for intent generation and redesign decisions.
    /// Output type is String for generating prompts and making decisions.
    #[cfg(feature = "agent")]
    #[allow(dead_code)]
    internal_agent: Box<dyn crate::agent::Agent<Output = String>>,

    /// The currently active execution strategy.
    /// None until first generation.
    strategy: Option<StrategyMap>,

    /// Agent registry
    agents: HashMap<String, Arc<dyn DynamicAgent + Send + Sync>>,

    /// Configuration
    #[allow(dead_code)] // TODO: Will be used for concurrency limiting and timeouts in Phase 5
    config: ParallelOrchestratorConfig,
}

impl ParallelOrchestrator {
    /// Creates a new ParallelOrchestrator with a given blueprint.
    ///
    /// Uses default internal agents (ClaudeCodeAgent and ClaudeCodeJsonAgent).
    /// Both internal agents are automatically wrapped with RetryAgent (max 3 retries)
    /// to ensure robustness in strategy generation and redesign decisions.
    /// InnerValidatorAgent is automatically registered as a fallback validator.
    ///
    /// # Arguments
    ///
    /// * `blueprint` - The workflow blueprint
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use llm_toolkit::orchestrator::{ParallelOrchestrator, BlueprintWorkflow};
    ///
    /// let blueprint = BlueprintWorkflow::new("My workflow")
    ///     .add_step(/* ... */);
    ///
    /// let mut orchestrator = ParallelOrchestrator::new(blueprint);
    /// ```
    #[cfg(feature = "agent")]
    pub fn new(blueprint: crate::orchestrator::BlueprintWorkflow) -> Self {
        use crate::agent::impls::RetryAgent;
        use crate::agent::impls::claude_code::{ClaudeCodeAgent, ClaudeCodeJsonAgent};

        Self {
            blueprint,
            agents: HashMap::new(),
            internal_json_agent: Box::new(RetryAgent::new(ClaudeCodeJsonAgent::new(), 3)),
            internal_agent: Box::new(RetryAgent::new(ClaudeCodeAgent::new(), 3)),
            strategy: None,
            config: ParallelOrchestratorConfig::default(),
        }
    }

    /// Creates a new ParallelOrchestrator with custom internal agents.
    ///
    /// This allows you to inject mock or alternative agents for testing or custom LLM backends.
    /// **IMPORTANT**: For production use, **wrap your agents with RetryAgent** before passing them
    /// to ensure robustness in strategy generation and redesign decisions.
    ///
    /// # Arguments
    ///
    /// * `blueprint` - The workflow blueprint
    /// * `internal_agent` - Agent for string outputs (intent generation, redesign decisions).
    ///   **Recommended**: Wrap with RetryAgent
    /// * `internal_json_agent` - Agent for StrategyMap generation.
    ///   **Recommended**: Wrap with RetryAgent
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use llm_toolkit::orchestrator::{ParallelOrchestrator, BlueprintWorkflow};
    /// use llm_toolkit::agent::impls::{RetryAgent, gemini::GeminiAgent};
    ///
    /// let blueprint = BlueprintWorkflow::new("My workflow");
    ///
    /// // Recommended: Wrap with RetryAgent for robustness
    /// let orchestrator = ParallelOrchestrator::with_internal_agents(
    ///     blueprint,
    ///     Box::new(RetryAgent::new(GeminiAgent::new(), 3)),
    ///     Box::new(RetryAgent::new(GeminiAgent::new(), 3)),
    /// );
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_internal_agents(
        blueprint: crate::orchestrator::BlueprintWorkflow,
        internal_agent: Box<dyn crate::agent::Agent<Output = String>>,
        internal_json_agent: Box<dyn crate::agent::Agent<Output = StrategyMap>>,
    ) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            internal_json_agent,
            internal_agent,
            strategy: None,
            config: ParallelOrchestratorConfig::default(),
        }
    }

    /// Creates a new ParallelOrchestrator without the internal agent (for testing).
    #[cfg(not(feature = "agent"))]
    pub fn new(blueprint: crate::orchestrator::BlueprintWorkflow) -> Self {
        Self {
            blueprint,
            agents: HashMap::new(),
            strategy: None,
            config: ParallelOrchestratorConfig::default(),
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

    /// Sets the strategy directly (for testing purposes).
    ///
    /// This method is intended for tests that need to bypass strategy generation
    /// and provide a pre-constructed StrategyMap.
    ///
    /// **Note**: This is primarily for testing. In production, strategies should be
    /// generated via the `execute` method which calls `generate_strategy` internally.
    #[doc(hidden)]
    pub fn set_strategy(&mut self, strategy: StrategyMap) {
        self.strategy = Some(strategy);
    }

    /// Sets the configuration directly (for testing purposes).
    ///
    /// This method is intended for tests that need to set custom configuration
    /// after orchestrator creation.
    ///
    /// **Note**: This is primarily for testing.
    #[doc(hidden)]
    pub fn set_config(&mut self, config: ParallelOrchestratorConfig) {
        self.config = config;
    }

    /// Generates an execution strategy from the blueprint for the given task.
    ///
    /// This method uses the internal JSON agent to analyze the blueprint and
    /// generate a concrete execution strategy (StrategyMap).
    ///
    /// # Arguments
    ///
    /// * `task` - The specific task/goal to accomplish
    ///
    /// # Returns
    ///
    /// Result containing the generated StrategyMap or an error
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Strategy generation fails
    /// - Generated strategy is invalid
    /// - No agents available
    #[cfg(feature = "agent")]
    async fn generate_strategy(&self, task: &str) -> Result<StrategyMap, OrchestratorError> {
        use crate::orchestrator::prompts::StrategyGenerationRequest;
        use crate::prompt::ToPrompt;

        debug!("Generating strategy for task: {}", task);

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
            None, // No user context for parallel orchestrator initially
        );

        let prompt = request.to_prompt();

        debug!("Strategy generation prompt:\n{}", prompt);

        // Generate strategy via internal agent
        let strategy = self
            .internal_json_agent
            .execute(prompt.into())
            .await
            .map_err(|e| OrchestratorError::StrategyGenerationFailed(e.to_string()))?;

        info!("Generated strategy with {} steps", strategy.steps.len());

        Ok(strategy)
    }

    /// Formats the list of available agents for strategy generation prompt.
    fn format_agent_list(&self) -> String {
        self.agents
            .keys()
            .map(|name| format!("- {}", name))
            .collect::<Vec<_>>()
            .join("\n")
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
    /// * `resume_from` - Optional path to a saved state file to resume from
    /// * `save_state_to` - Optional path to save state after each segment
    ///
    /// # Returns
    ///
    /// An `ParallelOrchestrationResult` containing execution status and context
    pub async fn execute(
        &mut self,
        task: &str,
        cancellation_token: CancellationToken,
        resume_from: Option<&Path>,
        save_state_to: Option<&Path>,
    ) -> Result<ParallelOrchestrationResult, OrchestratorError> {
        // Generate strategy if not present
        #[cfg(feature = "agent")]
        if self.strategy.is_none() {
            info!("No strategy set, generating from blueprint...");
            let strategy = self.generate_strategy(task).await?;
            self.strategy = Some(strategy);
        }

        // Get strategy reference (must exist at this point)
        let strategy = self
            .strategy
            .as_ref()
            .ok_or_else(OrchestratorError::no_strategy)?;

        let total_steps = strategy.steps.len();

        async {
            info!("Starting parallel orchestration for task: {}", task);

            let (prefix_instructions, truncated_due_to_loop) =
                Self::collect_parallel_prefix(strategy);

            if truncated_due_to_loop {
                debug!(
                    "Loop boundary encountered; limiting parallel execution to {} instruction(s)",
                    prefix_instructions.len()
                );
            }

            let segments = Self::build_segments(&prefix_instructions);

            // Initialize or restore state
            let (shared_context, mut global_exec_state) = if let Some(resume_path) = resume_from {
                // Resume from saved state
                info!("Resuming orchestration from state file: {:?}", resume_path);
                let state_json = tokio::fs::read_to_string(resume_path).await.map_err(|e| {
                    OrchestratorError::ExecutionFailed(format!(
                        "Failed to read resume state file: {}",
                        e
                    ))
                })?;

                let state: OrchestrationState = serde_json::from_str(&state_json).map_err(|e| {
                    OrchestratorError::ExecutionFailed(format!(
                        "Failed to deserialize state: {}",
                        e
                    ))
                })?;

                (Arc::new(Mutex::new(state.context)), state.execution_manager)
            } else {
                // Start fresh
                let context = Arc::new(Mutex::new(HashMap::new()));
                {
                    let mut ctx = context.lock().await;
                    ctx.insert("task".to_string(), JsonValue::String(task.to_string()));
                }
                (context, ExecutionStateManager::new())
            };

            let mut steps_executed_total = 0usize;
            let mut steps_skipped_total = 0usize;

            for (segment_index, segment) in segments.iter().enumerate() {
                if !segment.steps.is_empty() {
                    // Check if all steps in this segment are already completed (when resuming)
                    let all_completed = segment.steps.iter().all(|step| {
                        matches!(
                            global_exec_state.get_state(&step.step_id),
                            Some(StepState::Completed)
                        )
                    });

                    if all_completed {
                        info!(
                            "Segment {} already completed, skipping execution",
                            segment_index
                        );
                        continue;
                    }

                    let segment_result = self
                        .execute_segment(
                            segment,
                            Arc::clone(&shared_context),
                            cancellation_token.clone(),
                            Some(&global_exec_state),
                        )
                        .await?;

                    steps_executed_total += segment_result.steps_executed;
                    steps_skipped_total += segment_result.exec_state.get_skipped_steps().len();

                    // Merge segment state into global state
                    for step in &segment.steps {
                        if let Some(state) = segment_result.exec_state.get_state(&step.step_id) {
                            global_exec_state.set_state(&step.step_id, state.clone());
                        }
                    }

                    // Check for paused steps (HIL)
                    for step in &segment.steps {
                        if let Some(StepState::PausedForApproval { message, .. }) =
                            global_exec_state.get_state(&step.step_id)
                        {
                            info!(
                                "Execution paused at step {} for human approval",
                                step.step_id
                            );
                            let final_context = shared_context.lock().await.clone();
                            steps_skipped_total +=
                                Self::count_steps_in_segments(&segments, segment_index + 1);

                            // Save state before returning if requested
                            if let Some(save_path) = save_state_to {
                                let state = OrchestrationState {
                                    context: final_context.clone(),
                                    execution_manager: global_exec_state.clone(),
                                };

                                let state_json =
                                    serde_json::to_string_pretty(&state).map_err(|e| {
                                        OrchestratorError::ExecutionFailed(format!(
                                            "Failed to serialize state: {}",
                                            e
                                        ))
                                    })?;

                                tokio::fs::write(save_path, state_json).await.map_err(|e| {
                                    OrchestratorError::ExecutionFailed(format!(
                                        "Failed to write state file: {}",
                                        e
                                    ))
                                })?;

                                debug!("State saved to {:?} before pause", save_path);
                            }

                            return Ok(ParallelOrchestrationResult::paused(
                                steps_executed_total,
                                steps_skipped_total,
                                final_context,
                                message.clone(),
                            ));
                        }
                    }

                    // Save state if requested
                    if let Some(save_path) = save_state_to {
                        let context_snapshot = shared_context.lock().await.clone();
                        let state = OrchestrationState {
                            context: context_snapshot,
                            execution_manager: global_exec_state.clone(),
                        };

                        let state_json = serde_json::to_string_pretty(&state).map_err(|e| {
                            OrchestratorError::ExecutionFailed(format!(
                                "Failed to serialize state: {}",
                                e
                            ))
                        })?;

                        tokio::fs::write(save_path, state_json).await.map_err(|e| {
                            OrchestratorError::ExecutionFailed(format!(
                                "Failed to write state file: {}",
                                e
                            ))
                        })?;

                        debug!("State saved to {:?}", save_path);
                    }

                    if segment_result.exec_state.has_failures() {
                        let final_context = shared_context.lock().await.clone();
                        steps_skipped_total +=
                            Self::count_steps_in_segments(&segments, segment_index + 1);

                        let failed_steps = segment_result.exec_state.get_failed_steps();
                        let mut has_timeout = false;
                        let mut error_details = Vec::new();

                        for (step_id, err) in &failed_steps {
                            if err.contains("timed out") {
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

    /// Executes a wave of independent steps concurrently with retry logic.
    ///
    /// This method wraps `execute_wave_once` and implements retry logic for transient errors.
    /// After each wave execution, failed steps with transient errors are retried up to
    /// `max_step_remediations` times.
    async fn execute_wave(
        &self,
        step_ids: Vec<String>,
        step_lookup: &HashMap<String, StrategyStep>,
        shared_context: Arc<Mutex<HashMap<String, JsonValue>>>,
        cancellation_token: CancellationToken,
    ) -> Vec<(String, Result<crate::agent::AgentOutput, OrchestratorError>)> {
        use std::collections::HashMap as StdHashMap;

        let max_retries = self.config.max_step_remediations;
        let mut retry_counts: StdHashMap<String, usize> = StdHashMap::new();
        let mut current_step_ids = step_ids;
        let mut final_results: StdHashMap<
            String,
            Result<crate::agent::AgentOutput, OrchestratorError>,
        > = StdHashMap::new();

        loop {
            // Execute current wave
            let wave_results = self
                .execute_wave_once(
                    current_step_ids.clone(),
                    step_lookup,
                    Arc::clone(&shared_context),
                    cancellation_token.clone(),
                )
                .await;

            // Classify results: successes and retriable failures
            let mut failed_steps_to_retry = Vec::new();

            for (step_id, result) in wave_results {
                match result {
                    Ok(output) => {
                        // Success - store and done
                        final_results.insert(step_id, Ok(output));
                    }
                    Err(ref err) => {
                        // Check if error is transient and we haven't exceeded retry limit
                        let is_transient = matches!(err, OrchestratorError::AgentError(agent_err) if agent_err.is_transient());
                        let current_retries = retry_counts.get(&step_id).copied().unwrap_or(0);

                        if is_transient && current_retries < max_retries {
                            // Retry this step
                            debug!(
                                step_id = %step_id,
                                retry_count = current_retries,
                                max_retries = max_retries,
                                error = %err,
                                "Step failed with transient error, will retry"
                            );
                            retry_counts.insert(step_id.clone(), current_retries + 1);
                            failed_steps_to_retry.push(step_id);
                        } else {
                            // Non-transient error or max retries exceeded - final failure
                            if is_transient {
                                warn!(
                                    step_id = %step_id,
                                    retry_count = current_retries,
                                    max_retries = max_retries,
                                    "Step exceeded maximum retry attempts"
                                );
                            }
                            final_results.insert(step_id, result);
                        }
                    }
                }
            }

            // If no steps to retry, we're done
            if failed_steps_to_retry.is_empty() {
                break;
            }

            // Prepare next retry wave
            current_step_ids = failed_steps_to_retry;
            info!("Retrying {} failed steps", current_step_ids.len());
        }

        // Convert HashMap back to Vec for return
        final_results.into_iter().collect()
    }

    /// Executes a wave of independent steps concurrently (single attempt, no retry).
    async fn execute_wave_once(
        &self,
        step_ids: Vec<String>,
        step_lookup: &HashMap<String, StrategyStep>,
        shared_context: Arc<Mutex<HashMap<String, JsonValue>>>,
        cancellation_token: CancellationToken,
    ) -> Vec<(String, Result<crate::agent::AgentOutput, OrchestratorError>)> {
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
                                    Ok(Ok(agent_output)) => Ok(agent_output),
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
        initial_exec_state: Option<&ExecutionStateManager>,
    ) -> Result<SegmentOutcome, OrchestratorError> {
        if segment.steps.is_empty() {
            return Ok(SegmentOutcome {
                exec_state: ExecutionStateManager::new(),
                steps_executed: 0,
            });
        }

        let strategy = self.strategy.as_ref().ok_or_else(|| {
            OrchestratorError::ExecutionFailed("Strategy not available".to_string())
        })?;
        let mut subset_strategy = StrategyMap::new(strategy.goal.clone());
        subset_strategy.steps = segment.steps.clone();

        let dep_graph = build_dependency_graph(&subset_strategy)?;
        let mut exec_state = ExecutionStateManager::new();

        // Initialize step states, preserving completed/paused states from resume
        for step in &segment.steps {
            if let Some(initial_state) = initial_exec_state
                && let Some(saved_state) = initial_state.get_state(&step.step_id)
            {
                // If the step was already completed or paused, preserve that state
                match saved_state {
                    StepState::Completed | StepState::PausedForApproval { .. } => {
                        exec_state.set_state(&step.step_id, saved_state.clone());
                        continue;
                    }
                    _ => {}
                }
            }

            // Default: mark as Pending
            exec_state.set_state(&step.step_id, StepState::Pending);
        }

        let step_lookup = Self::create_step_lookup(&segment.steps);

        for step_id in dep_graph.get_zero_dependency_steps() {
            // Only mark as Ready if not already Completed or PausedForApproval
            if !matches!(
                exec_state.get_state(&step_id),
                Some(StepState::Completed) | Some(StepState::PausedForApproval { .. })
            ) {
                exec_state.set_state(&step_id, StepState::Ready);
                debug!(step_id = %step_id, "Step marked as Ready (no dependencies)");
            }
        }

        // Unlock dependents for already-completed steps (when resuming)
        for step in &segment.steps {
            if matches!(
                exec_state.get_state(&step.step_id),
                Some(StepState::Completed)
            ) {
                debug!(step_id = %step.step_id, "Unlocking dependents of already-completed step");
                self.unlock_dependents(&step.step_id, &dep_graph, &mut exec_state);
            }
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
                .execute_wave(
                    ready_steps,
                    &step_lookup,
                    Arc::clone(&shared_context),
                    cancellation_token.clone(),
                )
                .await;

            // Process results
            for (step_id, result) in results {
                match result {
                    Ok(agent_output) => {
                        match agent_output {
                            crate::agent::AgentOutput::Success(value) => {
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
                                        ctx.insert(output_key, value);
                                    }
                                }

                                self.unlock_dependents(&step_id, &dep_graph, &mut exec_state);
                            }
                            crate::agent::AgentOutput::RequiresApproval {
                                message_for_human,
                                current_payload,
                            } => {
                                info!(step_id = %step_id, "Step requires approval");
                                exec_state.set_state(
                                    &step_id,
                                    StepState::PausedForApproval {
                                        message: message_for_human,
                                        payload: current_payload,
                                    },
                                );
                                // Note: We do NOT call cascade_skipped here, as this is not a failure
                            }
                        }
                    }
                    Err(e) => {
                        warn!(step_id = %step_id, error = %e, "Step failed");
                        exec_state.set_state(&step_id, StepState::Failed(e.to_string()));
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
    // We'll add tests incrementally
    // TODO: Fix this test - signature changed
    // #[test]
    // fn test_orchestrator_creation() {
    //     let strategy = StrategyMap::new("Test".to_string());
    //     let orchestrator = ParallelOrchestrator::new(strategy);
    //
    //     assert_eq!(orchestrator.agents.len(), 0);
    // }
}
