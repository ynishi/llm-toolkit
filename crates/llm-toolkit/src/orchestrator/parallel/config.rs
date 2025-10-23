//! Configuration for parallel orchestrator execution.
//!
//! This module provides configuration options for controlling concurrency,
//! timeouts, and other execution parameters.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for parallel orchestrator execution.
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::orchestrator::parallel::ParallelOrchestratorConfig;
/// use std::time::Duration;
///
/// let config = ParallelOrchestratorConfig::new()
///     .with_max_concurrent_tasks(5)
///     .with_step_timeout(Duration::from_secs(300))
///     .with_max_step_remediations(3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelOrchestratorConfig {
    /// Maximum number of steps that can execute concurrently.
    ///
    /// If `None`, no limit is applied and all ready tasks in a wave
    /// can execute concurrently.
    ///
    /// Use this to:
    /// - Control resource usage (CPU, memory, API rate limits)
    /// - Comply with external API rate limits
    /// - Prevent overwhelming the system
    pub max_concurrent_tasks: Option<usize>,

    /// Timeout for individual step execution.
    ///
    /// If `None`, no timeout is applied.
    ///
    /// Use this to:
    /// - Prevent stalled steps from blocking the workflow indefinitely
    /// - Set bounds on workflow execution time
    /// - Recover resources from slow operations
    pub step_timeout: Option<Duration>,

    /// Maximum number of retry attempts per step.
    ///
    /// When a step fails with a transient error, the orchestrator will
    /// automatically retry it up to this many times.
    ///
    /// **Counting behavior:**
    /// - Each failure increments the step's retry counter
    /// - When counter reaches this limit, the step is marked as failed
    /// - Example: `max_step_remediations = 3` allows 3 total attempts (initial + 2 retries)
    ///
    /// **Default:** 3 (allows initial attempt + 2 retries)
    pub max_step_remediations: usize,

    /// Enable generation of validation steps after each execution step.
    ///
    /// When enabled, the orchestrator will generate validation steps that verify
    /// the output of execution steps, improving reliability and error detection.
    /// This provides:
    /// - **Quality assurance:** Automatic validation of step outputs
    /// - **Early error detection:** Problems caught before downstream steps
    /// - **Better debugging:** Clear validation failures vs execution failures
    ///
    /// **When to disable:**
    /// - Performance-critical scenarios where validation overhead is unacceptable
    /// - When agents already include comprehensive internal validation
    /// - Testing scenarios where you want to observe raw execution behavior
    ///
    /// **Default:** `true` (enabled for better reliability)
    #[serde(default = "default_true")]
    pub enable_validation: bool,
}

/// Helper function for serde default value of `true`.
fn default_true() -> bool {
    true
}

impl Default for ParallelOrchestratorConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelOrchestratorConfig {
    /// Creates a new configuration with default values.
    ///
    /// Default values:
    /// - `max_concurrent_tasks`: `None` (unlimited)
    /// - `step_timeout`: `None` (no timeout)
    /// - `max_step_remediations`: `3` (initial attempt + 2 retries)
    /// - `enable_validation`: `true` (enabled for better reliability)
    pub fn new() -> Self {
        Self {
            max_concurrent_tasks: None,
            step_timeout: None,
            max_step_remediations: 3,
            enable_validation: true,
        }
    }

    /// Sets the maximum number of concurrent tasks.
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum number of steps that can run in parallel
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = ParallelOrchestratorConfig::new()
    ///     .with_max_concurrent_tasks(5);
    /// ```
    pub fn with_max_concurrent_tasks(mut self, max: usize) -> Self {
        self.max_concurrent_tasks = Some(max);
        self
    }

    /// Sets the per-step timeout duration.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum duration a single step can run
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::time::Duration;
    ///
    /// let config = ParallelOrchestratorConfig::new()
    ///     .with_step_timeout(Duration::from_secs(300)); // 5 minutes
    /// ```
    pub fn with_step_timeout(mut self, timeout: Duration) -> Self {
        self.step_timeout = Some(timeout);
        self
    }

    /// Removes the concurrency limit, allowing unlimited parallel tasks.
    pub fn with_unlimited_concurrency(mut self) -> Self {
        self.max_concurrent_tasks = None;
        self
    }

    /// Removes the step timeout.
    pub fn with_no_timeout(mut self) -> Self {
        self.step_timeout = None;
        self
    }

    /// Sets the maximum number of retry attempts per step.
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum number of retry attempts (e.g., 3 = initial + 2 retries)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = ParallelOrchestratorConfig::new()
    ///     .with_max_step_remediations(5); // Allow 5 attempts total
    /// ```
    pub fn with_max_step_remediations(mut self, max: usize) -> Self {
        self.max_step_remediations = max;
        self
    }

    /// Sets whether validation steps should be generated after execution steps.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable validation step generation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Disable validation for performance-critical scenarios
    /// let config = ParallelOrchestratorConfig::new()
    ///     .with_validation(false);
    /// ```
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParallelOrchestratorConfig::default();
        assert!(config.max_concurrent_tasks.is_none());
        assert!(config.step_timeout.is_none());
    }

    #[test]
    fn test_new_config() {
        let config = ParallelOrchestratorConfig::new();
        assert!(config.max_concurrent_tasks.is_none());
        assert!(config.step_timeout.is_none());
    }

    #[test]
    fn test_with_concurrency_limit() {
        let config = ParallelOrchestratorConfig::new().with_max_concurrent_tasks(5);
        assert_eq!(config.max_concurrent_tasks, Some(5));
    }

    #[test]
    fn test_with_timeout() {
        let timeout = Duration::from_secs(300);
        let config = ParallelOrchestratorConfig::new().with_step_timeout(timeout);
        assert_eq!(config.step_timeout, Some(timeout));
    }

    #[test]
    fn test_builder_chain() {
        let config = ParallelOrchestratorConfig::new()
            .with_max_concurrent_tasks(10)
            .with_step_timeout(Duration::from_secs(600));

        assert_eq!(config.max_concurrent_tasks, Some(10));
        assert_eq!(config.step_timeout, Some(Duration::from_secs(600)));
    }

    #[test]
    fn test_with_unlimited_concurrency() {
        let config = ParallelOrchestratorConfig::new()
            .with_max_concurrent_tasks(5)
            .with_unlimited_concurrency();

        assert!(config.max_concurrent_tasks.is_none());
    }

    #[test]
    fn test_with_no_timeout() {
        let config = ParallelOrchestratorConfig::new()
            .with_step_timeout(Duration::from_secs(300))
            .with_no_timeout();

        assert!(config.step_timeout.is_none());
    }

    #[test]
    fn test_config_clone() {
        let config1 = ParallelOrchestratorConfig::new()
            .with_max_concurrent_tasks(5)
            .with_step_timeout(Duration::from_secs(300));

        let config2 = config1.clone();

        assert_eq!(config2.max_concurrent_tasks, Some(5));
        assert_eq!(config2.step_timeout, Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_config_debug() {
        let config = ParallelOrchestratorConfig::new().with_max_concurrent_tasks(5);

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("max_concurrent_tasks"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_enable_validation_default() {
        let config = ParallelOrchestratorConfig::new();
        assert!(config.enable_validation); // Default is true
    }

    #[test]
    fn test_with_validation() {
        let config = ParallelOrchestratorConfig::new().with_validation(false);
        assert!(!config.enable_validation);

        let config2 = ParallelOrchestratorConfig::new().with_validation(true);
        assert!(config2.enable_validation);
    }

    #[test]
    fn test_validation_in_builder_chain() {
        let config = ParallelOrchestratorConfig::new()
            .with_max_concurrent_tasks(10)
            .with_validation(false)
            .with_step_timeout(Duration::from_secs(600));

        assert_eq!(config.max_concurrent_tasks, Some(10));
        assert!(!config.enable_validation);
        assert_eq!(config.step_timeout, Some(Duration::from_secs(600)));
    }
}
