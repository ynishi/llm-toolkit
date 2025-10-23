//! Configuration for orchestrator execution behavior.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for orchestrator execution behavior.
///
/// This struct centralizes all configuration parameters for the orchestrator,
/// making it easier to manage defaults and customize behavior.
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::orchestrator::OrchestratorConfig;
///
/// // Use default configuration
/// let config = OrchestratorConfig::default();
///
/// // Customize specific values
/// let custom_config = OrchestratorConfig {
///     max_step_remediations: 5,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Maximum number of remediations (redesigns/retries) allowed per step.
    ///
    /// When a step fails, the orchestrator can attempt to fix it through:
    /// - Retry (same step)
    /// - Tactical redesign (modify remaining steps)
    /// - Full regeneration (regenerate entire strategy)
    ///
    /// This limit prevents infinite loops on a single problematic step.
    ///
    /// **Counting behavior:**
    /// - Each failure increments the step's remediation counter
    /// - When counter reaches this limit, orchestrator stops with `MaxStepRemediationsExceeded`
    /// - Example: `max_step_remediations = 3` allows 3 total attempts (initial + 2 retries)
    ///
    /// **Default:** 3 (allows initial attempt + 2 retries)
    pub max_step_remediations: usize,

    /// Maximum total number of redesigns allowed for the entire workflow.
    ///
    /// This is a global limit across all steps to prevent runaway execution
    /// and control API costs.
    ///
    /// **Counting behavior:**
    /// - Initial strategy generation is NOT counted
    /// - Each Retry/TacticalRedesign/FullRegenerate increments the counter
    /// - When counter reaches this limit, orchestrator stops with `MaxTotalRedesignsExceeded`
    /// - Example: `max_total_redesigns = 10` allows 11 total executions (initial + 10 redesigns)
    ///
    /// **Default:** 10 (allows initial strategy + 10 redesigns)
    pub max_total_redesigns: usize,

    /// Minimum interval between step executions.
    ///
    /// This provides proactive rate limiting by introducing a delay after each step execution,
    /// preventing burst API calls that could trigger 429 (Too Many Requests) errors.
    ///
    /// **Why this matters:**
    /// - Each step typically requires 2+ API calls (intent generation + execution)
    /// - Without delays, orchestrators can make 12+ calls in 30 seconds
    /// - Many LLM APIs have strict rate limits (e.g., 10 requests/minute for Gemini)
    ///
    /// **Behavior:**
    /// - Applied after each step completes (before starting next step)
    /// - `Duration::ZERO` means no delay (backward compatible)
    /// - Recommended: `Duration::from_millis(500)` to `Duration::from_secs(1)` for most APIs
    ///
    /// **Example:**
    /// ```ignore
    /// use std::time::Duration;
    /// use llm_toolkit::orchestrator::OrchestratorConfig;
    ///
    /// let config = OrchestratorConfig {
    ///     min_step_interval: Duration::from_millis(500), // 500ms between steps
    ///     ..Default::default()
    /// };
    /// ```
    ///
    /// **Default:** `Duration::ZERO` (no delay)
    pub min_step_interval: Duration,

    /// Enable fast path for intent generation when all placeholders are resolved.
    ///
    /// When enabled, the orchestrator will skip LLM-based intent generation for steps
    /// where all placeholders in the intent template can be resolved directly from context.
    /// This provides:
    /// - **Performance improvement:** Reduces latency from seconds to milliseconds
    /// - **Cost reduction:** Eliminates unnecessary LLM API calls
    /// - **Deterministic behavior:** Template substitution is predictable
    ///
    /// **When fast path is used:**
    /// - All placeholders in `intent_template` have corresponding values in context
    /// - Simple string substitution is sufficient
    ///
    /// **When LLM path is used (default):**
    /// - LLM generates high-quality, context-aware intents
    /// - Agent expertise is considered for prompt optimization
    /// - Better for complex scenarios and thin agent architectures
    ///
    /// **Trade-offs:**
    /// - Fast path: Higher performance, lower quality (simple substitution)
    /// - LLM path: Lower performance, higher quality (semantic understanding)
    ///
    /// **Example:**
    /// ```ignore
    /// use llm_toolkit::orchestrator::OrchestratorConfig;
    ///
    /// // Enable fast path optimization (for thick agents with simple templates)
    /// let config = OrchestratorConfig {
    ///     enable_fast_path_intent_generation: true,
    ///     ..Default::default()
    /// };
    /// ```
    ///
    /// **Default:** `false` (disabled, prioritizes quality for thin agent architectures)
    pub enable_fast_path_intent_generation: bool,

    /// Maximum total number of loop iterations allowed across the entire workflow.
    ///
    /// This is a global safety limit to prevent runaway costs from loop execution,
    /// especially important when loops are controlled by LLM decisions or complex templates.
    ///
    /// **Counting behavior:**
    /// - Each iteration of any loop increments the global counter
    /// - When counter reaches this limit, orchestrator stops with error
    /// - Example: With 2 loops each executing 3 times, total iterations = 6
    ///
    /// **Design constraint:**
    /// - Nested loops are not supported (enforced by `LoopBlock::validate()`)
    /// - Only single-level loops are allowed
    ///
    /// **Example:**
    /// ```ignore
    /// use llm_toolkit::orchestrator::OrchestratorConfig;
    ///
    /// let config = OrchestratorConfig {
    ///     max_total_loop_iterations: 100, // Allow up to 100 loop iterations total
    ///     ..Default::default()
    /// };
    /// ```
    ///
    /// **Default:** 50 (reasonable limit for most workflows)
    pub max_total_loop_iterations: usize,

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

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_step_remediations: 3,
            max_total_redesigns: 10,
            min_step_interval: Duration::ZERO,
            enable_fast_path_intent_generation: false,
            max_total_loop_iterations: 50,
            enable_validation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.max_step_remediations, 3);
        assert_eq!(config.max_total_redesigns, 10);
        assert_eq!(config.min_step_interval, Duration::ZERO);
        assert!(!config.enable_fast_path_intent_generation); // Default is false (quality over performance)
        assert_eq!(config.max_total_loop_iterations, 50);
        assert!(config.enable_validation); // Default is true (enabled for reliability)
    }

    #[test]
    fn test_partial_override() {
        let config = OrchestratorConfig {
            max_step_remediations: 5,
            ..Default::default()
        };
        assert_eq!(config.max_step_remediations, 5);
        assert_eq!(config.max_total_redesigns, 10); // Uses default
    }

    #[test]
    fn test_clone() {
        let config1 = OrchestratorConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.max_step_remediations, config2.max_step_remediations);
        assert_eq!(config1.max_total_redesigns, config2.max_total_redesigns);
        assert_eq!(config1.min_step_interval, config2.min_step_interval);
        assert_eq!(
            config1.enable_fast_path_intent_generation,
            config2.enable_fast_path_intent_generation
        );
        assert_eq!(
            config1.max_total_loop_iterations,
            config2.max_total_loop_iterations
        );
        assert_eq!(config1.enable_validation, config2.enable_validation);
    }

    #[test]
    fn test_min_step_interval_configuration() {
        let config = OrchestratorConfig {
            min_step_interval: Duration::from_millis(500),
            ..Default::default()
        };
        assert_eq!(config.min_step_interval, Duration::from_millis(500));
        assert_eq!(config.max_step_remediations, 3); // Uses default
        assert_eq!(config.max_total_redesigns, 10); // Uses default
    }

    #[test]
    fn test_min_step_interval_zero() {
        let config = OrchestratorConfig::default();
        assert!(config.min_step_interval.is_zero());
    }

    #[test]
    fn test_enable_fast_path_intent_generation_default() {
        let config = OrchestratorConfig::default();
        assert!(!config.enable_fast_path_intent_generation); // Default is false
    }

    #[test]
    fn test_enable_fast_path_intent_generation_override() {
        let config = OrchestratorConfig {
            enable_fast_path_intent_generation: true, // Explicitly enable for performance
            ..Default::default()
        };
        assert!(config.enable_fast_path_intent_generation);
        assert_eq!(config.max_step_remediations, 3); // Uses default
        assert_eq!(config.max_total_redesigns, 10); // Uses default
    }
}
