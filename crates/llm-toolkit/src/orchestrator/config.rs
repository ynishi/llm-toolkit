//! Configuration for orchestrator execution behavior.

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
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum number of remediations (redesigns/retries) allowed per step.
    ///
    /// When a step fails, the orchestrator can attempt to fix it through:
    /// - Retry (same step)
    /// - Tactical redesign (modify remaining steps)
    /// - Full regeneration (regenerate entire strategy)
    ///
    /// This limit prevents infinite loops on a single problematic step.
    pub max_step_remediations: usize,

    /// Maximum total number of redesigns allowed for the entire workflow.
    ///
    /// This is a global limit across all steps to prevent runaway execution
    /// and control API costs.
    pub max_total_redesigns: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_step_remediations: 3,
            max_total_redesigns: 10,
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
    }
}
