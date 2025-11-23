//! Unified execution context - Timeline-based context management
//!
//! This module provides `ExecutionContext` enum which unifies environment context
//! (from orchestrator/external) and detected context (from detectors) into a
//! single timeline-based structure.

use super::detected_context::DetectedContext;
use super::env_context::EnvContext;
use serde::{Deserialize, Serialize};

/// Unified execution context for timeline-based context management.
///
/// This enum wraps both environment context (raw from orchestrator/external)
/// and detected context (inferred by detectors) into a single structure.
///
/// Payloads store a `Vec<ExecutionContext>` to maintain a full timeline
/// of context evolution:
///
/// ```rust,ignore
/// // Timeline example:
/// contexts: vec![
///     ExecutionContext::Env(env_ctx1),          // Initial from Orc
///     ExecutionContext::Detected(detected1),     // Layer 1 detection
///     ExecutionContext::Detected(detected2),     // Layer 2 enrichment
///     ExecutionContext::Env(env_ctx2),          // Updated from Orc
///     ExecutionContext::Detected(detected3),     // Re-detection
/// ]
/// ```
///
/// # Usage Pattern
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Payload, ExecutionContext, EnvContext, DetectedContext};
///
/// // From orchestrator
/// let env_ctx = EnvContext::new().with_redesign_count(3);
/// let payload = Payload::text("Review code")
///     .push_context(ExecutionContext::Env(env_ctx));
///
/// // From detector (Layer 1)
/// let detected1 = DetectedContext::new()
///     .with_task_health(TaskHealth::AtRisk)
///     .detected_by("RuleDetector");
/// let payload = payload.push_context(ExecutionContext::Detected(detected1));
///
/// // From detector (Layer 2)
/// let detected2 = DetectedContext::new()
///     .with_user_state("confused")
///     .detected_by("LLMDetector");
/// let payload = payload.push_context(ExecutionContext::Detected(detected2));
///
/// // Access latest detected context
/// if let Some(detected) = payload.latest_detected_context() {
///     let render = detected.to_render_context();
///     // Use for expertise filtering...
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionContext {
    /// Environment context from orchestrator or external sources.
    ///
    /// Contains factual runtime information like step info, journal summary,
    /// and redesign count.
    Env(EnvContext),

    /// Detected context from detector analysis.
    ///
    /// Contains inferred information like task_type, task_health, user_states,
    /// and the finalized RenderContext.
    Detected(DetectedContext),
}

impl ExecutionContext {
    /// Returns true if this is an Env variant.
    pub fn is_env(&self) -> bool {
        matches!(self, ExecutionContext::Env(_))
    }

    /// Returns true if this is a Detected variant.
    pub fn is_detected(&self) -> bool {
        matches!(self, ExecutionContext::Detected(_))
    }

    /// Attempts to extract EnvContext reference.
    pub fn as_env(&self) -> Option<&EnvContext> {
        match self {
            ExecutionContext::Env(env) => Some(env),
            _ => None,
        }
    }

    /// Attempts to extract DetectedContext reference.
    pub fn as_detected(&self) -> Option<&DetectedContext> {
        match self {
            ExecutionContext::Detected(detected) => Some(detected),
            _ => None,
        }
    }

    /// Consumes self and attempts to extract EnvContext.
    pub fn into_env(self) -> Option<EnvContext> {
        match self {
            ExecutionContext::Env(env) => Some(env),
            _ => None,
        }
    }

    /// Consumes self and attempts to extract DetectedContext.
    pub fn into_detected(self) -> Option<DetectedContext> {
        match self {
            ExecutionContext::Detected(detected) => Some(detected),
            _ => None,
        }
    }
}

/// Helper extension for Vec<ExecutionContext> to extract latest contexts.
pub trait ExecutionContextExt {
    /// Returns the latest EnvContext in the timeline.
    fn latest_env(&self) -> Option<&EnvContext>;

    /// Returns the latest DetectedContext in the timeline.
    fn latest_detected(&self) -> Option<&DetectedContext>;

    /// Returns all EnvContexts in chronological order.
    fn all_envs(&self) -> Vec<&EnvContext>;

    /// Returns all DetectedContexts in chronological order.
    fn all_detected(&self) -> Vec<&DetectedContext>;
}

impl ExecutionContextExt for Vec<ExecutionContext> {
    fn latest_env(&self) -> Option<&EnvContext> {
        self.iter().rev().find_map(|ctx| ctx.as_env())
    }

    fn latest_detected(&self) -> Option<&DetectedContext> {
        self.iter().rev().find_map(|ctx| ctx.as_detected())
    }

    fn all_envs(&self) -> Vec<&EnvContext> {
        self.iter().filter_map(|ctx| ctx.as_env()).collect()
    }

    fn all_detected(&self) -> Vec<&DetectedContext> {
        self.iter().filter_map(|ctx| ctx.as_detected()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::env_context::StepInfo;
    use crate::context::TaskHealth;

    #[test]
    fn test_execution_context_variants() {
        let env = EnvContext::new().with_redesign_count(2);
        let detected = DetectedContext::new()
            .with_task_health(TaskHealth::AtRisk)
            .detected_by("Test");

        let ctx_env = ExecutionContext::Env(env.clone());
        let ctx_detected = ExecutionContext::Detected(detected.clone());

        assert!(ctx_env.is_env());
        assert!(!ctx_env.is_detected());
        assert!(ctx_detected.is_detected());
        assert!(!ctx_detected.is_env());

        assert_eq!(ctx_env.as_env(), Some(&env));
        assert_eq!(ctx_detected.as_detected(), Some(&detected));
    }

    #[test]
    fn test_execution_context_ext() {
        let contexts = vec![
            ExecutionContext::Env(EnvContext::new().with_redesign_count(1)),
            ExecutionContext::Detected(
                DetectedContext::new()
                    .with_task_health(TaskHealth::AtRisk)
                    .detected_by("Layer1"),
            ),
            ExecutionContext::Env(EnvContext::new().with_redesign_count(3)),
            ExecutionContext::Detected(
                DetectedContext::new()
                    .with_user_state("confused")
                    .detected_by("Layer2"),
            ),
        ];

        assert_eq!(contexts.latest_env().unwrap().redesign_count, 3);
        assert_eq!(
            contexts.latest_detected().unwrap().detected_by,
            vec!["Layer2"]
        );

        assert_eq!(contexts.all_envs().len(), 2);
        assert_eq!(contexts.all_detected().len(), 2);
    }

    #[test]
    fn test_timeline_chronological_order() {
        let contexts = vec![
            ExecutionContext::Env(
                EnvContext::new().with_step_info(StepInfo::new("step_1", "First", "Agent1")),
            ),
            ExecutionContext::Detected(DetectedContext::new().detected_by("Detector1")),
            ExecutionContext::Detected(DetectedContext::new().detected_by("Detector2")),
        ];

        let detected_list = contexts.all_detected();
        assert_eq!(detected_list[0].detected_by, vec!["Detector1"]);
        assert_eq!(detected_list[1].detected_by, vec!["Detector2"]);
    }
}
