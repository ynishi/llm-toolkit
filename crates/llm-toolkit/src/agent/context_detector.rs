//! Context detector - Layered detection of context from execution state
//!
//! This module provides a trait-based abstraction for detecting context information
//! (task_type, task_health, user_states) from Payload and ExecutionContext.
//!
//! # Layered Detection Pattern
//!
//! Detectors can be stacked to progressively enrich context:
//!
//! 1. **Layer 1 (Rule-Based)**: Fast, deterministic detection using heuristics
//! 2. **Layer 2 (LLM-Based)**: Slower, more flexible detection using lightweight LLM
//! 3. **Layer 3 (Specialized)**: Domain-specific detectors
//!
//! Each layer enriches the `DetectedContext` which is then merged into the Payload.

use super::detected_context::DetectedContext;
use super::error::AgentError;
use super::payload::Payload;
use async_trait::async_trait;

/// Trait for context detection from Payload.
///
/// Implementors analyze Payload contents and ExecutionContext to infer
/// higher-level context like task_type, task_health, and user_states.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::{ContextDetector, Payload, DetectedContext};
///
/// struct MyDetector;
///
/// #[async_trait]
/// impl ContextDetector for MyDetector {
///     async fn detect(&self, payload: &Payload) -> Result<DetectedContext, AgentError> {
///         let mut detected = DetectedContext::new();
///
///         // Analyze payload...
///         if let Some(exec_ctx) = payload.execution_context() {
///             if exec_ctx.redesign_count > 2 {
///                 detected = detected.with_task_health(TaskHealth::AtRisk);
///             }
///         }
///
///         Ok(detected.detected_by("MyDetector"))
///     }
/// }
/// ```
#[async_trait]
pub trait ContextDetector: Send + Sync {
    /// Detects context from the given payload.
    ///
    /// Returns a `DetectedContext` containing inferred information.
    /// The detector should add its name using `detected_by()`.
    async fn detect(&self, payload: &Payload) -> Result<DetectedContext, AgentError>;

    /// Returns the name of this detector (for debugging/tracing).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Helper extension trait for easy payload enrichment with detectors.
///
/// This allows chaining detectors fluently:
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Payload, DetectContextExt};
///
/// let payload = Payload::text("Review code")
///     .detect_with(&rule_detector).await?
///     .detect_with(&llm_detector).await?;
/// ```
#[async_trait]
pub trait DetectContextExt: Sized {
    /// Runs a detector on this payload and merges the result.
    ///
    /// If the payload already has a `DetectedContext`, the new detection
    /// is merged using `DetectedContext::merge()`.
    async fn detect_with<D: ContextDetector>(self, detector: &D) -> Result<Self, AgentError>;
}

#[async_trait]
impl DetectContextExt for Payload {
    async fn detect_with<D: ContextDetector>(self, detector: &D) -> Result<Self, AgentError> {
        let detected = detector.detect(&self).await?;
        Ok(self.merge_detected_context(detected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::EnvContext;
    use crate::context::TaskHealth;

    // Mock detector for testing
    struct MockDetector {
        should_detect_at_risk: bool,
    }

    #[async_trait]
    impl ContextDetector for MockDetector {
        async fn detect(&self, payload: &Payload) -> Result<DetectedContext, AgentError> {
            let mut detected = DetectedContext::new();

            if let Some(env_ctx) = payload.latest_env_context()
                && self.should_detect_at_risk
                && env_ctx.redesign_count > 2
            {
                detected = detected.with_task_health(TaskHealth::AtRisk);
            }

            Ok(detected.detected_by("MockDetector"))
        }

        fn name(&self) -> &str {
            "MockDetector"
        }
    }

    #[tokio::test]
    async fn test_context_detector_basic() {
        let env_ctx = EnvContext::new().with_redesign_count(3);

        let payload = Payload::text("Test").with_env_context(env_ctx);

        let detector = MockDetector {
            should_detect_at_risk: true,
        };

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
        assert_eq!(detected.detected_by, vec!["MockDetector"]);
    }

    #[tokio::test]
    async fn test_detect_context_ext() {
        let env_ctx = EnvContext::new().with_redesign_count(3);

        let detector = MockDetector {
            should_detect_at_risk: true,
        };

        let payload = Payload::text("Test")
            .with_env_context(env_ctx)
            .detect_with(&detector)
            .await
            .unwrap();

        let detected = payload.detected_context().unwrap();
        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
    }

    #[tokio::test]
    async fn test_layered_detection() {
        let env_ctx = EnvContext::new().with_redesign_count(3);

        let detector1 = MockDetector {
            should_detect_at_risk: true,
        };

        let detector2 = MockDetector {
            should_detect_at_risk: false, // Different behavior
        };

        let payload = Payload::text("Test")
            .with_env_context(env_ctx)
            .detect_with(&detector1)
            .await
            .unwrap()
            .detect_with(&detector2)
            .await
            .unwrap();

        let detected = payload.detected_context().unwrap();
        // detector1's detection should still be present
        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
        // Both detectors recorded
        assert_eq!(detected.detected_by.len(), 2);
    }
}
