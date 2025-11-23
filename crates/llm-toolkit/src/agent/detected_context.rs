//! Detected context - Inferred information from execution context
//!
//! This module provides `DetectedContext` which contains information inferred
//! by context detectors from raw `EnvContext`. Multiple layers of detectors
//! can progressively enrich this context.

use super::expertise::RenderContext;
use crate::context::TaskHealth;
use serde::{Deserialize, Serialize};

/// Detected/inferred context from analysis of EnvContext.
///
/// This struct contains information that has been inferred by detectors
/// (rule-based or LLM-based) from raw `EnvContext` and payload contents.
///
/// **IMPORTANT**: This struct wraps `RenderContext`, which is automatically
/// built from detected information. ExpertiseAgent extracts `render` for
/// expertise filtering.
///
/// # Layered Detection Pattern
///
/// ```rust,ignore
/// // Layer 1: Rule-based detection
/// let detected1 = RuleBasedDetector::new()
///     .detect(&payload)  // → DetectedContext { task_health: AtRisk, ... }
///     .await?;
///
/// let payload = payload.push_context(ExecutionContext::Detected(detected1));
///
/// // Layer 2: LLM-based enrichment
/// let detected2 = LLMDetector::new(haiku_agent)
///     .detect(&payload)  // → DetectedContext { user_states: ["confused"], ... }
///     .await?;
///
/// let payload = payload.push_context(ExecutionContext::Detected(detected2));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedContext {
    /// Detected task type (e.g., "security-review", "debug", "code-review")
    pub task_type: Option<String>,

    /// Detected task health status
    pub task_health: Option<TaskHealth>,

    /// Detected user states (e.g., "beginner", "confused", "expert")
    pub user_states: Vec<String>,

    /// Detection confidence scores (0.0 - 1.0)
    pub confidence: Option<ConfidenceScores>,

    /// Source of detection (for debugging/tracing)
    pub detected_by: Vec<String>,

    /// Finalized RenderContext built from detected information.
    ///
    /// This is automatically updated when detected fields are modified.
    /// ExpertiseAgent uses this for expertise filtering.
    pub render: RenderContext,
}

impl DetectedContext {
    /// Creates a new empty detected context.
    pub fn new() -> Self {
        Self {
            task_type: None,
            task_health: None,
            user_states: Vec::new(),
            confidence: None,
            detected_by: Vec::new(),
            render: RenderContext::default(),
        }
    }

    /// Sets the detected task type and updates render context.
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self.update_render();
        self
    }

    /// Sets the detected task health and updates render context.
    pub fn with_task_health(mut self, health: TaskHealth) -> Self {
        self.task_health = Some(health);
        self.update_render();
        self
    }

    /// Adds a detected user state and updates render context.
    pub fn with_user_state(mut self, state: impl Into<String>) -> Self {
        self.user_states.push(state.into());
        self.update_render();
        self
    }

    /// Sets confidence scores.
    pub fn with_confidence(mut self, confidence: ConfidenceScores) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Records the detector that produced this context.
    pub fn detected_by(mut self, detector_name: impl Into<String>) -> Self {
        self.detected_by.push(detector_name.into());
        self
    }

    /// Updates the internal RenderContext based on detected information.
    ///
    /// This is called automatically by `with_*` methods.
    fn update_render(&mut self) {
        let mut render = RenderContext::default();

        if let Some(task_type) = &self.task_type {
            render = render.with_task_type(task_type.clone());
        }

        if let Some(health) = self.task_health {
            render = render.with_task_health(health);
        }

        for state in &self.user_states {
            render = render.with_user_state(state.clone());
        }

        self.render = render;
    }

    /// Extracts the RenderContext for expertise filtering.
    ///
    /// This is what ExpertiseAgent uses.
    pub fn to_render_context(&self) -> RenderContext {
        self.render.clone()
    }

    /// Merges another detected context into this one.
    ///
    /// This is useful for layered detection where each layer enriches
    /// the context progressively.
    ///
    /// # Merge Strategy
    ///
    /// - `task_type`: Keep existing if present, otherwise use other
    /// - `task_health`: Use other if present (later layers can refine)
    /// - `user_states`: Append from other (deduplicated)
    /// - `confidence`: Merge scores
    /// - `detected_by`: Append detector names
    /// - `render`: Automatically rebuilt from merged detected info
    pub fn merge(mut self, other: DetectedContext) -> Self {
        // task_type: keep existing if present
        if self.task_type.is_none() {
            self.task_type = other.task_type;
        }

        // task_health: use other if present (refinement)
        if other.task_health.is_some() {
            self.task_health = other.task_health;
        }

        // user_states: append and deduplicate
        for state in other.user_states {
            if !self.user_states.contains(&state) {
                self.user_states.push(state);
            }
        }

        // confidence: merge
        if let Some(other_conf) = other.confidence {
            self.confidence = Some(if let Some(existing) = self.confidence {
                existing.merge(other_conf)
            } else {
                other_conf
            });
        }

        // detected_by: append
        self.detected_by.extend(other.detected_by);

        // Update render context based on merged info
        self.update_render();

        self
    }
}

impl Default for DetectedContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Confidence scores for detected information.
///
/// Useful for tracking detection quality, especially when using
/// LLM-based detectors or complex heuristics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceScores {
    /// Confidence in task_type detection (0.0 - 1.0)
    pub task_type: Option<f64>,

    /// Confidence in task_health detection (0.0 - 1.0)
    pub task_health: Option<f64>,

    /// Confidence in user_states detection (0.0 - 1.0)
    pub user_states: Option<f64>,
}

impl ConfidenceScores {
    /// Creates a new confidence scores struct.
    pub fn new() -> Self {
        Self {
            task_type: None,
            task_health: None,
            user_states: None,
        }
    }

    /// Sets task_type confidence.
    pub fn with_task_type(mut self, score: f64) -> Self {
        self.task_type = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Sets task_health confidence.
    pub fn with_task_health(mut self, score: f64) -> Self {
        self.task_health = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Sets user_states confidence.
    pub fn with_user_states(mut self, score: f64) -> Self {
        self.user_states = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Merges with another confidence scores, taking the maximum.
    pub fn merge(self, other: ConfidenceScores) -> Self {
        Self {
            task_type: Self::max_optional(self.task_type, other.task_type),
            task_health: Self::max_optional(self.task_health, other.task_health),
            user_states: Self::max_optional(self.user_states, other.user_states),
        }
    }

    fn max_optional(a: Option<f64>, b: Option<f64>) -> Option<f64> {
        match (a, b) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        }
    }
}

impl Default for ConfidenceScores {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detected_context_builder() {
        let ctx = DetectedContext::new()
            .with_task_type("security-review")
            .with_task_health(TaskHealth::AtRisk)
            .with_user_state("beginner")
            .detected_by("RuleBasedDetector");

        assert_eq!(ctx.task_type, Some("security-review".to_string()));
        assert_eq!(ctx.task_health, Some(TaskHealth::AtRisk));
        assert_eq!(ctx.user_states, vec!["beginner"]);
        assert_eq!(ctx.detected_by, vec!["RuleBasedDetector"]);
    }

    #[test]
    fn test_detected_context_merge() {
        let ctx1 = DetectedContext::new()
            .with_task_type("debug")
            .with_user_state("beginner")
            .detected_by("Layer1");

        let ctx2 = DetectedContext::new()
            .with_task_health(TaskHealth::AtRisk)
            .with_user_state("confused")
            .detected_by("Layer2");

        let merged = ctx1.merge(ctx2);

        assert_eq!(merged.task_type, Some("debug".to_string()));
        assert_eq!(merged.task_health, Some(TaskHealth::AtRisk));
        assert_eq!(merged.user_states, vec!["beginner", "confused"]);
        assert_eq!(merged.detected_by, vec!["Layer1", "Layer2"]);
    }

    #[test]
    fn test_detected_context_merge_dedup_user_states() {
        let ctx1 = DetectedContext::new()
            .with_user_state("beginner")
            .with_user_state("confused");

        let ctx2 = DetectedContext::new()
            .with_user_state("beginner") // duplicate
            .with_user_state("expert");

        let merged = ctx1.merge(ctx2);

        assert_eq!(merged.user_states, vec!["beginner", "confused", "expert"]);
    }

    #[test]
    fn test_confidence_scores() {
        let conf = ConfidenceScores::new()
            .with_task_type(0.9)
            .with_task_health(0.8);

        assert_eq!(conf.task_type, Some(0.9));
        assert_eq!(conf.task_health, Some(0.8));
        assert_eq!(conf.user_states, None);
    }

    #[test]
    fn test_confidence_scores_merge() {
        let conf1 = ConfidenceScores::new()
            .with_task_type(0.7)
            .with_task_health(0.5);

        let conf2 = ConfidenceScores::new()
            .with_task_type(0.9)
            .with_user_states(0.8);

        let merged = conf1.merge(conf2);

        assert_eq!(merged.task_type, Some(0.9)); // max
        assert_eq!(merged.task_health, Some(0.5));
        assert_eq!(merged.user_states, Some(0.8));
    }

    #[test]
    fn test_confidence_scores_clamp() {
        let conf = ConfidenceScores::new()
            .with_task_type(1.5) // > 1.0
            .with_task_health(-0.2); // < 0.0

        assert_eq!(conf.task_type, Some(1.0));
        assert_eq!(conf.task_health, Some(0.0));
    }
}
