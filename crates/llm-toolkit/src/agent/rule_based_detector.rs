//! Rule-based context detector - Fast, deterministic context detection
//!
//! This module provides a hardcoded, rule-based detector that uses heuristics
//! to infer context from Payload and EnvContext without requiring LLM calls.

use super::context_detector::ContextDetector;
use super::detected_context::{ConfidenceScores, DetectedContext};
use super::error::AgentError;
use super::payload::{Payload, PayloadContent};
use crate::context::TaskHealth;
use async_trait::async_trait;

/// Rule-based context detector using hardcoded heuristics.
///
/// This detector analyzes EnvContext and Payload contents to infer:
/// - `task_health`: Based on redesign count, failure rate, etc.
/// - `task_type`: Based on keywords in text content
/// - Confidence scores for each detection
///
/// # Design
///
/// - **Fast**: No LLM calls, pure heuristics
/// - **Deterministic**: Same input always produces same output
/// - **Layer 1**: Intended as first detection layer before LLM-based detectors
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Payload, EnvContext, RuleBasedDetector};
/// use llm_toolkit::agent::context_detector::DetectContextExt;
///
/// let env = EnvContext::new().with_redesign_count(3);
/// let payload = Payload::text("Review this security-critical code")
///     .with_env_context(env);
///
/// let detector = RuleBasedDetector::new();
/// let payload = payload.detect_with(&detector).await?;
///
/// // Detected: task_health=AtRisk, task_type="security-review"
/// ```
#[derive(Debug, Clone, Default)]
pub struct RuleBasedDetector {
    /// Minimum redesign count to trigger AtRisk
    pub at_risk_threshold: usize,

    /// Minimum failure rate to trigger AtRisk (0.0 - 1.0)
    pub failure_rate_threshold: f64,
}

impl RuleBasedDetector {
    /// Creates a new rule-based detector with default thresholds.
    pub fn new() -> Self {
        Self {
            at_risk_threshold: 2,
            failure_rate_threshold: 0.4, // 40% failure rate
        }
    }

    /// Creates a detector with custom thresholds.
    pub fn with_thresholds(at_risk_threshold: usize, failure_rate_threshold: f64) -> Self {
        Self {
            at_risk_threshold,
            failure_rate_threshold,
        }
    }

    /// Detects task health from EnvContext.
    fn detect_task_health(&self, payload: &Payload) -> (Option<TaskHealth>, f64) {
        if let Some(env) = payload.latest_env_context() {
            let mut confidence: f64 = 0.0;
            let mut is_at_risk = false;

            // Rule 1: High redesign count
            if env.redesign_count > self.at_risk_threshold {
                is_at_risk = true;
                confidence += 0.4;
            }

            // Rule 2: Low success rate from journal
            if let Some(journal) = &env.journal_summary {
                if journal.success_rate < (1.0 - self.failure_rate_threshold) {
                    is_at_risk = true;
                    confidence += 0.3;
                }

                // Rule 3: Consecutive failures
                if journal.consecutive_failures > 2 {
                    is_at_risk = true;
                    confidence += 0.3;
                }
            }

            let health = if is_at_risk {
                Some(TaskHealth::AtRisk)
            } else {
                Some(TaskHealth::OnTrack)
            };

            (health, confidence.min(1.0))
        } else {
            (None, 0.0)
        }
    }

    /// Detects task type from Payload text content.
    fn detect_task_type(&self, payload: &Payload) -> (Option<String>, f64) {
        let text = payload
            .contents()
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Text(t) => Some(t.as_str()),
                PayloadContent::Message { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        // Keyword-based detection
        let patterns = vec![
            (
                "security-review",
                vec!["security", "vulnerability", "exploit", "auth"],
                0.8,
            ),
            (
                "code-review",
                vec!["review", "pr", "pull request", "refactor"],
                0.7,
            ),
            (
                "debug",
                vec!["debug", "error", "bug", "fix", "crash"],
                0.8,
            ),
            (
                "implementation",
                vec!["implement", "feature", "add", "create"],
                0.6,
            ),
            (
                "test",
                vec!["test", "spec", "coverage"],
                0.7,
            ),
        ];

        for (task_type, keywords, base_confidence) in patterns {
            let matches = keywords
                .iter()
                .filter(|kw| text.contains(*kw))
                .count();

            if matches > 0 {
                let confidence = (matches as f64 / keywords.len() as f64) * base_confidence;
                return (Some(task_type.to_string()), confidence);
            }
        }

        (None, 0.0)
    }
}

#[async_trait]
impl ContextDetector for RuleBasedDetector {
    async fn detect(&self, payload: &Payload) -> Result<DetectedContext, AgentError> {
        let mut detected = DetectedContext::new();

        // Detect task health
        let (health, health_confidence) = self.detect_task_health(payload);
        let has_health = health.is_some();
        if let Some(h) = health {
            detected = detected.with_task_health(h);
        }

        // Detect task type
        let (task_type, type_confidence) = self.detect_task_type(payload);
        let has_task_type = task_type.is_some();
        if let Some(tt) = task_type {
            detected = detected.with_task_type(tt);
        }

        // Add confidence scores
        let mut confidence = ConfidenceScores::new();
        if has_health {
            confidence = confidence.with_task_health(health_confidence);
        }
        if has_task_type {
            confidence = confidence.with_task_type(type_confidence);
        }
        detected = detected.with_confidence(confidence);

        // Mark source
        detected = detected.detected_by("RuleBasedDetector");

        Ok(detected)
    }

    fn name(&self) -> &str {
        "RuleBasedDetector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{EnvContext, JournalSummary};

    #[tokio::test]
    async fn test_detect_task_health_at_risk() {
        let detector = RuleBasedDetector::new();
        let env = EnvContext::new().with_redesign_count(3);
        let payload = Payload::text("Test").with_env_context(env);

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
        assert!(detected.confidence.unwrap().task_health.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_detect_task_health_from_journal() {
        let detector = RuleBasedDetector::new();
        let journal = JournalSummary::new(10, 5).with_consecutive_failures(3);
        let env = EnvContext::new().with_journal_summary(journal);
        let payload = Payload::text("Test").with_env_context(env);

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
    }

    #[tokio::test]
    async fn test_detect_task_type_security() {
        let detector = RuleBasedDetector::new();
        let payload = Payload::text("Review this security-critical authentication code");

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_type, Some("security-review".to_string()));
        assert!(detected.confidence.unwrap().task_type.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_detect_task_type_debug() {
        let detector = RuleBasedDetector::new();
        let payload = Payload::text("Debug this error and fix the bug");

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_type, Some("debug".to_string()));
    }

    #[tokio::test]
    async fn test_detect_combined() {
        let detector = RuleBasedDetector::new();
        let env = EnvContext::new().with_redesign_count(3);
        let payload = Payload::text("Debug this security vulnerability")
            .with_env_context(env);

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
        assert!(detected.task_type.is_some()); // Could be "debug" or "security-review"
        assert_eq!(detected.detected_by, vec!["RuleBasedDetector"]);
    }

    #[tokio::test]
    async fn test_custom_thresholds() {
        let detector = RuleBasedDetector::with_thresholds(5, 0.6);
        let env = EnvContext::new().with_redesign_count(3);
        let payload = Payload::text("Test").with_env_context(env);

        let detected = detector.detect(&payload).await.unwrap();

        // Should be on track with higher threshold
        assert_eq!(detected.task_health, Some(TaskHealth::OnTrack));
    }
}
