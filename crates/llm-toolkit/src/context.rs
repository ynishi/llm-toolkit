//! Context types for agent behavior and expertise activation.
//!
//! This module defines core types for controlling agent behavior based on
//! task context, health status, and priority levels.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Priority: Instruction strength/enforcement level
///
/// Controls how strongly a knowledge fragment should be enforced during
/// prompt generation. Higher priority fragments appear first and are
/// treated as more critical constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Critical: Absolute must-follow (violation = error / strong negative constraint)
    Critical,
    /// High: Recommended/emphasized (explicit instruction)
    High,
    /// Normal: Standard context (general guidance)
    #[default]
    Normal,
    /// Low: Reference information (background info)
    Low,
}

// Custom ordering: Critical > High > Normal > Low
impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl Priority {
    /// Returns a numeric value for ordering (higher = more important)
    pub fn weight(&self) -> u8 {
        match self {
            Priority::Critical => 4,
            Priority::High => 3,
            Priority::Normal => 2,
            Priority::Low => 1,
        }
    }

    /// Returns a display label for visualization
    pub fn label(&self) -> &'static str {
        match self {
            Priority::Critical => "CRITICAL",
            Priority::High => "HIGH",
            Priority::Normal => "NORMAL",
            Priority::Low => "LOW",
        }
    }
}

/// TaskHealth: Task health/quality status indicator
///
/// Represents the current state of a task for triggering adaptive behavior.
/// Enables "gear shifting" based on progress and quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskHealth {
    /// On track: Proceed confidently (Action: Go/SpeedUp)
    OnTrack,

    /// At risk: Proceed cautiously with verification (Action: Review/Clarify)
    AtRisk,

    /// Off track: Stop and reassess (Action: Stop/Reject)
    OffTrack,
}

impl TaskHealth {
    /// Returns a display label for visualization
    pub fn label(&self) -> &'static str {
        match self {
            TaskHealth::OnTrack => "On Track",
            TaskHealth::AtRisk => "At Risk",
            TaskHealth::OffTrack => "Off Track",
        }
    }

    /// Returns an emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            TaskHealth::OnTrack => "✅",
            TaskHealth::AtRisk => "⚠️",
            TaskHealth::OffTrack => "🚫",
        }
    }
}

/// ContextProfile: Activation conditions for knowledge fragments
///
/// Defines when a fragment should be included in the generated prompt
/// based on various contextual factors.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextProfile {
    /// Always active (no conditions)
    #[default]
    Always,

    /// Conditionally active based on context matching
    Conditional {
        /// Task types (e.g., "Debug", "Ideation", "Review")
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        task_types: Vec<String>,

        /// User states (e.g., "Beginner", "Confused", "Expert")
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        user_states: Vec<String>,

        /// Task health status for behavior adjustment
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_health: Option<TaskHealth>,
    },
}

impl ContextProfile {
    /// Check if this context profile matches the given context
    ///
    /// **Note**: This method uses the legacy `ContextMatcher` API.
    /// It will be updated to use `RenderContext` in the next phase.
    pub fn matches(&self, context: &ContextMatcher) -> bool {
        match self {
            ContextProfile::Always => true,
            ContextProfile::Conditional {
                task_types,
                user_states,
                task_health,
            } => {
                let task_match = task_types.is_empty()
                    || context
                        .task_type
                        .as_ref()
                        .map(|t| task_types.contains(t))
                        .unwrap_or(false);

                let user_match = user_states.is_empty()
                    || context
                        .user_state
                        .as_ref()
                        .map(|s| user_states.contains(s))
                        .unwrap_or(false);

                let health_match = task_health
                    .as_ref()
                    .map(|h| context.task_health.as_ref() == Some(h))
                    .unwrap_or(true);

                task_match && user_match && health_match
            }
        }
    }
}

/// ContextMatcher: Runtime context for matching conditional fragments (Legacy)
///
/// **Deprecated**: This type will be removed in favor of `RenderContext`.
/// Used to evaluate whether conditional fragments should be activated.
#[derive(Debug, Clone, Default)]
pub struct ContextMatcher {
    pub task_type: Option<String>,
    pub user_state: Option<String>,
    pub task_health: Option<TaskHealth>,
}

impl ContextMatcher {
    /// Create a new context matcher
    pub fn new() -> Self {
        Self::default()
    }

    /// Set task type
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Set user state
    pub fn with_user_state(mut self, user_state: impl Into<String>) -> Self {
        self.user_state = Some(user_state.into());
        self
    }

    /// Set task health
    pub fn with_task_health(mut self, task_health: TaskHealth) -> Self {
        self.task_health = Some(task_health);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_weight() {
        assert_eq!(Priority::Critical.weight(), 4);
        assert_eq!(Priority::High.weight(), 3);
        assert_eq!(Priority::Normal.weight(), 2);
        assert_eq!(Priority::Low.weight(), 1);
    }

    #[test]
    fn test_context_always_matches() {
        let profile = ContextProfile::Always;
        let context = ContextMatcher::new();
        assert!(profile.matches(&context));
    }

    #[test]
    fn test_context_conditional_matching() {
        let profile = ContextProfile::Conditional {
            task_types: vec!["Debug".to_string()],
            user_states: vec![],
            task_health: Some(TaskHealth::AtRisk),
        };

        let matching_context = ContextMatcher::new()
            .with_task_type("Debug")
            .with_task_health(TaskHealth::AtRisk);

        let non_matching_context = ContextMatcher::new()
            .with_task_type("Review")
            .with_task_health(TaskHealth::OnTrack);

        assert!(profile.matches(&matching_context));
        assert!(!profile.matches(&non_matching_context));
    }
}
