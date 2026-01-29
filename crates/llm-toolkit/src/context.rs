//! Context types for agent behavior and expertise activation.
//!
//! This module defines core types for controlling agent behavior based on
//! task context, health status, and priority levels.

#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Priority: Instruction strength/enforcement level
///
/// Controls how strongly a knowledge fragment should be enforced during
/// prompt generation. Higher priority fragments appear first and are
/// treated as more critical constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
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
}
