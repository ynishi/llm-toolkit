//! Context-aware prompt rendering (Phase 2)
//!
//! This module provides context-aware prompt generation capabilities,
//! allowing dynamic filtering and ordering of knowledge fragments based
//! on runtime context.

use crate::context::{ContextProfile, TaskHealth};

/// Runtime context for prompt rendering
///
/// Encapsulates the current state that determines which knowledge fragments
/// should be included and how they should be prioritized.
///
/// # Examples
///
/// ```
/// use llm_toolkit::agent::expertise::RenderContext;
/// use llm_toolkit::TaskHealth;
///
/// let context = RenderContext::new()
///     .with_task_type("security-review")
///     .with_user_state("beginner")
///     .with_task_health(TaskHealth::AtRisk);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderContext {
    /// Current task type (e.g., "security-review", "code-review", "debug")
    pub task_type: Option<String>,

    /// User states (e.g., "beginner", "expert", "confused")
    pub user_states: Vec<String>,

    /// Current task health status
    pub task_health: Option<TaskHealth>,
}

impl RenderContext {
    /// Create a new empty render context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the task type
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::RenderContext;
    ///
    /// let context = RenderContext::new()
    ///     .with_task_type("security-review");
    /// ```
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Add a user state
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::RenderContext;
    ///
    /// let context = RenderContext::new()
    ///     .with_user_state("beginner")
    ///     .with_user_state("confused");
    /// ```
    pub fn with_user_state(mut self, state: impl Into<String>) -> Self {
        self.user_states.push(state.into());
        self
    }

    /// Set the task health
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::RenderContext;
    /// use llm_toolkit::TaskHealth;
    ///
    /// let context = RenderContext::new()
    ///     .with_task_health(TaskHealth::AtRisk);
    /// ```
    pub fn with_task_health(mut self, health: TaskHealth) -> Self {
        self.task_health = Some(health);
        self
    }

    /// Check if this context matches a ContextProfile
    ///
    /// A context matches a profile if:
    /// - Profile is `Always` → always matches
    /// - Profile is `Conditional`:
    ///   - If `task_types` is non-empty, current task_type must be in the list
    ///   - If `user_states` is non-empty, at least one user_state must match
    ///   - If `task_health` is set, current health must match
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::RenderContext;
    /// use llm_toolkit::{ContextProfile, TaskHealth};
    ///
    /// let context = RenderContext::new()
    ///     .with_task_type("security-review")
    ///     .with_task_health(TaskHealth::AtRisk);
    ///
    /// let profile = ContextProfile::Conditional {
    ///     task_types: vec!["security-review".to_string()],
    ///     user_states: vec![],
    ///     task_health: Some(TaskHealth::AtRisk),
    /// };
    ///
    /// assert!(context.matches(&profile));
    /// ```
    pub fn matches(&self, profile: &ContextProfile) -> bool {
        match profile {
            ContextProfile::Always => true,
            ContextProfile::Conditional {
                task_types,
                user_states,
                task_health,
            } => {
                // Check task_type match
                let task_type_match = if task_types.is_empty() {
                    true // No task type constraint
                } else {
                    self.task_type
                        .as_ref()
                        .map(|tt| task_types.contains(tt))
                        .unwrap_or(false)
                };

                // Check user_state match (at least one must match)
                let user_state_match = if user_states.is_empty() {
                    true // No user state constraint
                } else {
                    self.user_states
                        .iter()
                        .any(|state| user_states.contains(state))
                };

                // Check task_health match
                let task_health_match = if let Some(required_health) = task_health {
                    self.task_health.as_ref() == Some(required_health)
                } else {
                    true // No health constraint
                };

                task_type_match && user_state_match && task_health_match
            }
        }
    }
}

/// Context-aware prompt renderer (Phase 2)
///
/// A wrapper type that combines an `Expertise` with a `RenderContext` to enable
/// context-aware prompt generation. Implements `ToPrompt` for seamless integration
/// with the DTO pattern.
///
/// # Examples
///
/// ## Direct Usage
///
/// ```
/// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
/// use llm_toolkit::agent::expertise::{ContextualPrompt, RenderContext};
/// use llm_toolkit::TaskHealth;
///
/// let expertise = Expertise::new("rust-reviewer", "1.0")
///     .with_fragment(WeightedFragment::new(
///         KnowledgeFragment::Text("Review Rust code".to_string())
///     ));
///
/// let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
///     .with_task_type("security-review")
///     .with_task_health(TaskHealth::AtRisk)
///     .to_prompt();
/// ```
///
/// ## DTO Integration
///
/// ```ignore
/// // With ToPrompt-based DTO pattern:
/// #[derive(ToPrompt)]
/// #[prompt(template = "Knowledge:\n{{expertise}}\n\nTask: {{task}}")]
/// struct RequestDto<'a> {
///     expertise: ContextualPrompt<'a>,
///     task: String,
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ContextualPrompt<'a> {
    expertise: &'a super::Expertise,
    context: RenderContext,
}

impl<'a> ContextualPrompt<'a> {
    /// Create a contextual prompt from expertise and context
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
    /// use llm_toolkit::agent::expertise::{ContextualPrompt, RenderContext};
    ///
    /// let expertise = Expertise::new("test", "1.0")
    ///     .with_fragment(WeightedFragment::new(
    ///         KnowledgeFragment::Text("Test".to_string())
    ///     ));
    ///
    /// let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new());
    /// ```
    pub fn from_expertise(expertise: &'a super::Expertise, context: RenderContext) -> Self {
        Self { expertise, context }
    }

    /// Set the task type
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
    /// use llm_toolkit::agent::expertise::{ContextualPrompt, RenderContext};
    ///
    /// let expertise = Expertise::new("test", "1.0")
    ///     .with_fragment(WeightedFragment::new(
    ///         KnowledgeFragment::Text("Test".to_string())
    ///     ));
    ///
    /// let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
    ///     .with_task_type("security-review");
    /// ```
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.context = self.context.with_task_type(task_type);
        self
    }

    /// Add a user state
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
    /// use llm_toolkit::agent::expertise::{ContextualPrompt, RenderContext};
    ///
    /// let expertise = Expertise::new("test", "1.0")
    ///     .with_fragment(WeightedFragment::new(
    ///         KnowledgeFragment::Text("Test".to_string())
    ///     ));
    ///
    /// let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
    ///     .with_user_state("beginner");
    /// ```
    pub fn with_user_state(mut self, state: impl Into<String>) -> Self {
        self.context = self.context.with_user_state(state);
        self
    }

    /// Set the task health
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
    /// use llm_toolkit::agent::expertise::{ContextualPrompt, RenderContext};
    /// use llm_toolkit::TaskHealth;
    ///
    /// let expertise = Expertise::new("test", "1.0")
    ///     .with_fragment(WeightedFragment::new(
    ///         KnowledgeFragment::Text("Test".to_string())
    ///     ));
    ///
    /// let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
    ///     .with_task_health(TaskHealth::AtRisk);
    /// ```
    pub fn with_task_health(mut self, health: TaskHealth) -> Self {
        self.context = self.context.with_task_health(health);
        self
    }

    /// Render the prompt with context
    ///
    /// This is called automatically when using `ToPrompt` trait.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::agent::expertise::{Expertise, WeightedFragment, KnowledgeFragment};
    /// use llm_toolkit::agent::expertise::{ContextualPrompt, RenderContext};
    ///
    /// let expertise = Expertise::new("test", "1.0")
    ///     .with_fragment(WeightedFragment::new(
    ///         KnowledgeFragment::Text("Test content".to_string())
    ///     ));
    ///
    /// let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
    ///     .to_prompt();
    ///
    /// assert!(prompt.contains("Test content"));
    /// ```
    pub fn to_prompt(&self) -> String {
        self.expertise.to_prompt_with_context(&self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_context_builder() {
        let context = RenderContext::new()
            .with_task_type("security-review")
            .with_user_state("beginner")
            .with_task_health(TaskHealth::AtRisk);

        assert_eq!(context.task_type, Some("security-review".to_string()));
        assert_eq!(context.user_states, vec!["beginner"]);
        assert_eq!(context.task_health, Some(TaskHealth::AtRisk));
    }

    #[test]
    fn test_matches_always() {
        let context = RenderContext::new();
        assert!(context.matches(&ContextProfile::Always));
    }

    #[test]
    fn test_matches_task_type() {
        let context = RenderContext::new().with_task_type("security-review");

        let profile = ContextProfile::Conditional {
            task_types: vec!["security-review".to_string()],
            user_states: vec![],
            task_health: None,
        };

        assert!(context.matches(&profile));

        let wrong_profile = ContextProfile::Conditional {
            task_types: vec!["code-review".to_string()],
            user_states: vec![],
            task_health: None,
        };

        assert!(!context.matches(&wrong_profile));
    }

    #[test]
    fn test_matches_user_state() {
        let context = RenderContext::new()
            .with_user_state("beginner")
            .with_user_state("confused");

        let profile = ContextProfile::Conditional {
            task_types: vec![],
            user_states: vec!["beginner".to_string()],
            task_health: None,
        };

        assert!(context.matches(&profile));

        let profile2 = ContextProfile::Conditional {
            task_types: vec![],
            user_states: vec!["expert".to_string()],
            task_health: None,
        };

        assert!(!context.matches(&profile2));
    }

    #[test]
    fn test_matches_task_health() {
        let context = RenderContext::new().with_task_health(TaskHealth::AtRisk);

        let profile = ContextProfile::Conditional {
            task_types: vec![],
            user_states: vec![],
            task_health: Some(TaskHealth::AtRisk),
        };

        assert!(context.matches(&profile));

        let wrong_profile = ContextProfile::Conditional {
            task_types: vec![],
            user_states: vec![],
            task_health: Some(TaskHealth::OnTrack),
        };

        assert!(!context.matches(&wrong_profile));
    }

    #[test]
    fn test_matches_combined() {
        let context = RenderContext::new()
            .with_task_type("security-review")
            .with_user_state("beginner")
            .with_task_health(TaskHealth::AtRisk);

        let profile = ContextProfile::Conditional {
            task_types: vec!["security-review".to_string()],
            user_states: vec!["beginner".to_string()],
            task_health: Some(TaskHealth::AtRisk),
        };

        assert!(context.matches(&profile));

        // Missing one condition
        let partial_profile = ContextProfile::Conditional {
            task_types: vec!["security-review".to_string()],
            user_states: vec!["expert".to_string()], // Wrong!
            task_health: Some(TaskHealth::AtRisk),
        };

        assert!(!context.matches(&partial_profile));
    }

    #[test]
    fn test_matches_no_constraints() {
        let context = RenderContext::new().with_task_type("anything");

        let profile = ContextProfile::Conditional {
            task_types: vec![],
            user_states: vec![],
            task_health: None,
        };

        assert!(context.matches(&profile));
    }
}
