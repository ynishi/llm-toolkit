//! Core expertise types and structures.
//!
//! This module defines the main Expertise type and related structures for
//! composing agent capabilities from weighted knowledge fragments.

use crate::context::{ContextMatcher, ContextProfile, Priority};
use crate::fragment::KnowledgeFragment;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Expertise: Agent capability package (Graph node)
///
/// Represents a complete agent expertise profile composed of weighted
/// knowledge fragments. Uses composition instead of inheritance for
/// flexible capability mixing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Expertise {
    /// Unique identifier
    pub id: String,

    /// Version string
    pub version: String,

    /// Tags for search and grouping (e.g., ["lang:rust", "role:reviewer", "style:friendly"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// Knowledge and capability components (weighted)
    pub content: Vec<WeightedFragment>,
}

impl Expertise {
    /// Create a new expertise profile
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            tags: Vec::new(),
            content: Vec::new(),
        }
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Add a weighted fragment
    pub fn with_fragment(mut self, fragment: WeightedFragment) -> Self {
        self.content.push(fragment);
        self
    }

    /// Generate a single prompt string from all fragments
    ///
    /// Fragments are ordered by priority (Critical → High → Normal → Low)
    pub fn to_prompt(&self) -> String {
        self.to_prompt_with_context(&ContextMatcher::default())
    }

    /// Generate a prompt string with context filtering
    ///
    /// Only includes fragments that match the given context conditions
    pub fn to_prompt_with_context(&self, context: &ContextMatcher) -> String {
        let mut result = format!("# Expertise: {} (v{})\n\n", self.id, self.version);

        if !self.tags.is_empty() {
            result.push_str("**Tags:** ");
            result.push_str(&self.tags.join(", "));
            result.push_str("\n\n");
        }

        result.push_str("---\n\n");

        // Sort fragments by priority (highest first)
        let mut sorted_fragments: Vec<_> = self
            .content
            .iter()
            .filter(|f| f.context.matches(context))
            .collect();
        sorted_fragments.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Group by priority
        let mut current_priority: Option<Priority> = None;
        for weighted in sorted_fragments {
            // Add priority header if changed
            if current_priority != Some(weighted.priority) {
                current_priority = Some(weighted.priority);
                result.push_str(&format!("## Priority: {}\n\n", weighted.priority.label()));
            }

            // Add fragment content
            result.push_str(&weighted.fragment.to_prompt());
            result.push('\n');
        }

        result
    }

    /// Generate a Mermaid graph representation
    pub fn to_mermaid(&self) -> String {
        let mut result = String::from("graph TD\n");

        // Root node (expertise)
        result.push_str(&format!("    ROOT[\"Expertise: {}\"]\n", self.id));

        // Add tag nodes if present
        if !self.tags.is_empty() {
            result.push_str("    TAGS[\"Tags\"]\n");
            result.push_str("    ROOT --> TAGS\n");
            for (i, tag) in self.tags.iter().enumerate() {
                let tag_id = format!("TAG{}", i);
                result.push_str(&format!("    {}[\"{}\"]\n", tag_id, tag));
                result.push_str(&format!("    TAGS --> {}\n", tag_id));
            }
        }

        // Add fragment nodes
        for (i, weighted) in self.content.iter().enumerate() {
            let node_id = format!("F{}", i);
            let summary = weighted.fragment.summary();
            let type_label = weighted.fragment.type_label();

            // Node with priority styling
            let style_class = match weighted.priority {
                Priority::Critical => ":::critical",
                Priority::High => ":::high",
                Priority::Normal => ":::normal",
                Priority::Low => ":::low",
            };

            result.push_str(&format!(
                "    {}[\"{} [{}]: {}\"]{}\n",
                node_id,
                weighted.priority.label(),
                type_label,
                summary,
                style_class
            ));
            result.push_str(&format!("    ROOT --> {}\n", node_id));

            // Add context info if conditional
            if let ContextProfile::Conditional {
                task_types,
                user_states,
                task_health,
            } = &weighted.context
            {
                let context_id = format!("C{}", i);
                let mut context_parts = Vec::new();

                if !task_types.is_empty() {
                    context_parts.push(format!("Tasks: {}", task_types.join(", ")));
                }
                if !user_states.is_empty() {
                    context_parts.push(format!("States: {}", user_states.join(", ")));
                }
                if let Some(health) = task_health {
                    context_parts.push(format!("Health: {}", health.label()));
                }

                if !context_parts.is_empty() {
                    result.push_str(&format!(
                        "    {}[\"Context: {}\"]\n",
                        context_id,
                        context_parts.join("; ")
                    ));
                    result.push_str(&format!("    {} -.-> {}\n", node_id, context_id));
                }
            }
        }

        // Add styling
        result.push_str("\n    classDef critical fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px\n");
        result.push_str("    classDef high fill:#ffd93d,stroke:#f08c00,stroke-width:2px\n");
        result.push_str("    classDef normal fill:#a0e7e5,stroke:#4ecdc4,stroke-width:1px\n");
        result.push_str("    classDef low fill:#e0e0e0,stroke:#999,stroke-width:1px\n");

        result
    }

    /// Generate a simple tree representation
    pub fn to_tree(&self) -> String {
        let mut result = format!("Expertise: {} (v{})\n", self.id, self.version);

        if !self.tags.is_empty() {
            result.push_str(&format!("├─ Tags: {}\n", self.tags.join(", ")));
        }

        result.push_str("└─ Content:\n");

        // Sort by priority
        let mut sorted_fragments: Vec<_> = self.content.iter().collect();
        sorted_fragments.sort_by(|a, b| b.priority.cmp(&a.priority));

        for (i, weighted) in sorted_fragments.iter().enumerate() {
            let is_last = i == sorted_fragments.len() - 1;
            let prefix = if is_last { "   └─" } else { "   ├─" };

            let summary = weighted.fragment.summary();
            let type_label = weighted.fragment.type_label();

            result.push_str(&format!(
                "{} [{}] {}: {}\n",
                prefix,
                weighted.priority.label(),
                type_label,
                summary
            ));

            // Add context info
            if let ContextProfile::Conditional {
                task_types,
                user_states,
                task_health,
            } = &weighted.context
            {
                let sub_prefix = if is_last { "      " } else { "   │  " };
                if !task_types.is_empty() {
                    result.push_str(&format!(
                        "{} └─ Tasks: {}\n",
                        sub_prefix,
                        task_types.join(", ")
                    ));
                }
                if !user_states.is_empty() {
                    result.push_str(&format!(
                        "{} └─ States: {}\n",
                        sub_prefix,
                        user_states.join(", ")
                    ));
                }
                if let Some(health) = task_health {
                    result.push_str(&format!(
                        "{} └─ Health: {} {}\n",
                        sub_prefix,
                        health.emoji(),
                        health.label()
                    ));
                }
            }
        }

        result
    }
}

/// WeightedFragment: Knowledge entity with metadata
///
/// Combines a knowledge fragment with its priority and activation context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WeightedFragment {
    /// Priority: Controls enforcement strength and ordering
    #[serde(default)]
    pub priority: Priority,

    /// Context: Activation conditions
    #[serde(default)]
    pub context: ContextProfile,

    /// Fragment: The actual knowledge content
    pub fragment: KnowledgeFragment,
}

impl WeightedFragment {
    /// Create a new weighted fragment with default priority and always-active context
    pub fn new(fragment: KnowledgeFragment) -> Self {
        Self {
            priority: Priority::default(),
            context: ContextProfile::default(),
            fragment,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set context profile
    pub fn with_context(mut self, context: ContextProfile) -> Self {
        self.context = context;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expertise_builder() {
        let expertise = Expertise::new("test", "1.0")
            .with_tag("test-tag")
            .with_fragment(WeightedFragment::new(KnowledgeFragment::Text(
                "Test content".to_string(),
            )));

        assert_eq!(expertise.id, "test");
        assert_eq!(expertise.version, "1.0");
        assert_eq!(expertise.tags.len(), 1);
        assert_eq!(expertise.content.len(), 1);
    }

    #[test]
    fn test_to_prompt_ordering() {
        let expertise = Expertise::new("test", "1.0")
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("Low priority".to_string()))
                    .with_priority(Priority::Low),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("Critical priority".to_string()))
                    .with_priority(Priority::Critical),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("Normal priority".to_string()))
                    .with_priority(Priority::Normal),
            );

        let prompt = expertise.to_prompt();

        // Critical should appear before Normal, Normal before Low
        let critical_pos = prompt.find("Critical priority").unwrap();
        let normal_pos = prompt.find("Normal priority").unwrap();
        let low_pos = prompt.find("Low priority").unwrap();

        assert!(critical_pos < normal_pos);
        assert!(normal_pos < low_pos);
    }

    #[test]
    fn test_context_filtering() {
        let expertise = Expertise::new("test", "1.0")
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("Always visible".to_string()))
                    .with_context(ContextProfile::Always),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text("Debug only".to_string()))
                    .with_context(ContextProfile::Conditional {
                        task_types: vec!["Debug".to_string()],
                        user_states: vec![],
                        task_health: None,
                    }),
            );

        // Without debug context
        let prompt1 = expertise.to_prompt_with_context(&ContextMatcher::new());
        assert!(prompt1.contains("Always visible"));
        assert!(!prompt1.contains("Debug only"));

        // With debug context
        let prompt2 =
            expertise.to_prompt_with_context(&ContextMatcher::new().with_task_type("Debug"));
        assert!(prompt2.contains("Always visible"));
        assert!(prompt2.contains("Debug only"));
    }

    #[test]
    fn test_to_tree() {
        let expertise = Expertise::new("test", "1.0")
            .with_tag("test-tag")
            .with_fragment(WeightedFragment::new(KnowledgeFragment::Text(
                "Test content".to_string(),
            )));

        let tree = expertise.to_tree();
        assert!(tree.contains("Expertise: test"));
        assert!(tree.contains("test-tag"));
        assert!(tree.contains("Test content"));
    }

    #[test]
    fn test_to_mermaid() {
        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let mermaid = expertise.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("Expertise: test"));
        assert!(mermaid.contains("Test content"));
    }
}
