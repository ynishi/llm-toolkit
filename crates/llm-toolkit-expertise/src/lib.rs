//! # llm-toolkit-expertise
//!
//! Agent as Code v2: Graph-based composition system for LLM agent capabilities.
//!
//! This library provides a flexible, composition-based approach to defining agent expertise
//! through weighted knowledge fragments. Instead of inheritance hierarchies, expertise is
//! built by composing independent fragments with priorities and contextual activation rules.
//!
//! ## Core Concepts
//!
//! - **Composition over Inheritance**: Build agents like RPG equipment sets
//! - **Weighted Fragments**: Knowledge with priority levels (Critical/High/Normal/Low)
//! - **Context-Driven**: Dynamic behavior based on TaskHealth and context
//!
//! ## Example
//!
//! ```rust
//! use llm_toolkit_expertise::{
//!     Expertise, WeightedFragment, KnowledgeFragment,
//!     Priority, ContextProfile, TaskHealth,
//! };
//!
//! let expertise = Expertise::new("code-reviewer", "1.0")
//!     .with_description("Rust code review specialist")
//!     .with_tag("lang:rust")
//!     .with_tag("role:reviewer")
//!     .with_fragment(
//!         WeightedFragment::new(KnowledgeFragment::Text(
//!             "Always verify code compiles before review".to_string()
//!         ))
//!         .with_priority(Priority::Critical)
//!     )
//!     .with_fragment(
//!         WeightedFragment::new(KnowledgeFragment::Logic {
//!             instruction: "Check for security issues".to_string(),
//!             steps: vec![
//!                 "Scan for SQL injection vulnerabilities".to_string(),
//!                 "Check input validation".to_string(),
//!             ],
//!         })
//!         .with_priority(Priority::High)
//!         .with_context(ContextProfile::Conditional {
//!             task_types: vec!["security-review".to_string()],
//!             user_states: vec![],
//!             task_health: None,
//!         })
//!     );
//!
//! // Generate prompt
//! let prompt = expertise.to_prompt();
//! println!("{}", prompt);
//!
//! // Generate tree visualization
//! let tree = expertise.to_tree();
//! println!("{}", tree);
//!
//! // Generate Mermaid graph
//! let mermaid = expertise.to_mermaid();
//! println!("{}", mermaid);
//! ```
//!
//! ## Context-Aware Rendering
//!
//! Phase 2 adds dynamic prompt rendering based on runtime context:
//!
//! ```rust
//! use llm_toolkit_expertise::{
//!     Expertise, WeightedFragment, KnowledgeFragment,
//!     RenderContext, ContextualPrompt, Priority, ContextProfile, TaskHealth,
//! };
//!
//! // Create expertise with conditional fragments
//! let expertise = Expertise::new("rust-tutor", "1.0")
//!     .with_fragment(
//!         WeightedFragment::new(KnowledgeFragment::Text(
//!             "You are a Rust tutor".to_string()
//!         ))
//!         .with_context(ContextProfile::Always)
//!     )
//!     .with_fragment(
//!         WeightedFragment::new(KnowledgeFragment::Text(
//!             "Provide detailed explanations".to_string()
//!         ))
//!         .with_context(ContextProfile::Conditional {
//!             task_types: vec![],
//!             user_states: vec!["beginner".to_string()],
//!             task_health: None,
//!         })
//!     );
//!
//! // Render with context
//! let beginner_context = RenderContext::new().with_user_state("beginner");
//! let prompt = expertise.to_prompt_with_render_context(&beginner_context);
//! // Includes both "Always" and "beginner" fragments
//!
//! // Or use ContextualPrompt wrapper
//! let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
//!     .with_user_state("beginner")
//!     .to_prompt();
//! ```
//!
//! ## JSON Schema
//!
//! This library supports JSON Schema generation for expertise definitions:
//!
//! ```rust
//! use llm_toolkit_expertise::dump_expertise_schema;
//!
//! let schema = dump_expertise_schema();
//! println!("{}", serde_json::to_string_pretty(&schema).unwrap());
//! ```

// Allow the crate to reference itself by name
extern crate self as llm_toolkit_expertise;

pub mod context;
pub mod fragment;
pub mod render;
pub mod types;

// Re-export main types
pub use context::{ContextMatcher, ContextProfile, Priority, TaskHealth};
pub use fragment::{Anchor, KnowledgeFragment};
pub use render::{ContextualPrompt, RenderContext};
pub use types::{Expertise, WeightedFragment};

// Optional integration with llm-toolkit
#[cfg(feature = "integration")]
mod integration;

/// Generate JSON Schema for Expertise type
///
/// Returns the JSON Schema as a serde_json::Value for inspection or storage.
pub fn dump_expertise_schema() -> serde_json::Value {
    let schema = schemars::schema_for!(Expertise);
    serde_json::to_value(&schema).expect("Failed to serialize schema")
}

/// Save Expertise JSON Schema to a file
///
/// # Errors
///
/// Returns an error if file writing fails.
pub fn save_expertise_schema(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    let schema = dump_expertise_schema();
    let json = serde_json::to_string_pretty(&schema)?;
    std::fs::write(path, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_schema() {
        let schema = dump_expertise_schema();
        assert!(schema.is_object());
        assert!(schema.get("$schema").is_some());
    }

    #[test]
    fn test_basic_expertise_creation() {
        let expertise = Expertise::new("test", "1.0")
            .with_description("Test expertise")
            .with_tag("test")
            .with_fragment(WeightedFragment::new(KnowledgeFragment::Text(
                "Test".to_string(),
            )));

        assert_eq!(expertise.id, "test");
        assert_eq!(expertise.version, "1.0");
        assert_eq!(expertise.description, Some("Test expertise".to_string()));
        assert_eq!(expertise.tags.len(), 1);
        assert_eq!(expertise.content.len(), 1);
    }

    #[test]
    fn test_to_prompt_generates_valid_output() {
        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let prompt = expertise.to_prompt();
        assert!(prompt.contains("Expertise: test"));
        assert!(prompt.contains("Test content"));
    }

    #[test]
    fn test_visualizations() {
        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test".to_string()),
        ));

        // Tree visualization
        let tree = expertise.to_tree();
        assert!(tree.contains("Expertise: test"));

        // Mermaid visualization
        let mermaid = expertise.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("Expertise: test"));
    }
}
