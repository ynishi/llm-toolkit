//! Integration with llm-toolkit
//!
//! This module provides implementations of llm-toolkit traits when the
//! "integration" feature is enabled.

use crate::Expertise;
use llm_toolkit::agent::{Capability, ToExpertise};
use llm_toolkit::prompt::{PromptPart, ToPrompt};

impl ToPrompt for Expertise {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        // Delegate to our own to_prompt() method and wrap in a Text PromptPart
        let prompt_text = Expertise::to_prompt(self);
        vec![PromptPart::Text(prompt_text)]
    }

    fn to_prompt(&self) -> String {
        // Delegate to our own to_prompt() method directly
        Expertise::to_prompt(self)
    }
}

impl ToExpertise for Expertise {
    fn description(&self) -> &str {
        // If explicit description exists, return it
        if let Some(desc) = &self.description {
            return desc;
        }

        // Fallback to id if no description is set
        // Note: For richer auto-generation, users should call get_description() explicitly
        // or set description via with_description()
        &self.id
    }

    fn capabilities(&self) -> Vec<Capability> {
        // Extract tool names from ToolDefinition fragments and convert to Capability
        self.extract_tool_names()
            .into_iter()
            .map(|name| Capability::new(name))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KnowledgeFragment, WeightedFragment};

    #[test]
    fn test_to_prompt_trait() {
        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let result = ToPrompt::to_prompt(&expertise);
        assert!(result.contains("Expertise: test"));
        assert!(result.contains("Test content"));
    }

    #[test]
    fn test_to_prompt_parts() {
        let expertise = Expertise::new("test", "1.0").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let parts = ToPrompt::to_prompt_parts(&expertise);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            PromptPart::Text(text) => {
                assert!(text.contains("Expertise: test"));
                assert!(text.contains("Test content"));
            }
            _ => panic!("Expected Text PromptPart"),
        }
    }

    #[test]
    fn test_auto_description() {
        // No explicit description - should fallback to id
        let expertise = Expertise::new("test-agent", "1.0");
        assert_eq!(expertise.description(), "test-agent");

        // With explicit description
        let expertise_with_desc = Expertise::new("test-agent", "1.0")
            .with_description("A test agent");
        assert_eq!(expertise_with_desc.description(), "A test agent");

        // get_description() auto-generates from first fragment
        let expertise_with_fragment = Expertise::new("test-agent", "1.0")
            .with_fragment(WeightedFragment::new(
                KnowledgeFragment::Text("You are a helpful assistant specialized in Rust programming. You provide clear, concise, and accurate answers.".to_string()),
            ));
        let auto_desc = expertise_with_fragment.get_description();
        assert!(auto_desc.starts_with("You are a helpful assistant"));
        assert!(auto_desc.len() <= 103); // ~100 chars + "..."
    }
}
