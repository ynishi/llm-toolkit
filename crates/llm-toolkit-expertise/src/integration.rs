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
        &self.description
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
        let expertise = Expertise::new("test", "1.0", "Test desc").with_fragment(WeightedFragment::new(
            KnowledgeFragment::Text("Test content".to_string()),
        ));

        let result = ToPrompt::to_prompt(&expertise);
        assert!(result.contains("Expertise: test"));
        assert!(result.contains("Test content"));
    }

    #[test]
    fn test_to_prompt_parts() {
        let expertise = Expertise::new("test", "1.0", "Test desc").with_fragment(WeightedFragment::new(
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
}
