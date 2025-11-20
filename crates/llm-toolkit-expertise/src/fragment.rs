//! Knowledge fragment types and definitions.
//!
//! This module defines the core knowledge representation units that can be
//! composed into expertise profiles.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// KnowledgeFragment: Minimal unit of knowledge
///
/// Represents different types of knowledge that can be incorporated
/// into an agent's expertise.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
pub enum KnowledgeFragment {
    /// Thinking logic and procedures
    Logic {
        /// High-level instruction
        instruction: String,
        /// Chain-of-Thought steps
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        steps: Vec<String>,
    },

    /// Behavioral guidelines with anchoring examples
    Guideline {
        /// The rule or guideline statement
        rule: String,
        /// Anchoring examples (positive/negative pairs)
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        anchors: Vec<Anchor>,
    },

    /// Quality evaluation criteria
    QualityStandard {
        /// List of evaluation criteria
        criteria: Vec<String>,
        /// Description of passing grade
        passing_grade: String,
    },

    /// Tool definition (interface)
    ToolDefinition(serde_json::Value),

    /// Free-form text knowledge
    Text(String),
}

impl KnowledgeFragment {
    /// Convert the fragment to a prompt string
    pub fn to_prompt(&self) -> String {
        match self {
            KnowledgeFragment::Logic { instruction, steps } => {
                let mut prompt = format!("## Logic\n{}\n", instruction);
                if !steps.is_empty() {
                    prompt.push_str("\n### Steps:\n");
                    for (i, step) in steps.iter().enumerate() {
                        prompt.push_str(&format!("{}. {}\n", i + 1, step));
                    }
                }
                prompt
            }
            KnowledgeFragment::Guideline { rule, anchors } => {
                let mut prompt = format!("## Guideline\n{}\n", rule);
                if !anchors.is_empty() {
                    prompt.push_str("\n### Examples:\n");
                    for anchor in anchors {
                        prompt.push_str(&anchor.to_prompt());
                        prompt.push('\n');
                    }
                }
                prompt
            }
            KnowledgeFragment::QualityStandard {
                criteria,
                passing_grade,
            } => {
                let mut prompt = String::from("## Quality Standard\n\n### Criteria:\n");
                for criterion in criteria {
                    prompt.push_str(&format!("- {}\n", criterion));
                }
                prompt.push_str(&format!("\n### Passing Grade:\n{}\n", passing_grade));
                prompt
            }
            KnowledgeFragment::ToolDefinition(value) => {
                format!(
                    "## Tool Definition\n```json\n{}\n```\n",
                    serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string())
                )
            }
            KnowledgeFragment::Text(text) => {
                format!("{}\n", text)
            }
        }
    }

    /// Get a short type label for visualization
    pub fn type_label(&self) -> &'static str {
        match self {
            KnowledgeFragment::Logic { .. } => "Logic",
            KnowledgeFragment::Guideline { .. } => "Guideline",
            KnowledgeFragment::QualityStandard { .. } => "Quality",
            KnowledgeFragment::ToolDefinition(_) => "Tool",
            KnowledgeFragment::Text(_) => "Text",
        }
    }

    /// Get a brief summary of the fragment content
    pub fn summary(&self) -> String {
        match self {
            KnowledgeFragment::Logic { instruction, .. } => truncate(instruction, 50),
            KnowledgeFragment::Guideline { rule, .. } => truncate(rule, 50),
            KnowledgeFragment::QualityStandard { criteria, .. } => {
                if criteria.is_empty() {
                    "No criteria".to_string()
                } else {
                    format!("{} criteria", criteria.len())
                }
            }
            KnowledgeFragment::ToolDefinition(_) => "Tool definition".to_string(),
            KnowledgeFragment::Text(text) => truncate(text, 50),
        }
    }
}

/// Anchor: Positive/negative example pair for behavioral anchoring
///
/// Provides concrete examples to establish standards and expectations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Anchor {
    /// Context or scenario
    pub context: String,
    /// Positive example (ideal form)
    pub positive: String,
    /// Negative example (form to avoid)
    pub negative: String,
    /// Explanation of why
    pub reason: String,
}

impl Anchor {
    /// Convert anchor to prompt format
    pub fn to_prompt(&self) -> String {
        format!(
            "**Context:** {}\n✅ **Good:** {}\n❌ **Bad:** {}\n**Why:** {}",
            self.context, self.positive, self.negative, self.reason
        )
    }
}

/// Helper function to truncate text for summaries
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logic_fragment_to_prompt() {
        let fragment = KnowledgeFragment::Logic {
            instruction: "Test instruction".to_string(),
            steps: vec!["Step 1".to_string(), "Step 2".to_string()],
        };
        let prompt = fragment.to_prompt();
        assert!(prompt.contains("## Logic"));
        assert!(prompt.contains("Test instruction"));
        assert!(prompt.contains("1. Step 1"));
        assert!(prompt.contains("2. Step 2"));
    }

    #[test]
    fn test_guideline_fragment_to_prompt() {
        let anchor = Anchor {
            context: "Test context".to_string(),
            positive: "Good example".to_string(),
            negative: "Bad example".to_string(),
            reason: "Because".to_string(),
        };
        let fragment = KnowledgeFragment::Guideline {
            rule: "Test rule".to_string(),
            anchors: vec![anchor],
        };
        let prompt = fragment.to_prompt();
        assert!(prompt.contains("## Guideline"));
        assert!(prompt.contains("Test rule"));
        assert!(prompt.contains("Good example"));
        assert!(prompt.contains("Bad example"));
    }

    #[test]
    fn test_fragment_type_label() {
        let logic = KnowledgeFragment::Logic {
            instruction: "Test".to_string(),
            steps: vec![],
        };
        assert_eq!(logic.type_label(), "Logic");

        let text = KnowledgeFragment::Text("Test".to_string());
        assert_eq!(text.type_label(), "Text");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a very long text", 10), "this is...");
    }
}
