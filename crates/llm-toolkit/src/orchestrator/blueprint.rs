//! BlueprintWorkflow - Flexible workflow definition.
//!
//! Blueprints define the high-level intent of a workflow using natural language
//! and optional Mermaid diagrams, rather than rigid type structures.

use serde::{Deserialize, Serialize};

/// A flexible workflow definition that guides the orchestrator.
///
/// Unlike traditional workflow engines that require strict type definitions,
/// BlueprintWorkflow uses natural language and optional visual representations
/// to describe the intended process. This allows the orchestrator's LLM to
/// interpret and adapt the workflow dynamically.
///
/// # Design Philosophy
///
/// - **Natural Language First**: The description is human-readable and LLM-parseable
/// - **Visual Optional**: Mermaid graphs provide additional clarity but aren't required
/// - **No Runtime Enforcement**: The blueprint is a guide, not a contract
///
/// # Example
///
/// ```rust
/// use llm_toolkit::orchestrator::BlueprintWorkflow;
///
/// let blueprint = BlueprintWorkflow {
///     description: r#"
///         Article Generation Workflow:
///         1. Analyze user requirements and identify research topics
///         2. Conduct web research from multiple sources
///         3. Structure and analyze collected information
///         4. Generate article outline
///         5. Write full article body
///         6. Create title and summary
///         7. Perform quality validation
///     "#.to_string(),
///
///     graph: Some(r#"
///         graph TD
///             A[Requirements] --> B[Research]
///             B --> C[Analysis]
///             C --> D[Outline]
///             D --> E[Writing]
///             E --> F[Title]
///             F --> G[Validation]
///             G -->|NG| D
///             G -->|OK| H[Done]
///     "#.to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintWorkflow {
    /// Natural language description of the workflow.
    ///
    /// This should describe what needs to be accomplished and the general
    /// sequence of steps, using language that both humans and LLMs can understand.
    pub description: String,

    /// Optional Mermaid diagram representing the workflow visually.
    ///
    /// This provides additional structure for the orchestrator to understand
    /// dependencies, loops, and decision points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<String>,
}

impl BlueprintWorkflow {
    /// Creates a new BlueprintWorkflow with only a description.
    pub fn new(description: String) -> Self {
        Self {
            description,
            graph: None,
        }
    }

    /// Creates a new BlueprintWorkflow with both description and graph.
    pub fn with_graph(description: String, graph: String) -> Self {
        Self {
            description,
            graph: Some(graph),
        }
    }

    /// Checks if this blueprint has a visual graph.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blueprint_creation() {
        let bp = BlueprintWorkflow::new("Test workflow".to_string());
        assert_eq!(bp.description, "Test workflow");
        assert!(!bp.has_graph());
    }

    #[test]
    fn test_blueprint_with_graph() {
        let bp = BlueprintWorkflow::with_graph("Test".to_string(), "graph TD\nA --> B".to_string());
        assert!(bp.has_graph());
    }

    #[test]
    fn test_blueprint_serialization() {
        let bp = BlueprintWorkflow::with_graph(
            "Workflow".to_string(),
            "graph LR\nStart --> End".to_string(),
        );

        let json = serde_json::to_string(&bp).unwrap();
        let deserialized: BlueprintWorkflow = serde_json::from_str(&json).unwrap();

        assert_eq!(bp.description, deserialized.description);
        assert_eq!(bp.graph, deserialized.graph);
    }
}
