//! Prompt definitions for orchestrator LLM interactions.
//!
//! This module uses llm-toolkit's own prompt generation capabilities

use serde::Serialize;

use crate::ToPrompt;

/// Request for generating an execution strategy from a blueprint and available agents.
#[derive(Serialize, ToPrompt)]
#[prompt(template = r##"
# Strategy Generation Task

You are an expert orchestrator tasked with creating a detailed execution strategy.

## User's Task
{{ task }}

## Available Agents
{{ agent_list }}

## Reference Workflow (Blueprint)
{{ blueprint_description }}

{% if blueprint_graph %}
### Visual Flow
```mermaid
{{ blueprint_graph }}
```
{% endif %}

---

## Your Task

Generate a detailed execution strategy as a JSON object with the following structure:

```json
{
  "goal": "A clear statement of what this strategy aims to achieve",
  "steps": [
    {
      "step_id": "step_1",
      "description": "What this step accomplishes",
      "assigned_agent": "AgentName",
      "intent_template": "The prompt to give the agent (can include placeholders like {previous_output})",
      "expected_output": "Description of what output is expected"
    }
  ]
}
```

**Guidelines:**
1. Analyze the user's task and the available agents' expertise
2. Use the blueprint as a reference for the general flow
3. Assign the most appropriate agent to each step
4. Create clear, actionable intent templates
5. Ensure steps build upon each other logically
6. Use placeholders like `{previous_output}`, `{user_request}` in intent templates

**Important:** Return ONLY the JSON object, no additional explanation.
"##)]
pub struct StrategyGenerationRequest {
    pub task: String,
    pub agent_list: String,
    pub blueprint_description: String,
    pub blueprint_graph: String,
}

impl StrategyGenerationRequest {
    /// Creates a new strategy generation request.
    pub fn new(
        task: String,
        agent_list: String,
        blueprint_description: String,
        blueprint_graph: Option<String>,
    ) -> Self {
        Self {
            task,
            agent_list,
            blueprint_description,
            blueprint_graph: blueprint_graph.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_request_creation() {
        let req = StrategyGenerationRequest::new(
            "Test task".to_string(),
            "Agent1, Agent2".to_string(),
            "Blueprint description".to_string(),
            Some("graph TD\nA --> B".to_string()),
        );

        assert_eq!(req.task, "Test task");
        assert_eq!(req.blueprint_graph, "graph TD\nA --> B");
    }

    #[test]
    fn test_strategy_request_to_prompt() {
        use crate::prompt::ToPrompt;

        let req = StrategyGenerationRequest::new(
            "Write an article".to_string(),
            "- WriterAgent: Expert writer".to_string(),
            "1. Research\n2. Write\n3. Review".to_string(),
            None,
        );

        let prompt = req.to_prompt();
        assert!(prompt.contains("Write an article"));
        assert!(prompt.contains("WriterAgent"));
        assert!(prompt.contains("Research"));
    }
}
