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
      "intent_template": "The prompt to give the agent (can include placeholders like {% raw %}{{ previous_output }}{% endraw %})",
      "expected_output": "Description of what output is expected",
      "requires_validation": true
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
6. Use Mustache/Jinja2-style placeholders with **double curly braces** like {% raw %}`{{ previous_output }}`{% endraw %}, {% raw %}`{{ user_request }}`{% endraw %} in intent templates (NOT single braces like `{previous_output}`)
7. **IMPORTANT:** Use ONLY double curly braces {% raw %}`{{ }}`{% endraw %}, NOT triple braces {% raw %}`{{{ }}}`{% endraw %}. Intent templates are plain text and do not require HTML escaping.
8. **Placeholder Reference Guide**: Intent templates can access context data using these patterns:
   - **Previous step output**: {% raw %}`{{ step_N_output }}`{% endraw %} or {% raw %}`{{ step_N_output.field }}`{% endraw %} (e.g., {% raw %}`{{ step_1_output.concept }}`{% endraw %})
   - **Previous step (convenience)**: {% raw %}`{{ previous_output }}`{% endraw %} refers to the immediately previous step's output
   - **User request data**: {% raw %}`{{ user_request.field }}`{% endraw %} if the Blueprint defines INPUT context (e.g., {% raw %}`{{ user_request.world_seed.aesthetics }}`{% endraw %})
   - **Other external context**: {% raw %}`{{ context_key.field }}`{% endraw %} for any context added before execution
   - **Nested field access**: Use dot notation to access nested JSON fields (e.g., {% raw %}`{{ step_2_output.data.items[0].name }}`{% endraw %})
9. **Add Validation Steps**: For any step that produces a critical artifact (e.g., a final document, a piece of code, a detailed plan), you SHOULD add a dedicated validation step immediately after it. Select the most appropriate validator agent from the 'Available Agents' list (e.g., InnerValidatorAgent for general validation, or domain-specific validators if available)

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

/// Request for generating an optimized intent prompt for an agent.
#[derive(Serialize, ToPrompt)]
#[prompt(template = r##"
# Intent Generation Task

You are generating an intent prompt that will be given to an agent.

## Critical Constraint

**The agent will receive ONLY the intent text you generate - no separate context.**
Therefore, you MUST embed all necessary context data directly into the intent prompt itself.

## Step Information
**Description**: {{ step_description }}
**Expected Output**: {{ expected_output }}
**Agent Expertise**: {{ agent_expertise }}

## Intent Template (Base structure)
{{ intent_template }}

## Available Context Data
{{ context_info }}

---

## Your Task

Generate the complete intent prompt by:
1. Taking the intent template as a base structure
2. **Replacing all placeholders (like {% raw %}{{ previous_output }}{% endraw %}, {% raw %}{{ step_3_output }}{% endraw %}) with actual context values**
3. Ensuring the resulting intent is self-contained and actionable
4. Making it specific and concrete - avoid abstract requests like "review" or "refine" without concrete instructions
5. Matching the agent's expertise and capabilities

## Example

This example shows how to transform a template with placeholders into a complete, self-contained intent.

### INPUT

**Step Description:** "Review and refine the article for clarity and technical accuracy"

**Intent Template:**
"Review the article in {% raw %}{{ step_3_output }}{% endraw %} and suggest improvements"

**Available Context:**
- step_3_output: "# Rust Ownership\n\nRust uses ownership to manage memory safely without garbage collection. The three rules are:\n1. Each value has an owner\n2. Only one owner at a time\n3. When owner goes out of scope, value is dropped"

### OUTPUT

You must generate a complete intent that embeds the actual article content:

```
Review the following article and suggest 3 specific improvements for clarity and technical accuracy:

# Rust Ownership

Rust uses ownership to manage memory safely without garbage collection. The three rules are:
1. Each value has an owner
2. Only one owner at a time
3. When owner goes out of scope, value is dropped

For each improvement, provide:
1. Section/Line: [specific location]
2. Issue: [what needs improvement]
3. Suggestion: [concrete fix]
```

**Key point:** The agent receives ONLY the OUTPUT text above. It cannot access {% raw %}{{ step_3_output }}{% endraw %} separately, so you must copy the actual content into the intent.

**IMPORTANT NOTE ON PLACEHOLDER SYNTAX:**
- Always use **double curly braces** {% raw %}`{{ placeholder_name }}`{% endraw %} (Mustache/Jinja2 style)
- DO NOT use single braces `{placeholder_name}` - this will NOT be recognized
- DO NOT use triple braces {% raw %}`{{{ placeholder_name }}}`{% endraw %} - intent templates are plain text and do not require HTML escaping
- Include spaces inside braces: {% raw %}`{{ name }}`{% endraw %} not {% raw %}`{{name}}`{% endraw %}

**Important:** Return ONLY the final intent prompt text, no additional explanation or metadata.
"##)]
pub struct IntentGenerationRequest {
    pub step_description: String,
    pub expected_output: String,
    pub agent_expertise: String,
    pub intent_template: String,
    pub context_info: String,
}

impl IntentGenerationRequest {
    /// Creates a new intent generation request.
    pub fn new(
        step_description: String,
        expected_output: String,
        agent_expertise: String,
        intent_template: String,
        context_info: String,
    ) -> Self {
        Self {
            step_description,
            expected_output,
            agent_expertise,
            intent_template,
            context_info,
        }
    }
}

/// Request for deciding whether and how to redesign the strategy after an error.
#[derive(Serialize, ToPrompt)]
#[prompt(template = r##"
# Redesign Decision Task

An error occurred during workflow execution. Analyze the situation and decide on the appropriate recovery strategy.

## Goal
{{ goal }}

## Progress
- Completed Steps: {{ completed_steps }} / {{ total_steps }}
- Failed Step: {{ failed_step_description }}
- Error: {{ error_message }}

## Completed Work So Far
{{ completed_context }}

---

## Your Task

Analyze the error and determine the appropriate recovery strategy:

1. **RETRY** - The error is transient (network timeout, temporary service unavailability). Simply retry the same step.
2. **TACTICAL** - The error is localized. The failed step and subsequent steps need redesign, but previous work is still valid.
3. **FULL** - The error is fundamental. The entire strategy needs to be reconsidered from scratch.

**Important:** Respond with ONLY one word: `RETRY`, `TACTICAL`, or `FULL`.
"##)]
pub struct RedesignDecisionRequest {
    pub goal: String,
    pub completed_steps: usize,
    pub total_steps: usize,
    pub failed_step_description: String,
    pub error_message: String,
    pub completed_context: String,
}

impl RedesignDecisionRequest {
    /// Creates a new redesign decision request.
    pub fn new(
        goal: String,
        completed_steps: usize,
        total_steps: usize,
        failed_step_description: String,
        error_message: String,
        completed_context: String,
    ) -> Self {
        Self {
            goal,
            completed_steps,
            total_steps,
            failed_step_description,
            error_message,
            completed_context,
        }
    }
}

/// Request for tactical redesign of remaining steps after a failure.
#[derive(Serialize, ToPrompt)]
#[prompt(template = r##"
# Tactical Redesign Task

A step in the workflow has failed. Redesign the remaining steps to work around the error while preserving completed work.

## Overall Goal
{{ goal }}

## Current Strategy
{{ current_strategy }}

## Failed Step
**Index**: {{ failed_step_index }}
**Description**: {{ failed_step_description }}
**Error**: {{ error_message }}

## Completed Work (Preserve This)
{{ completed_context }}

## Available Agents
{{ agent_list }}

---

## Your Task

Redesign the steps starting from the failed step onwards to achieve the goal. You may:
- Modify the failed step to avoid the error
- Insert new steps to work around the problem
- Remove unnecessary steps
- Change agent assignments
- Adjust intent templates

Generate a JSON array of `StrategyStep` objects with the following structure:

```json
[
  {
    "step_id": "step_X",
    "description": "What this step accomplishes",
    "assigned_agent": "AgentName",
    "intent_template": "The prompt template (can include {% raw %}{{ previous_output }}{% endraw %}, etc. - use double braces)",
    "expected_output": "Description of expected output"
  }
]
```

**Placeholder Reference Guide**: Intent templates can access context data using:
- **Previous step output**: {% raw %}`{{ step_N_output }}`{% endraw %} or {% raw %}`{{ step_N_output.field }}`{% endraw %} (e.g., {% raw %}`{{ step_1_output.concept }}`{% endraw %})
- **Previous step (convenience)**: {% raw %}`{{ previous_output }}`{% endraw %}
- **User request data**: {% raw %}`{{ user_request.field }}`{% endraw %} for Blueprint INPUT context
- **Nested fields**: Use dot notation (e.g., {% raw %}`{{ step_2_output.data.count }}`{% endraw %})

**Important:** Return ONLY the JSON array, no additional explanation.
"##)]
pub struct TacticalRedesignRequest {
    pub goal: String,
    pub current_strategy: String,
    pub failed_step_index: usize,
    pub failed_step_description: String,
    pub error_message: String,
    pub completed_context: String,
    pub agent_list: String,
}

impl TacticalRedesignRequest {
    /// Creates a new tactical redesign request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        goal: String,
        current_strategy: String,
        failed_step_index: usize,
        failed_step_description: String,
        error_message: String,
        completed_context: String,
        agent_list: String,
    ) -> Self {
        Self {
            goal,
            current_strategy,
            failed_step_index,
            failed_step_description,
            error_message,
            completed_context,
            agent_list,
        }
    }
}

/// Request for full strategy regeneration after a fundamental failure.
#[derive(Serialize, ToPrompt)]
#[prompt(template = r##"
# Full Strategy Regeneration Task

The previous execution strategy failed fundamentally. Generate a completely new strategy that learns from the failure.

## Original Task
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

## Previous Attempt

### Failed Strategy
{{ failed_strategy }}

### What Went Wrong
{{ error_summary }}

### Completed Work (Can Be Referenced)
{{ completed_work }}

---

## Your Task

Analyze the failure and create a **completely new strategy** that:
1. Avoids the mistakes of the previous approach
2. Takes a different angle or uses different agents if needed
3. Leverages any completed work that's still valid
4. Has a higher chance of success

Generate a JSON object with the following structure:

```json
{
  "goal": "A clear statement of what this strategy aims to achieve",
  "steps": [
    {
      "step_id": "step_1",
      "description": "What this step accomplishes",
      "assigned_agent": "AgentName",
      "intent_template": "The prompt to give the agent (can include placeholders like {% raw %}{{ previous_output }}{% endraw %})",
      "expected_output": "Description of what output is expected"
    }
  ]
}
```

**Placeholder Reference Guide**: Intent templates can access context data using:
- **Previous step output**: {% raw %}`{{ step_N_output }}`{% endraw %} or {% raw %}`{{ step_N_output.field }}`{% endraw %} (e.g., {% raw %}`{{ step_1_output.concept }}`{% endraw %})
- **Previous step (convenience)**: {% raw %}`{{ previous_output }}`{% endraw %}
- **User request data**: {% raw %}`{{ user_request.field }}`{% endraw %} for Blueprint INPUT context
- **Nested fields**: Use dot notation (e.g., {% raw %}`{{ step_2_output.data.count }}`{% endraw %})

**Important:** Return ONLY the JSON object, no additional explanation.
"##)]
pub struct FullRegenerateRequest {
    pub task: String,
    pub agent_list: String,
    pub blueprint_description: String,
    pub blueprint_graph: String,
    pub failed_strategy: String,
    pub error_summary: String,
    pub completed_work: String,
}

impl FullRegenerateRequest {
    /// Creates a new full regeneration request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: String,
        agent_list: String,
        blueprint_description: String,
        blueprint_graph: Option<String>,
        failed_strategy: String,
        error_summary: String,
        completed_work: String,
    ) -> Self {
        Self {
            task,
            agent_list,
            blueprint_description,
            blueprint_graph: blueprint_graph.unwrap_or_default(),
            failed_strategy,
            error_summary,
            completed_work,
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

        // CRITICAL: Verify that placeholder examples are preserved (not expanded by minijinja)
        assert!(
            prompt.contains("{{ previous_output }}"),
            "Placeholder {{ previous_output }} should be preserved in the prompt, not expanded to empty string"
        );
    }

    #[test]
    fn test_intent_generation_request() {
        let req = IntentGenerationRequest::new(
            "Analyze user requirements".to_string(),
            "List of requirements".to_string(),
            "Expert in requirement analysis".to_string(),
            "Analyze: {user_input}".to_string(),
            "user_input: Build a web app".to_string(),
        );

        let prompt = req.to_prompt();
        assert!(prompt.contains("Analyze user requirements"));
        assert!(prompt.contains("Expert in requirement analysis"));
    }

    #[test]
    fn test_redesign_decision_request() {
        let req = RedesignDecisionRequest::new(
            "Complete the task".to_string(),
            2,
            5,
            "Step 3 failed".to_string(),
            "Network timeout".to_string(),
            "Steps 1-2 completed".to_string(),
        );

        let prompt = req.to_prompt();
        assert!(prompt.contains("Complete the task"));
        assert!(prompt.contains("Network timeout"));
    }
}
