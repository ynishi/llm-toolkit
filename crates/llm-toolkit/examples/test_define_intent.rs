use llm_toolkit::define_intent;
use llm_toolkit::intent::{IntentExtractionError, IntentExtractor};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[define_intent]
#[intent(
    prompt = "Analyze the user's intent from this message: {{ user_input }}\n\n{{ intents_doc }}\n\nRespond with the intent.",
    extractor_tag = "intent"
)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Represents various user intents for a task management system
enum TaskIntent {
    /// Create a new task
    CreateTask,
    /// Update an existing task
    UpdateTask,
    /// Delete a task
    DeleteTask,
    /// List all tasks
    ListTasks,
    /// Mark a task as complete
    CompleteTask,
}

impl FromStr for TaskIntent {
    type Err = IntentExtractionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "CreateTask" => Ok(TaskIntent::CreateTask),
            "UpdateTask" => Ok(TaskIntent::UpdateTask),
            "DeleteTask" => Ok(TaskIntent::DeleteTask),
            "ListTasks" => Ok(TaskIntent::ListTasks),
            "CompleteTask" => Ok(TaskIntent::CompleteTask),
            _ => Err(IntentExtractionError::ParseFailed {
                value: format!("Unknown intent: {}", s),
            }),
        }
    }
}

fn main() {
    // Test the generated prompt function
    let prompt = build_task_intent_prompt("I want to create a new task for tomorrow");
    println!("Generated prompt:\n{}\n", prompt);

    // Test the extractor
    let extractor = TaskIntentExtractor;
    println!("Extractor tag: {}\n", TaskIntentExtractor::EXTRACTOR_TAG);

    // Simulate an LLM response
    let llm_response = "<intent>CreateTask</intent> The user wants to create a new task.";

    match extractor.extract_intent(llm_response) {
        Ok(intent) => {
            println!("Extracted intent: {:?}", intent);
        }
        Err(e) => {
            println!("Failed to extract intent: {:?}", e);
        }
    }

    // Test with a more complex example
    let prompt2 = build_task_intent_prompt("Show me all my pending tasks");
    println!("\nAnother prompt:\n{}", prompt2);

    let llm_response2 = "<intent>ListTasks</intent> The user wants to see their tasks.";
    match extractor.extract_intent(llm_response2) {
        Ok(intent) => {
            println!("Extracted intent: {:?}", intent);
        }
        Err(e) => {
            println!("Failed to extract intent: {:?}", e);
        }
    }
}
