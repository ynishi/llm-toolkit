//! Basic example of using Expandable and Selectable traits.
//!
//! This example demonstrates how to:
//! 1. Define actions that implement Expandable and Selectable
//! 2. Register them in a SelectionRegistry
//! 3. Present them to an agent for selection
//! 4. Expand selected actions into prompts
//!
//! Run with: cargo run --example expandable_basic --features agent

use llm_toolkit::agent::Payload;
use llm_toolkit::intent::expandable::{Expandable, Selectable, SelectionRegistry};

/// Example actions that an agent can perform
#[derive(Debug, Clone)]
enum ToolAction {
    FileRead { path: String },
    FileWrite { path: String, content: String },
    Calculate { expression: String },
    WebSearch { query: String },
}

impl Expandable for ToolAction {
    fn expand(&self) -> Payload {
        match self {
            ToolAction::FileRead { path } => Payload::from(format!(
                "Read the contents of the file at path: {}\nReturn the file contents.",
                path
            )),
            ToolAction::FileWrite { path, content } => Payload::from(format!(
                "Write the following content to file at path: {}\n\nContent:\n{}\n\nConfirm when done.",
                path, content
            )),
            ToolAction::Calculate { expression } => Payload::from(format!(
                "Calculate the result of: {}\nReturn only the numeric result.",
                expression
            )),
            ToolAction::WebSearch { query } => Payload::from(format!(
                "Search the web for: {}\nReturn a summary of the top 3 results.",
                query
            )),
        }
    }
}

impl Selectable for ToolAction {
    fn selection_id(&self) -> &str {
        match self {
            ToolAction::FileRead { .. } => "file_read",
            ToolAction::FileWrite { .. } => "file_write",
            ToolAction::Calculate { .. } => "calculate",
            ToolAction::WebSearch { .. } => "web_search",
        }
    }

    fn description(&self) -> &str {
        match self {
            ToolAction::FileRead { .. } => "Read contents from a file",
            ToolAction::FileWrite { .. } => "Write content to a file",
            ToolAction::Calculate { .. } => "Perform mathematical calculations",
            ToolAction::WebSearch { .. } => "Search the web for information",
        }
    }
}

fn main() {
    println!("=== Expandable and Selectable Basic Example ===\n");

    // Create a registry and register actions
    let mut registry = SelectionRegistry::new();

    registry.register(ToolAction::FileRead {
        path: "/tmp/example.txt".to_string(),
    });

    registry.register(ToolAction::FileWrite {
        path: "/tmp/output.txt".to_string(),
        content: "Hello, World!".to_string(),
    });

    registry.register(ToolAction::Calculate {
        expression: "2 + 2".to_string(),
    });

    registry.register(ToolAction::WebSearch {
        query: "Rust programming".to_string(),
    });

    println!("Registered {} actions\n", registry.len());

    // Generate prompt section for LLM
    let prompt_section = registry.to_prompt_section();
    println!("Prompt section for LLM:");
    println!("{}", prompt_section);
    println!();

    // Simulate LLM selecting an action
    println!("=== Simulating Action Selection ===\n");

    let selected_id = "calculate";
    println!("LLM selected: {}", selected_id);

    if let Some(action) = registry.get(selected_id) {
        println!("Action description: {}", action.description());

        // Expand the action into a prompt
        let expanded = action.expand();
        println!("\nExpanded prompt:");
        println!("{}", expanded.to_text());
    } else {
        println!("Action not found!");
    }

    println!("\n=== Example Complete ===");
}
