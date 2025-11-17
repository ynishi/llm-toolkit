//! Example combining define_intent! macro with Expandable trait.
//!
//! This demonstrates a more realistic use case where:
//! 1. The LLM selects actions using define_intent!
//! 2. Selected actions implement Expandable for prompt generation
//! 3. The expanded prompts can be fed back to the LLM
//!
//! Run with: cargo run --example expandable_with_intent --features agent,derive

use llm_toolkit::agent::Payload;
use llm_toolkit::define_intent;
use llm_toolkit::intent::expandable::{Expandable, Selectable, SelectionRegistry};
use llm_toolkit::IntentExtractor;

// Define a set of actions using define_intent!
#[define_intent]
#[intent(
    prompt = r#"
You are a helpful assistant with access to the following tools:

{{ tools_doc }}

Based on the user's request, select the most appropriate tool to use.
Respond with: <tool>TOOL_NAME</tool>

User request: {{ user_request }}
"#,
    extractor_tag = "tool"
)]
/// Tools available to the assistant
#[derive(Debug, Clone, PartialEq)]
pub enum Tool {
    /// Search for information on the web
    WebSearch,
    /// Read a file from the filesystem
    FileRead,
    /// Write content to a file
    FileWrite,
    /// Perform mathematical calculations
    Calculator,
}

// Implement FromStr for Intent extraction
impl std::str::FromStr for Tool {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "WebSearch" => Ok(Tool::WebSearch),
            "FileRead" => Ok(Tool::FileRead),
            "FileWrite" => Ok(Tool::FileWrite),
            "Calculator" => Ok(Tool::Calculator),
            _ => Err(format!("Unknown tool: {}", s)),
        }
    }
}

// Implement Expandable for Tool
impl Expandable for Tool {
    fn expand(&self) -> Payload {
        match self {
            Tool::WebSearch => Payload::from(
                "You have access to web search. \
                 What query would you like to search for? \
                 Provide the search query.",
            ),
            Tool::FileRead => Payload::from(
                "You have access to file reading. \
                 What file path would you like to read? \
                 Provide the absolute file path.",
            ),
            Tool::FileWrite => Payload::from(
                "You have access to file writing. \
                 Provide the file path and content in the format:\n\
                 Path: /path/to/file\n\
                 Content: your content here",
            ),
            Tool::Calculator => Payload::from(
                "You have access to a calculator. \
                 What calculation would you like to perform? \
                 Provide a mathematical expression (e.g., 2+2, 5*3-1).",
            ),
        }
    }
}

// Implement Selectable for Tool
impl Selectable for Tool {
    fn selection_id(&self) -> &str {
        match self {
            Tool::WebSearch => "web_search",
            Tool::FileRead => "file_read",
            Tool::FileWrite => "file_write",
            Tool::Calculator => "calculator",
        }
    }

    fn description(&self) -> &str {
        match self {
            Tool::WebSearch => "Search for information on the web",
            Tool::FileRead => "Read a file from the filesystem",
            Tool::FileWrite => "Write content to a file",
            Tool::Calculator => "Perform mathematical calculations",
        }
    }
}

fn main() {
    println!("=== Expandable with Intent Example ===\n");

    // Create a SelectionRegistry
    let mut registry = SelectionRegistry::new();
    registry.register(Tool::WebSearch);
    registry.register(Tool::FileRead);
    registry.register(Tool::FileWrite);
    registry.register(Tool::Calculator);

    println!("Registered {} tools\n", registry.len());

    // Generate tools documentation
    let tools_doc = registry.to_prompt_section_with_title("Available Tools");

    // Build the intent selection prompt
    let user_request = "I need to calculate the sum of 150 and 275";
    let prompt = build_tool_prompt(&tools_doc, user_request);

    println!("=== Step 1: LLM Selects Tool ===\n");
    println!("Prompt sent to LLM:");
    println!("{}\n", prompt);

    // Simulate LLM response
    let llm_response = "Based on the request, I'll use the calculator tool. <tool>Calculator</tool>";
    println!("LLM Response:");
    println!("{}\n", llm_response);

    // Extract the selected tool using the generated extractor
    let extractor = ToolExtractor;
    match extractor.extract_intent(llm_response) {
        Ok(tool) => {
            println!("=== Step 2: Expand Selected Tool ===\n");
            println!("Selected tool: {:?}", tool);

            // Expand the tool into a follow-up prompt
            let expanded_payload = tool.expand();
            println!("\nExpanded prompt for LLM:");
            println!("{}\n", expanded_payload.to_text());

            println!("=== Step 3: LLM Executes Tool ===\n");

            // In a real scenario, you would send the expanded prompt to the LLM
            // and get the actual result. Here we simulate it.
            let simulated_execution = match tool {
                Tool::Calculator => {
                    "The user wants to calculate 150 + 275. The result is: 425"
                }
                _ => "Tool execution result would appear here",
            };

            println!("Tool execution result:");
            println!("{}", simulated_execution);
        }
        Err(e) => {
            eprintln!("Failed to extract tool: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nIn a real ReAct loop, this process would continue:");
    println!("1. LLM analyzes results");
    println!("2. Either completes the task or selects another tool");
    println!("3. Process repeats until task is complete");
}
