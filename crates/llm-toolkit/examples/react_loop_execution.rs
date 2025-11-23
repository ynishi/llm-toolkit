//! Example demonstrating ReAct loop execution with action selection and expansion.
//!
//! This example shows how to use the `react_loop` function to implement
//! a ReAct (Reasoning + Acting) pattern where an LLM:
//! 1. Selects actions from a registry
//! 2. Actions are expanded into prompts
//! 3. Results are accumulated until task completion
//!
//! Run with: cargo run --example react_loop_execution --features agent,derive

use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::intent::expandable::{
    ReActConfig, SelectionRegistry, react_loop, simple_tag_selector,
};
use std::sync::{Arc, Mutex};

use llm_toolkit::intent::expandable::{Expandable, Selectable};

// Define calculator actions
#[derive(Debug, Clone, PartialEq)]
pub enum CalculatorAction {
    /// Add two numbers
    Add { a: String, b: String },

    /// Multiply two numbers
    Multiply { a: String, b: String },

    /// Get the result
    GetResult,
}

// Manually implement Expandable
impl Expandable for CalculatorAction {
    fn expand(&self) -> Payload {
        match self {
            CalculatorAction::Add { a, b } => Payload::from(format!(
                "Calculate the sum: {} + {}\nReturn only the numeric result.",
                a, b
            )),
            CalculatorAction::Multiply { a, b } => Payload::from(format!(
                "Calculate the product: {} * {}\nReturn only the numeric result.",
                a, b
            )),
            CalculatorAction::GetResult => Payload::from("Return the final result."),
        }
    }
}

// Manually implement Selectable
impl Selectable for CalculatorAction {
    fn selection_id(&self) -> &str {
        match self {
            CalculatorAction::Add { .. } => "add",
            CalculatorAction::Multiply { .. } => "multiply",
            CalculatorAction::GetResult => "get_result",
        }
    }

    fn description(&self) -> &str {
        match self {
            CalculatorAction::Add { .. } => "Add two numbers",
            CalculatorAction::Multiply { .. } => "Multiply two numbers",
            CalculatorAction::GetResult => "Get the result",
        }
    }
}

/// Mock agent that simulates LLM responses for demonstration
struct MockCalculatorAgent {
    iteration: Arc<Mutex<usize>>,
}

impl MockCalculatorAgent {
    fn new() -> Self {
        Self {
            iteration: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait::async_trait]
impl Agent for MockCalculatorAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock calculator agent for demonstration";
        &EXPERTISE
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let prompt = payload.to_text();
        let mut iter = self.iteration.lock().unwrap();
        *iter += 1;
        let iteration = *iter;

        println!("\n=== Agent Execution (iteration {}) ===", iteration);
        println!("Received prompt:\n{}\n", prompt);

        // Simulate different responses based on iteration
        let response = match iteration {
            1 => {
                // First call: LLM selects Add action
                "<action>add</action>\nI'll start by adding 5 and 3.".to_string()
            }
            2 => {
                // Second call: Execute the Add action
                "The result of 5 + 3 is: 8".to_string()
            }
            3 => {
                // Third call: LLM selects Multiply action
                "<action>multiply</action>\nNow I'll multiply the result by 2.".to_string()
            }
            4 => {
                // Fourth call: Execute the Multiply action
                "The result of 8 * 2 is: 16".to_string()
            }
            5 => {
                // Fifth call: LLM indicates completion
                "DONE\nThe final result is 16. Task complete!".to_string()
            }
            _ => {
                // Fallback
                "DONE\nTask complete".to_string()
            }
        };

        println!("Agent response:\n{}\n", response);
        Ok(response)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ReAct Loop Execution Example ===\n");

    // 1. Create a registry with calculator actions
    let mut registry = SelectionRegistry::new();

    registry.register(CalculatorAction::Add {
        a: "5".to_string(),
        b: "3".to_string(),
    });

    registry.register(CalculatorAction::Multiply {
        a: "8".to_string(),
        b: "2".to_string(),
    });

    registry.register(CalculatorAction::GetResult);

    println!("Registered {} actions:", registry.len());
    println!("{}\n", registry.to_prompt_section());

    // 2. Create a mock agent
    let agent = MockCalculatorAgent::new();

    // 3. Create a selector function using the simple_tag_selector helper
    let selector = simple_tag_selector("action", "DONE");

    // 4. Configure the ReAct loop
    let config = ReActConfig::new()
        .with_max_iterations(10)
        .with_completion_marker("DONE")
        .with_accumulate_results(true);

    println!("=== Starting ReAct Loop ===\n");

    // 5. Execute the ReAct loop
    let result = react_loop(&agent, &registry, "Calculate (5 + 3) * 2", selector, config).await?;

    println!("\n=== ReAct Loop Complete ===");
    println!("Final result:\n{}", result);

    println!("\n=== Summary ===");
    println!("The ReAct loop successfully:");
    println!("1. Selected the 'add' action");
    println!("2. Expanded it into a calculation prompt");
    println!("3. Executed and got result: 8");
    println!("4. Selected the 'multiply' action");
    println!("5. Expanded it into another calculation prompt");
    println!("6. Executed and got result: 16");
    println!("7. Completed with final result");

    Ok(())
}
