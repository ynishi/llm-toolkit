//! Example: Using predefined strategy with Orchestrator
//!
//! This example demonstrates how to use `set_strategy_map()` to bypass
//! automatic strategy generation and execute a predefined workflow.
//!
//! Run with: cargo run --example orchestrator_with_predefined_strategy --features agent,derive

use llm_toolkit::agent::impls::ClaudeCodeAgent;
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator, StrategyMap, StrategyStep};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🎯 Orchestrator with Predefined Strategy Example\n");

    // Check if claude CLI is available
    println!("🔍 Checking for claude CLI...");
    if !ClaudeCodeAgent::is_available() {
        eprintln!("❌ claude CLI not found in PATH");
        eprintln!("\n💡 Please install the Claude CLI:");
        eprintln!("   npm install -g @anthropic-ai/cli");
        std::process::exit(1);
    }
    println!("✅ claude CLI found\n");

    // Create a blueprint (not used for strategy generation in this example)
    let blueprint =
        BlueprintWorkflow::new("Article writing workflow with predefined steps".to_string());

    // Create orchestrator
    let mut orchestrator = Orchestrator::new(blueprint);
    orchestrator.add_agent(ClaudeCodeAgent::new());

    // Define a custom strategy manually
    let mut strategy = StrategyMap::new("Write a technical article about Rust".to_string());

    // Step 1: Create outline
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Create a detailed outline for the article".to_string(),
        "ClaudeCodeAgent".to_string(),
        "Create an outline for an article about: {{ task }}. Include main sections and key points."
            .to_string(),
        "Article outline with sections and key points".to_string(),
    ));

    // Step 2: Write introduction
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Write an engaging introduction".to_string(),
        "ClaudeCodeAgent".to_string(),
        "Based on this outline: {{ previous_output }}, write an engaging introduction for the article."
            .to_string(),
        "Introduction paragraph".to_string(),
    ));

    // Step 3: Write main content
    strategy.add_step(StrategyStep::new(
        "step_3".to_string(),
        "Write the main content sections".to_string(),
        "ClaudeCodeAgent".to_string(),
        "Using this outline: {{ previous_output }}, write the main content sections with examples and explanations."
            .to_string(),
        "Main content with examples".to_string(),
    ));

    println!("📋 Predefined Strategy:");
    println!("  Goal: {}", strategy.goal);
    println!("  Steps:");
    for (i, step) in strategy.steps.iter().enumerate() {
        println!(
            "    {}. {} (Agent: {})",
            i + 1,
            step.description,
            step.assigned_agent
        );
    }
    println!();

    // Set the predefined strategy
    orchestrator.set_strategy_map(strategy);

    // Verify the strategy is set
    if let Some(current_strategy) = orchestrator.strategy_map() {
        println!(
            "✅ Strategy is set with {} steps\n",
            current_strategy.steps.len()
        );
    }

    // Execute with the predefined strategy
    // Note: The orchestrator will skip strategy generation and use the predefined steps
    println!("🚀 Executing with predefined strategy...\n");

    let task = "Rust's ownership system and borrowing rules";
    let result = orchestrator.execute(task).await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            println!("\n✅ Workflow completed successfully!\n");
            println!("📊 Execution Summary:");
            println!("  Steps executed: {}", result.steps_executed);
            println!("  Redesigns triggered: {}", result.redesigns_triggered);

            if let Some(output) = result.final_output {
                let output_str =
                    serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string());
                println!("\n📄 Final Output:\n{}\n", output_str);
            }
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            let error_msg = result
                .error_message
                .unwrap_or_else(|| "Unknown error".to_string());
            eprintln!("\n❌ Workflow failed: {}\n", error_msg);
            std::process::exit(1);
        }
    }

    Ok(())
}
