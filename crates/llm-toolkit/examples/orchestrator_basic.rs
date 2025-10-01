//! Basic orchestrator example demonstrating multi-agent workflow execution.
//!
//! This example shows how to:
//! - Create a BlueprintWorkflow with natural language description
//! - Add agents to the orchestrator
//! - Execute a task with automatic strategy generation
//! - Handle errors with adaptive redesign
//!
//! Run with: cargo run --example orchestrator_basic --features agent,derive

use llm_toolkit::agent::impls::ClaudeCodeAgent;
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger - set RUST_LOG=debug to see detailed execution logs
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🎭 Orchestrator Basic Example\n");
    println!("💡 Set RUST_LOG=debug for detailed logs, RUST_LOG=trace for full output\n");

    // Check if claude CLI is available
    println!("🔍 Checking for claude CLI...");
    if !ClaudeCodeAgent::is_available() {
        eprintln!("❌ claude CLI not found in PATH");
        eprintln!("\n💡 Please install the Claude CLI:");
        eprintln!("   npm install -g @anthropic-ai/cli");
        eprintln!("   or visit: https://github.com/anthropics/anthropic-sdk-typescript\n");
        std::process::exit(1);
    }
    println!("✅ claude CLI found\n");

    // Define a workflow blueprint using natural language
    let blueprint = BlueprintWorkflow::with_graph(
        r#"
        Technical Article Generation Workflow:
        1. Analyze the topic and create a detailed outline
        2. Research key concepts and gather information
        3. Write the main content section by section
        4. Generate a compelling title and summary
        5. Review and refine the final article
        "#
        .to_string(),
        r#"
        graph TD
            A[Topic Analysis] --> B[Research]
            B --> C[Content Writing]
            C --> D[Title & Summary]
            D --> E[Review & Refine]
        "#
        .to_string(),
    );

    // Create orchestrator with the blueprint
    let mut orchestrator = Orchestrator::new(blueprint);

    // Add available agents
    // Note: ClaudeCodeAgent requires 'claude' CLI to be available
    let claude_agent = ClaudeCodeAgent::new();
    orchestrator.add_agent(Box::new(claude_agent));

    println!("📋 Available agents:");
    for agent_name in orchestrator.list_agents() {
        println!("  - {}", agent_name);
    }
    println!();

    // Execute the workflow
    println!("🚀 Starting workflow execution...\n");

    let task = "Write a beginner-friendly article about Rust's ownership system";

    match orchestrator.execute(task).await {
        Ok(result) => {
            println!("\n✅ Workflow completed successfully!\n");
            println!("📄 Final Output:\n{}\n", result);
        }
        Err(e) => {
            eprintln!("\n❌ Workflow failed: {}\n", e);
            eprintln!(
                "💡 Tip: Make sure the 'claude' CLI is installed and available in your PATH."
            );
            eprintln!(
                "   Install it from: https://github.com/anthropics/anthropic-sdk-typescript/tree/main/packages/cli"
            );
            std::process::exit(1);
        }
    }

    Ok(())
}
