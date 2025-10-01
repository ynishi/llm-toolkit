//! Example demonstrating model selection for different backends.
//!
//! This example shows how to use the `model` attribute to select
//! specific models for both Claude and Gemini backends.
//!
//! Run with: cargo run --example agent_model_selection --features agent

use llm_toolkit::{Agent, agent::Agent as AgentTrait};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct CodeAnalysis {
    language: String,
    complexity: String,
    suggestions: Vec<String>,
}

// Claude with Sonnet 4.5 (default)
#[derive(Agent)]
#[agent(
    expertise = "Analyzing code quality and patterns",
    output = "CodeAnalysis"
)]
struct ClaudeSonnetAgent;

// Claude with Opus 4 (most capable)
#[derive(Agent)]
#[agent(
    expertise = "Analyzing code quality and patterns",
    output = "CodeAnalysis",
    backend = "claude",
    model = "opus"
)]
struct ClaudeOpusAgent;

// Gemini Flash (fast)
#[derive(Agent)]
#[agent(
    expertise = "Analyzing code quality and patterns",
    output = "CodeAnalysis",
    backend = "gemini",
    model = "flash"
)]
struct GeminiFlashAgent;

// Gemini Pro (capable)
#[derive(Agent)]
#[agent(
    expertise = "Analyzing code quality and patterns",
    output = "CodeAnalysis",
    backend = "gemini",
    model = "pro"
)]
struct GeminiProAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🎯 Agent Model Selection Example\n");

    let sonnet = ClaudeSonnetAgent;
    let opus = ClaudeOpusAgent;
    let flash = GeminiFlashAgent;
    let pro = GeminiProAgent;

    println!("📊 Available Agent Configurations:\n");

    println!("1. Claude Sonnet 4.5 (Default)");
    println!("   Backend: claude");
    println!("   Model: claude-sonnet-4.5");
    println!("   Profile: Balanced performance and speed");
    println!("   Expertise: {}\n", AgentTrait::expertise(&sonnet));

    println!("2. Claude Opus 4");
    println!("   Backend: claude");
    println!("   Model: claude-opus-4");
    println!("   Profile: Most capable");
    println!("   Expertise: {}\n", AgentTrait::expertise(&opus));

    println!("3. Gemini Flash");
    println!("   Backend: gemini");
    println!("   Model: gemini-2.5-flash");
    println!("   Profile: Fast and efficient");
    println!("   Expertise: {}\n", AgentTrait::expertise(&flash));

    println!("4. Gemini Pro");
    println!("   Backend: gemini");
    println!("   Model: gemini-2.5-pro");
    println!("   Profile: Most capable");
    println!("   Expertise: {}\n", AgentTrait::expertise(&pro));

    println!("✨ Features:");
    println!("   - Model selection at compile time");
    println!("   - Same interface across all models");
    println!("   - Type-safe outputs");
    println!("   - Easy switching between models\n");

    println!("💡 Model Selection Guide:");
    println!("   Claude:");
    println!("   - sonnet/sonnet-4.5: Balanced (default)");
    println!("   - opus/opus-4: Most capable");
    println!("\n   Gemini:");
    println!("   - flash: Fast and efficient (default)");
    println!("   - pro: Most capable");
}
