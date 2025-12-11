//! Example demonstrating model selection for different backends.
//!
//! This example shows how to use the `model` attribute to select
//! specific models for Claude, Gemini, and Codex backends.
//!
//! Run with: cargo run --example agent_model_selection --features agent

#![allow(deprecated)]

use llm_toolkit::{Agent, agent::Agent as AgentTrait};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, llm_toolkit::ToPrompt)]
#[prompt(mode = "full")]
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

// Codex with GPT-5.1-Codex (default, optimized for coding)
#[derive(Agent)]
#[agent(
    expertise = "Analyzing code quality and patterns with deep coding focus",
    output = "CodeAnalysis",
    backend = "codex",
    model = "gpt-5.1-codex"
)]
struct CodexGpt51Agent;

// Codex with GPT-5.1-Codex-Mini (cost-effective)
#[derive(Agent)]
#[agent(
    expertise = "Analyzing code quality and patterns with deep coding focus",
    output = "CodeAnalysis",
    backend = "codex",
    model = "gpt-5.1-codex-mini"
)]
struct CodexGpt51MiniAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🎯 Agent Model Selection Example\n");

    let sonnet = ClaudeSonnetAgent;
    let opus = ClaudeOpusAgent;
    let flash = GeminiFlashAgent;
    let pro = GeminiProAgent;
    let codex_gpt51 = CodexGpt51Agent;
    let codex_gpt51_mini = CodexGpt51MiniAgent;

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

    println!("5. Codex GPT-5.1-Codex");
    println!("   Backend: codex");
    println!("   Model: gpt-5.1-codex");
    println!("   Profile: Optimized for long-running, agentic coding tasks");
    println!("   Expertise: {}\n", AgentTrait::expertise(&codex_gpt51));

    println!("6. Codex GPT-5.1-Codex-Mini");
    println!("   Backend: codex");
    println!("   Model: gpt-5.1-codex-mini");
    println!("   Profile: Smaller, more cost-effective version");
    println!(
        "   Expertise: {}\n",
        AgentTrait::expertise(&codex_gpt51_mini)
    );

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
    println!("\n   Codex:");
    println!("   - gpt-5.1-codex: Optimized for agentic coding (default for macOS/Linux)");
    println!("   - gpt-5.1-codex-mini: Cost-effective alternative");
    println!("   - gpt-5.1: General coding and agentic tasks (default for Windows)");
}
