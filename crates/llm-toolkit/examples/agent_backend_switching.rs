//! Example demonstrating backend switching in the Agent derive macro.
//!
//! This example shows how to use the `backend` attribute to switch between
//! different LLM providers (Claude, Gemini) while keeping the same interface.
//!
//! Run with: cargo run --example agent_backend_switching --features agent

use llm_toolkit::{Agent, agent::Agent as AgentTrait};
use serde::{Deserialize, Serialize};

// Define a structured output type
#[derive(Serialize, Deserialize, Debug, llm_toolkit::ToPrompt)]
#[prompt(mode = "full")]
struct TechExplanation {
    topic: String,
    key_points: Vec<String>,
    complexity: String,
}

// Agent using Claude backend (default)
#[derive(Agent)]
#[agent(
    expertise = "Explaining technical concepts in simple terms",
    output = "TechExplanation"
)]
struct ClaudeExplainerAgent;

// Agent using Gemini backend with Flash model
#[derive(Agent)]
#[agent(
    expertise = "Explaining technical concepts in simple terms",
    output = "TechExplanation",
    backend = "gemini",
    model = "flash"
)]
struct GeminiFlashExplainerAgent;

// Agent using Gemini backend with Pro model
#[derive(Agent)]
#[agent(
    expertise = "Explaining technical concepts in simple terms",
    output = "TechExplanation",
    backend = "gemini",
    model = "pro"
)]
struct GeminiProExplainerAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🔄 Agent Backend Switching Example\n");

    // Create agent instances
    let claude_agent = ClaudeExplainerAgent;
    let gemini_flash_agent = GeminiFlashExplainerAgent;
    let gemini_pro_agent = GeminiProExplainerAgent;

    println!("📊 Available Agents:\n");

    println!("1. Claude Agent (default)");
    println!("   Backend: claude");
    println!("   Expertise: {}\n", AgentTrait::expertise(&claude_agent));

    println!("2. Gemini Flash Agent");
    println!("   Backend: gemini");
    println!("   Model: gemini-2.5-flash");
    println!(
        "   Expertise: {}\n",
        AgentTrait::expertise(&gemini_flash_agent)
    );

    println!("3. Gemini Pro Agent");
    println!("   Backend: gemini");
    println!("   Model: gemini-2.5-pro");
    println!(
        "   Expertise: {}\n",
        AgentTrait::expertise(&gemini_pro_agent)
    );

    println!("✨ Key Features:");
    println!("   - Same Agent trait interface across all backends");
    println!("   - Type-safe structured outputs");
    println!("   - Compile-time backend selection");
    println!("   - Easy model configuration\n");

    println!("💡 Usage:");
    println!("   To use these agents, call agent.execute(intent).await");
    println!("   All agents return TechExplanation regardless of backend\n");

    println!("🔧 Configuration:");
    println!("   - Default backend: claude (ClaudeCodeAgent)");
    println!("   - Gemini backend: requires `gemini` CLI in PATH");
    println!("   - Models: flash (fast), pro (capable)");
}
