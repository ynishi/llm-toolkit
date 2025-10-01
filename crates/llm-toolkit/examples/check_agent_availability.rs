//! Example demonstrating agent availability checking.
//!
//! This example shows how to use the `is_available()` method to check
//! if agent backends are ready before attempting to use them.
//!
//! Run with: cargo run --example check_agent_availability --features agent

use llm_toolkit::agent::Agent;
use llm_toolkit::agent::impls::{ClaudeCodeAgent, GeminiAgent};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🔍 Agent Availability Checker\n");

    // Check Claude CLI
    println!("📋 Checking Claude CLI...");
    let claude_agent = ClaudeCodeAgent::new();
    match claude_agent.is_available().await {
        Ok(()) => {
            println!("✅ Claude CLI is available and ready");
            println!("   Backend: claude");
            println!("   Command: claude\n");
        }
        Err(e) => {
            println!("❌ Claude CLI is not available");
            println!("   Error: {}", e);
            println!("   Install: npm install -g @anthropic-ai/cli\n");
        }
    }

    // Check Gemini CLI
    println!("📋 Checking Gemini CLI...");
    let gemini_agent = GeminiAgent::new();
    match gemini_agent.is_available().await {
        Ok(()) => {
            println!("✅ Gemini CLI is available and ready");
            println!("   Backend: gemini");
            println!("   Command: gemini\n");
        }
        Err(e) => {
            println!("❌ Gemini CLI is not available");
            println!("   Error: {}", e);
            println!("   Install instructions: See Gemini CLI documentation\n");
        }
    }

    println!("💡 Tip: Use agent.is_available() before executing tasks");
    println!("   to ensure the backend is ready and provide better error messages.");
}
