//! Simple check to verify claude CLI is available.
//!
//! Run with: cargo run --example check_claude --features agent

use llm_toolkit::agent::impls::ClaudeCodeAgent;

fn main() {
    println!("🔍 Checking for claude CLI...\n");

    if ClaudeCodeAgent::is_available() {
        println!("✅ claude CLI found in PATH");
        println!("   The orchestrator should work correctly.\n");
    } else {
        println!("❌ claude CLI not found in PATH");
        println!("\n💡 Please install the Claude CLI:");
        println!("   npm install -g @anthropic-ai/cli");
        println!("   or visit: https://github.com/anthropics/anthropic-sdk-typescript\n");
        // std::process::exit(1);
    }
}
