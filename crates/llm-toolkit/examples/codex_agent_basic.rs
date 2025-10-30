//! Simple check to verify Codex CLI is available and test basic execution.
//!
//! Run with: cargo run --example codex_agent_basic --features agent

use llm_toolkit::agent::Agent;
use llm_toolkit::agent::impls::CodexAgent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking for Codex CLI...\n");

    if !CodexAgent::is_available() {
        println!("❌ codex CLI not found in PATH");
        println!("\n💡 Please install the Codex CLI:");
        println!("   Visit: https://github.com/codexlang/codex-cli");
        println!("   or run: brew install codex (if available)\n");
        return Ok(());
    }

    println!("✅ Codex CLI found in PATH");
    println!("   Testing basic execution...\n");

    // Test basic execution
    let agent = CodexAgent::new()
        .with_model_str("sonnet")
        .with_approval_policy("never"); // No approval for this test

    let prompt = "What is 2+2? Answer with just the number.";
    println!("🤖 Sending prompt: {}\n", prompt);

    match agent.execute(prompt.to_string().into()).await {
        Ok(response) => {
            println!("✅ CodexAgent execution successful!");
            println!("📝 Response: {}\n", response.trim());
        }
        Err(e) => {
            println!("❌ CodexAgent execution failed: {}", e);
            println!("\n💡 This might be due to:");
            println!("   - Codex CLI not properly configured");
            println!("   - API key not set");
            println!("   - Network issues\n");
        }
    }

    Ok(())
}
