//! Example demonstrating ExecutionProfile usage with different agents.
//!
//! This example shows how to configure agents with different execution profiles
//! (Creative, Balanced, Deterministic) to control their behavior.

use llm_toolkit::agent::impls::{ClaudeCodeAgent, GeminiAgent};
use llm_toolkit::agent::{Agent, ExecutionProfile};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger to see the profile settings in action
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    println!("=== ExecutionProfile Demo ===\n");

    // 1. ClaudeCodeAgent with Creative profile
    println!("1. ClaudeCodeAgent with Creative profile:");
    println!("   Note: Claude CLI doesn't support parameter tuning, so this will log a warning");
    let claude_creative = ClaudeCodeAgent::new().with_execution_profile(ExecutionProfile::Creative);

    println!("   Agent: {}", claude_creative.name());
    println!("   Expertise: {}\n", claude_creative.expertise());

    // 2. ClaudeCodeAgent with Balanced profile (default)
    println!("2. ClaudeCodeAgent with Balanced profile (default):");
    let claude_balanced = ClaudeCodeAgent::new().with_execution_profile(ExecutionProfile::Balanced);

    println!("   Agent: {}", claude_balanced.name());
    println!("   Expertise: {}\n", claude_balanced.expertise());

    // 3. ClaudeCodeAgent with Deterministic profile
    println!("3. ClaudeCodeAgent with Deterministic profile:");
    let claude_deterministic =
        ClaudeCodeAgent::new().with_execution_profile(ExecutionProfile::Deterministic);

    println!("   Agent: {}", claude_deterministic.name());
    println!("   Expertise: {}\n", claude_deterministic.expertise());

    // 4. GeminiAgent with Creative profile
    println!("4. GeminiAgent with Creative profile:");
    println!("   This will use temperature=0.9, top_p=0.95");
    let gemini_creative = GeminiAgent::new().with_execution_profile(ExecutionProfile::Creative);

    println!("   Agent: {}", gemini_creative.name());
    println!("   Expertise: {}\n", gemini_creative.expertise());

    // 5. GeminiAgent with Deterministic profile
    println!("5. GeminiAgent with Deterministic profile:");
    println!("   This will use temperature=0.1, top_p=0.8");
    let gemini_deterministic =
        GeminiAgent::new().with_execution_profile(ExecutionProfile::Deterministic);

    println!("   Agent: {}", gemini_deterministic.name());
    println!("   Expertise: {}\n", gemini_deterministic.expertise());

    // 6. Demonstrating builder pattern chaining
    println!("6. Builder pattern with multiple configurations:");
    let _gemini_custom = GeminiAgent::new()
        .with_model_str("pro")
        .with_execution_profile(ExecutionProfile::Creative)
        .with_system_prompt("You are a creative assistant");

    println!("   GeminiAgent configured with Pro model + Creative profile + custom system prompt");

    println!("\n=== Demo Complete ===");
    println!("\nKey Points:");
    println!("- ExecutionProfile provides a semantic way to configure agent behavior");
    println!("- ClaudeCodeAgent: Logs profile but doesn't apply parameters (CLI limitation)");
    println!("- GeminiAgent: Converts profile to actual temperature/top_p parameters");
    println!("- Creative: High randomness, diverse outputs");
    println!("- Balanced: Moderate randomness (default)");
    println!("- Deterministic: Low randomness, consistent outputs");

    Ok(())
}
