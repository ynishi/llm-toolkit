//! Example demonstrating #[agent(profile = ...)] attribute usage.

use llm_toolkit::agent::Agent;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Summary {
    title: String,
    key_points: Vec<String>,
}

#[llm_toolkit_macros::agent(
    expertise = "Summarizing content creatively",
    output = "Summary",
    backend = "gemini",
    profile = "Creative"
)]
struct CreativeSummarizerAgent;

#[llm_toolkit_macros::agent(
    expertise = "Summarizing content precisely",
    output = "Summary",
    backend = "gemini",
    profile = "Deterministic"
)]
struct PreciseSummarizerAgent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    println!("=== Agent with Profile Attribute Demo ===\n");

    // 1. Creative profile agent
    println!("1. CreativeSummarizerAgent (profile = Creative):");
    let creative_agent = CreativeSummarizerAgent::default();
    println!("   Agent: {}", creative_agent.name());
    println!("   Expertise: {}\n", creative_agent.expertise());

    // 2. Deterministic profile agent
    println!("2. PreciseSummarizerAgent (profile = Deterministic):");
    let precise_agent = PreciseSummarizerAgent::default();
    println!("   Agent: {}", precise_agent.name());
    println!("   Expertise: {}\n", precise_agent.expertise());

    println!("=== Demo Complete ===");
    println!("\nThe agents are configured with different ExecutionProfile settings:");
    println!("- Creative: temperature=0.9, top_p=0.95 (more diverse outputs)");
    println!("- Deterministic: temperature=0.1, top_p=0.8 (more consistent outputs)");

    Ok(())
}
