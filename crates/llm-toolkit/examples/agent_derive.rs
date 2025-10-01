//! Example demonstrating the Agent derive macro.
//!
//! This example shows how to use #[derive(Agent)] to automatically implement
//! the Agent trait for custom types with structured outputs.
//!
//! Run with: cargo run --example agent_derive --features agent

use llm_toolkit::{Agent, agent::Agent as AgentTrait};
use serde::{Deserialize, Serialize};

// Define a structured output type for the agent
#[derive(Serialize, Deserialize, Debug)]
struct ArticleSummary {
    title: String,
    key_points: Vec<String>,
    word_count: usize,
}

// Use the Agent derive macro to automatically implement the Agent trait
#[derive(Agent)]
#[agent(
    expertise = "Summarizing articles and extracting key information",
    output = "ArticleSummary"
)]
struct ArticleSummarizerAgent;

// Another example with a different output type
#[derive(Serialize, Deserialize, Debug)]
struct CodeReview {
    overall_quality: String,
    issues: Vec<String>,
    suggestions: Vec<String>,
}

#[derive(Agent)]
#[agent(
    expertise = "Reviewing code for quality, bugs, and best practices",
    output = "CodeReview"
)]
struct CodeReviewerAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🤖 Agent Derive Macro Example\n");

    // Create agent instances
    let summarizer = ArticleSummarizerAgent;
    let reviewer = CodeReviewerAgent;

    // Show agent information
    println!("📝 Article Summarizer Agent");
    println!("   Expertise: {}", AgentTrait::expertise(&summarizer));
    println!("   Output Type: ArticleSummary\n");

    println!("🔍 Code Reviewer Agent");
    println!("   Expertise: {}", AgentTrait::expertise(&reviewer));
    println!("   Output Type: CodeReview\n");

    println!("✨ The Agent derive macro automatically implements:");
    println!("   - Agent::expertise() method");
    println!("   - Agent::execute() method with structured JSON output");
    println!("   - Type-safe deserialization to the specified output type\n");

    println!("💡 To use these agents, call agent.execute(intent).await");
    println!("   The agent will use ClaudeCodeAgent internally and parse");
    println!("   the response into the specified output type.");
}
