//! Test example for automatic JSON enforcement feature.
//!
//! This example demonstrates how the #[derive(Agent)] macro automatically
//! adds JSON schema instructions for structured outputs, but NOT for String outputs.
//!
//! Run with: cargo run --example agent_auto_json_test --features agent

use llm_toolkit::Agent;
use llm_toolkit::agent::Agent as AgentTrait;
use serde::{Deserialize, Serialize};

// Structured output - SHOULD get JSON enforcement
#[derive(Serialize, Deserialize, Debug)]
struct ReviewResult {
    quality_score: u8,
    issues: Vec<String>,
    recommendations: Vec<String>,
}

#[derive(Agent)]
#[agent(
    expertise = "Review code quality and provide structured feedback",
    output = "ReviewResult"
)]
struct CodeReviewAgent;

// String output - should NOT get JSON enforcement
#[derive(Agent)]
#[agent(
    expertise = "Explain code concepts in simple terms",
    output = "String"
)]
struct ExplainerAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🧪 Auto-JSON Enforcement Test\n");
    println!("{}", "=".repeat(70));

    // Test 1: Structured output agent
    let reviewer = CodeReviewAgent;
    println!("\n📊 Agent with STRUCTURED output (ReviewResult):");
    println!("---");
    println!("{}", AgentTrait::expertise(&reviewer));
    println!("---");

    assert!(
        AgentTrait::expertise(&reviewer).contains("IMPORTANT"),
        "Structured output should have JSON enforcement"
    );
    assert!(
        AgentTrait::expertise(&reviewer).contains("ReviewResult"),
        "Should mention the output type name"
    );

    println!("\n✅ PASS: JSON enforcement added for structured output\n");

    // Test 2: String output agent
    println!("{}", "=".repeat(70));
    println!("\n📝 Agent with STRING output:");
    println!("---");
    let explainer = ExplainerAgent;
    println!("{}", AgentTrait::expertise(&explainer));
    println!("---");

    assert!(
        !AgentTrait::expertise(&explainer).contains("IMPORTANT"),
        "String output should NOT have JSON enforcement"
    );
    assert!(
        !AgentTrait::expertise(&explainer).contains("JSON"),
        "String output should not mention JSON"
    );

    println!("\n✅ PASS: No JSON enforcement for String output\n");
    println!("{}", "=".repeat(70));
    println!("\n🎉 All tests passed! Auto-JSON enforcement working correctly.");
}
