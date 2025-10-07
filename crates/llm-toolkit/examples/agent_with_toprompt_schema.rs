//! Example demonstrating ToPrompt::prompt_schema() integration with Agent macro.
//!
//! This shows how the Agent macro automatically uses detailed schemas when
//! the output type implements ToPrompt.
//!
//! Run with: cargo run --example agent_with_toprompt_schema --features agent,derive

use llm_toolkit::agent::Agent as AgentTrait;
use llm_toolkit::{Agent, ToPrompt};
use serde::{Deserialize, Serialize};

// Case 1: With ToPrompt + mode - should get detailed schema
#[derive(Serialize, Deserialize, Debug, ToPrompt)]
#[prompt(mode = "full")]
struct DetailedReview {
    /// Overall quality score from 0 to 100
    quality_score: u8,

    /// List of identified issues
    issues: Vec<String>,

    /// Actionable recommendations for improvement
    recommendations: Vec<String>,
}

#[derive(Agent)]
#[agent(
    expertise = "Review code quality and provide detailed feedback",
    output = "DetailedReview"
)]
struct DetailedReviewAgent;

// Case 2: With ToPrompt + mode but no doc comments - should get basic field list
#[derive(Serialize, Deserialize, Debug, ToPrompt)]
#[prompt(mode = "full")]
struct SimpleReview {
    score: u8,
    comments: Vec<String>,
}

#[derive(Agent)]
#[agent(expertise = "Provide simple code review", output = "SimpleReview")]
struct SimpleReviewAgent;

// Case 3: String output - no JSON enforcement
#[derive(Agent)]
#[agent(expertise = "Explain code in plain language", output = "String")]
struct ExplainerAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🔬 ToPrompt Schema Integration Test\n");
    println!("{}", "=".repeat(80));

    // Test 1: With ToPrompt
    println!("\n📊 Case 1: Agent with ToPrompt-derived output");
    println!("{}", "-".repeat(80));
    let agent1 = DetailedReviewAgent;
    let expertise1 = AgentTrait::expertise(&agent1);
    println!("{}", expertise1);
    println!("{}", "-".repeat(80));

    assert!(
        expertise1.contains("DetailedReview"),
        "Should contain type name in schema"
    );
    assert!(
        expertise1.contains("quality_score"),
        "Should contain field names from ToPrompt"
    );
    assert!(
        expertise1.contains("Overall quality score"),
        "Should contain doc comments from ToPrompt"
    );
    println!("\n✅ PASS: Detailed schema from ToPrompt::prompt_schema()");

    // Test 2: Without doc comments
    println!("\n{}", "=".repeat(80));
    println!("\n📝 Case 2: ToPrompt without doc comments");
    println!("{}", "-".repeat(80));
    let agent2 = SimpleReviewAgent;
    let expertise2 = AgentTrait::expertise(&agent2);
    println!("{}", expertise2);
    println!("{}", "-".repeat(80));

    assert!(
        expertise2.contains("SimpleReview"),
        "Should mention type name"
    );
    assert!(
        expertise2.contains("IMPORTANT"),
        "Should have JSON instruction"
    );
    assert!(
        expertise2.contains("score") || expertise2.contains("SimpleReview:"),
        "Should have field names from ToPrompt"
    );
    println!("\n✅ PASS: ToPrompt schema without doc comments");

    // Test 3: String output
    println!("\n{}", "=".repeat(80));
    println!("\n📄 Case 3: String output (no JSON enforcement)");
    println!("{}", "-".repeat(80));
    let agent3 = ExplainerAgent;
    let expertise3 = AgentTrait::expertise(&agent3);
    println!("{}", expertise3);
    println!("{}", "-".repeat(80));

    assert!(
        !expertise3.contains("IMPORTANT"),
        "Should NOT have JSON instruction"
    );
    assert!(!expertise3.contains("JSON"), "Should not mention JSON");
    println!("\n✅ PASS: No JSON enforcement for String output");

    println!("\n{}", "=".repeat(80));
    println!("\n🎉 All tests passed!");
    println!("\n📚 Summary:");
    println!("  • ToPrompt with doc comments → Detailed schema with descriptions");
    println!("  • ToPrompt without doc comments → Field names only");
    println!("  • String output → No JSON enforcement");
}
