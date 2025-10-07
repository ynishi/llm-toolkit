//! Test that String output doesn't get JSON schema enforcement

use llm_toolkit::agent::Agent;

#[llm_toolkit_macros::agent(
    expertise = "Explaining concepts in simple terms",
    output = "String",
    backend = "claude"
)]
struct ExplainerAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = ExplainerAgent::default();

    println!("=== String Output Test ===\n");
    println!("Agent: {}", agent.name());
    println!("Expertise:\n{}\n", agent.expertise());

    // Should NOT contain JSON schema instructions
    assert!(!agent.expertise().contains("JSON"));
    assert!(!agent.expertise().contains("json"));

    println!("✅ PASS: No JSON schema for String output");

    Ok(())
}
