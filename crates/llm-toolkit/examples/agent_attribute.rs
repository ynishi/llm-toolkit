//! Example demonstrating the Agent attribute macro with Generic support.
//!
//! This example shows how to use #[agent(...)] attribute macro which generates
//! a struct with Generic inner agent, allowing for agent injection.
//!
//! Run with: cargo run --example agent_attribute --features agent

use llm_toolkit::agent::impls::ClaudeCodeAgent;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct TechnicalDoc {
    title: String,
    sections: Vec<String>,
    code_examples: Vec<String>,
}

// Using attribute macro - generates Generic struct with injection support
#[llm_toolkit_macros::agent(
    expertise = "Writing technical documentation with code examples",
    output = "TechnicalDoc"
)]
struct DocWriterAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("📄 Agent Attribute Macro Example\n");

    // Method 1: Using Default (most common)
    let _agent1 = DocWriterAgent::default();
    println!("✅ Method 1: Default");
    println!("   let agent = DocWriterAgent::default();");
    println!("   Uses: ClaudeCodeAgent (default backend)\n");

    // Method 2: Using convenience constructor
    let _agent2 = DocWriterAgent::with_claude_model("opus-4");
    println!("✅ Method 2: Convenience constructor");
    println!("   let agent = DocWriterAgent::with_claude_model(\"opus-4\");");
    println!("   Uses: ClaudeCodeAgent with Opus 4 model\n");

    // Method 3: Inject custom agent
    let custom_claude = ClaudeCodeAgent::new().with_model_str("sonnet-4.5");
    let _agent3 = DocWriterAgent::new(custom_claude);
    println!("✅ Method 3: Custom agent injection");
    println!("   let custom = ClaudeCodeAgent::new().with_model_str(\"sonnet-4.5\");");
    println!("   let agent = DocWriterAgent::new(custom);");
    println!("   Uses: Custom configured agent\n");

    println!("📊 Key Features:");
    println!("   - Generic struct with default type parameter");
    println!("   - Agent injection for testing and customization");
    println!("   - Static dispatch for performance");
    println!("   - Multiple constructor patterns\n");

    println!("🔬 Testing Example:");
    println!("   // Mock agent for testing");
    println!("   struct MockAgent {{ response: String }}");
    println!("   impl Agent for MockAgent {{ ... }}");
    println!("   ");
    println!("   let mock = MockAgent {{ response: \"...\".to_string() }};");
    println!("   let agent = DocWriterAgent::new(mock); // Inject mock!");

    println!("\n💡 Compare with #[derive(Agent)]:");
    println!("   Derive: Simple, stateless (creates agent each time)");
    println!("   Attribute: Generic, stateful (reuses inner agent)");
}
