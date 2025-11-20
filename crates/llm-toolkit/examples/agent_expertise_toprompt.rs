//! Example: Agent with ToPrompt expertise
//!
//! This example demonstrates that the #[agent] macro can accept ToPrompt-implementing
//! types (like Expertise) as the expertise parameter.

use llm_toolkit::agent::{Agent};
use llm_toolkit::prompt::ToPrompt;
use llm_toolkit_macros::agent;

// Simple mock type that implements ToPrompt
#[derive(Clone)]
struct MockExpertise {
    description: &'static str,
}

impl ToPrompt for MockExpertise {
    fn to_prompt(&self) -> String {
        format!("Mock Expertise: {}", self.description)
    }
}

// Define a const expertise
const MOCK_EXPERTISE: MockExpertise = MockExpertise {
    description: "I am a mock expertise for testing",
};

// Function that returns expertise
fn get_reviewer_expertise() -> MockExpertise {
    MockExpertise {
        description: "I am a Rust code reviewer with expertise in safety and performance",
    }
}

// Test 1: Agent with const expression
#[agent(expertise = MOCK_EXPERTISE, output = "String")]
struct ConstExpertiseAgent;

// Test 2: Agent with function call
#[agent(expertise = get_reviewer_expertise(), output = "String")]
struct FunctionExpertiseAgent;

// Test 3: Still supports string literals (backward compatibility)
#[agent(expertise = "Simple string expertise", output = "String")]
struct StringExpertiseAgent;

fn main() {
    // All three should compile successfully
    println!("=== Agent with ToPrompt Expertise Example ===\n");
    println!("All agent definitions compiled successfully!\n");

    // Test that expertise() method works
    let agent1 = ConstExpertiseAgent::default();
    println!("1. ConstExpertiseAgent expertise:\n   {}\n", agent1.expertise());

    let agent2 = FunctionExpertiseAgent::default();
    println!("2. FunctionExpertiseAgent expertise:\n   {}\n", agent2.expertise());

    let agent3 = StringExpertiseAgent::default();
    println!("3. StringExpertiseAgent expertise:\n   {}\n", agent3.expertise());

    println!("✅ The #[agent] macro now supports:");
    println!("   - String literals (traditional)");
    println!("   - Const expressions implementing ToPrompt");
    println!("   - Function calls returning ToPrompt types");
}
