//! Example: Using Expertise type from llm-toolkit-expertise with Agent macro
//!
//! This example demonstrates the full integration: using llm-toolkit-expertise's
//! Expertise type as the expertise parameter in the #[agent] macro.

use llm_toolkit::agent::Agent;
use llm_toolkit_expertise::{
    Anchor, ContextProfile, Expertise, KnowledgeFragment, Priority, TaskHealth, WeightedFragment,
};
use llm_toolkit_macros::agent;

// Create a function that builds our expertise
fn create_rust_reviewer_expertise() -> Expertise {
    Expertise::new("rust-code-reviewer", "1.0.0")
        .with_tag("lang:rust")
        .with_tag("role:reviewer")
        .with_tag("style:thorough")
        // Critical: Always verify compilation
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "ALWAYS run `cargo check` before reviewing code. Never review code that doesn't compile.".to_string(),
            ))
            .with_priority(Priority::Critical),
        )
        // High: Security checks for at-risk tasks
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Perform security vulnerability scan".to_string(),
                steps: vec![
                    "Check for SQL injection vulnerabilities".to_string(),
                    "Verify input validation and sanitization".to_string(),
                    "Review error handling for information leakage".to_string(),
                    "Check for unsafe code blocks and justify their usage".to_string(),
                ],
            })
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec!["security-review".to_string()],
                user_states: vec![],
                task_health: Some(TaskHealth::AtRisk),
            }),
        )
        // Normal: Code quality guidelines
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Guideline {
                rule: "Prefer explicit error handling over unwrap/expect".to_string(),
                anchors: vec![Anchor {
                    context: "Parsing user input".to_string(),
                    positive: "let value = parse_input(s).map_err(|e| Error::InvalidInput(e))?;".to_string(),
                    negative: "let value = parse_input(s).unwrap();".to_string(),
                    reason: "Unwrap can panic. Use proper error handling for user input.".to_string(),
                }],
            })
            .with_priority(Priority::Normal),
        )
        // Low: Documentation suggestions
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Consider adding doc comments for public APIs. Use `cargo doc` to preview.".to_string(),
            ))
            .with_priority(Priority::Low),
        )
}

// Now use the Expertise with the #[agent] macro!
#[agent(expertise = create_rust_reviewer_expertise(), output = "String")]
struct RustCodeReviewer;

fn main() {
    println!("=== Agent with Expertise Integration Example ===\n");

    // Create the agent
    let agent = RustCodeReviewer::default();

    println!("Agent created successfully!\n");
    println!("--- Agent Expertise (generated from Expertise type) ---\n");
    println!("{}", agent.expertise());

    println!("\n--- Key Benefits ---");
    println!("✅ Rich, structured expertise definition");
    println!("✅ Priority-based knowledge ordering");
    println!("✅ Context-aware fragment activation");
    println!("✅ Graph visualization support");
    println!("✅ JSON Schema export");
    println!("✅ Seamless integration with llm-toolkit Agent");

    println!("\n--- How it works ---");
    println!("1. Define Expertise with weighted fragments");
    println!("2. Use expertise = function_call() in #[agent] macro");
    println!("3. Macro calls .to_prompt() and caches result");
    println!("4. Agent automatically uses the expertise!");
}
