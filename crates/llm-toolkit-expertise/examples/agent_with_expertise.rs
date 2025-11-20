//! Example: Using Expertise with Agent macro
//!
//! Demonstrates how to use llm-toolkit-expertise's Expertise type
//! with the #[agent(expertise = ...)] macro attribute.

use llm_toolkit_expertise::{
    ContextProfile, Expertise, KnowledgeFragment, Priority, TaskHealth, WeightedFragment, Anchor,
};

// Create an Expertise dynamically
fn create_code_reviewer() -> Expertise {
    Expertise::new("rust-code-reviewer", "1.0.0")
        .with_tag("lang:rust")
        .with_tag("role:reviewer")
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
                    "Check for unsafe code blocks".to_string(),
                    "Verify input validation and sanitization".to_string(),
                    "Review error handling for information leakage".to_string(),
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
}

fn main() {
    println!("=== Expertise-Based Agent Example ===\n");

    // Create expertise
    let expertise = create_code_reviewer();

    // Display the expertise
    println!("--- Generated Expertise Prompt ---\n");
    println!("{}", expertise.to_prompt());

    println!("\n--- Tree Visualization ---\n");
    println!("{}", expertise.to_tree());

    println!("\n=== Usage with Agent Macro ===\n");
    println!("With the ToPrompt integration, you can now use:");
    println!();
    println!("```rust");
    println!("const EXPERTISE: Expertise = create_code_reviewer();");
    println!();
    println!("#[derive(Agent)]");
    println!("#[agent(expertise = EXPERTISE, output = \"String\")]");
    println!("struct CodeReviewerAgent;");
    println!("```");
    println!();
    println!("The expertise() method will automatically call EXPERTISE.to_prompt()");
    println!("and cache the result for efficiency!");
}
