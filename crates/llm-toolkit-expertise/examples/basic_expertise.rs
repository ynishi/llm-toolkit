//! Basic expertise creation and usage example

use llm_toolkit_expertise::{
    Anchor, ContextProfile, Expertise, KnowledgeFragment, Priority, TaskHealth, WeightedFragment,
};

fn main() {
    println!("=== Basic Expertise Example ===\n");

    // Create a code review expertise
    let expertise = Expertise::new("rust-code-reviewer", "1.0.0")
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
        // Quality standards
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::QualityStandard {
                criteria: vec![
                    "Code compiles without warnings".to_string(),
                    "All tests pass".to_string(),
                    "No clippy warnings".to_string(),
                    "Public APIs have documentation".to_string(),
                ],
                passing_grade: "All criteria must be met before approval".to_string(),
            })
            .with_priority(Priority::High),
        );

    // Display expertise in various formats
    println!("--- Tree Visualization ---\n");
    println!("{}", expertise.to_tree());

    println!("\n--- Generated Prompt (All Contexts) ---\n");
    println!("{}", expertise.to_prompt());

    println!("\n--- Generated Prompt (Security Review, At Risk) ---\n");
    use llm_toolkit_expertise::ContextMatcher;
    let security_context = ContextMatcher::new()
        .with_task_type("security-review")
        .with_task_health(TaskHealth::AtRisk);
    println!("{}", expertise.to_prompt_with_context(&security_context));

    println!("\n--- Mermaid Graph ---\n");
    println!("{}", expertise.to_mermaid());

    println!("\n--- JSON Serialization ---\n");
    let json = serde_json::to_string_pretty(&expertise).expect("Failed to serialize");
    println!("{}", json);
}
