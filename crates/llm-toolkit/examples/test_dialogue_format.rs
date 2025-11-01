use llm_toolkit::agent::dialogue::{Speaker, format_messages_to_prompt};

fn main() {
    // Example 1: Short messages (will use simple format)
    println!("=== Example 1: Short Messages (Simple Format) ===\n");
    let short_messages = vec![
        (
            Speaker::System,
            "Discuss the implementation plan for the new feature".to_string(),
        ),
        (
            Speaker::agent("Alice", "Engineer"),
            "I think we should use microservices architecture".to_string(),
        ),
        (
            Speaker::agent("Bob", "Designer"),
            "The UI needs to be responsive and intuitive".to_string(),
        ),
    ];

    println!("{}", format_messages_to_prompt(&short_messages));

    // Example 2: Long messages (will use multipart format)
    println!("\n\n=== Example 2: Long Messages (Multipart Format) ===\n");
    let long_content = format!(
        "This is a very detailed technical discussion about the architecture. {}",
        "We need to consider scalability, maintainability, and performance. ".repeat(20)
    );
    let long_messages = vec![
        (
            Speaker::System,
            "Review the architecture proposal".to_string(),
        ),
        (Speaker::agent("Alice", "Engineer"), long_content),
        (
            Speaker::agent("Carol", "Marketer"),
            "From a marketing perspective, we should focus on user-facing features.".to_string(),
        ),
    ];

    println!("{}", format_messages_to_prompt(&long_messages));
}
