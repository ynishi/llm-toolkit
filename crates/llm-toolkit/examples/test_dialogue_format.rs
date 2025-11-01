use llm_toolkit::agent::dialogue::turn_input::*;

fn main() {
    let participants = vec![
        ParticipantInfo::new("Alice", "Engineer", "Expert in system architecture"),
        ParticipantInfo::new("Bob", "Designer", "Focuses on user experience"),
        ParticipantInfo::new("Carol", "Marketer", "Marketing strategy specialist"),
    ];

    let context = vec![
        ContextMessage::with_metadata(
            "Alice",
            "Engineer",
            "I think we should use microservices architecture",
            1,
            1699000000,
        ),
        ContextMessage::with_metadata(
            "Bob",
            "Designer",
            "The UI needs to be responsive and intuitive",
            1,
            1699000010,
        ),
    ];

    let turn_input = TurnInput::with_dialogue_context(
        "Discuss the implementation plan for the new feature",
        context,
        participants,
        "Bob",
    );

    println!("=== Simple Formatter ===\n");
    println!(
        "{}",
        turn_input.to_prompt_with_formatter(&SimpleContextFormatter)
    );

    println!("\n\n=== Multipart Formatter ===\n");
    println!(
        "{}",
        turn_input.to_prompt_with_formatter(&MultipartContextFormatter)
    );
}
