//! Example: Creating Dialogues using DialogueBlueprint
//!
//! This example demonstrates the new Dialogue API for multi-agent conversations:
//! 1. Creating a Dialogue with pre-defined participants (manual team setup)
//! 2. Creating a Dialogue with LLM-generated participants (automatic team generation)
//! 3. Dynamic participant management (adding/removing guests)
//! 4. Understanding structured output (Vec<DialogueTurn>)

use llm_toolkit::agent::dialogue::DialogueBlueprint;
use llm_toolkit::agent::persona::Persona;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== Dialogue API Examples ===\n");

    // ============================================================================
    // USE CASE 1: Pre-defined Participants (Manual Team Setup)
    // ============================================================================
    //
    // When you have a specific team in mind and want full control over the
    // personas, you can define them manually and pass them to the blueprint.
    // This approach skips LLM generation and is useful for:
    // - Reusing a proven team composition
    // - Loading personas from JSON files
    // - Having precise control over team members
    // - Testing with specific persona configurations
    //
    println!("## Use Case 1: Pre-defined Participants\n");

    // Define the team manually
    let hr_saas_team = vec![
        Persona {
            name: "Sarah Johnson".to_string(),
            role: "Product Owner".to_string(),
            background: "10 years in B2B SaaS product management, specializing in HR tech. \
                         Deep understanding of HR manager pain points and workflow optimization."
                .to_string(),
            communication_style: "Strategic and user-focused. Asks clarifying questions about \
                                 business value. Pushes for MVP scope while maintaining vision."
                .to_string(),
        },
        Persona {
            name: "Marcus Lee".to_string(),
            role: "Senior UX Designer".to_string(),
            background: "8 years designing enterprise SaaS products. Expert in human-centered \
                         design. Previously led UX redesign for a major HRIS platform."
                .to_string(),
            communication_style: "Visual and empathetic. Uses concrete user scenarios. \
                                 Advocates strongly for simplicity and intuitive workflows."
                .to_string(),
        },
        Persona {
            name: "Priya Patel".to_string(),
            role: "Senior Backend Engineer".to_string(),
            background: "12 years building scalable web services. Expert in REST API design, \
                         PostgreSQL, and microservices architecture."
                .to_string(),
            communication_style: "Technical and precise. Focuses on data models, API contracts, \
                                 and system scalability. Pragmatic about trade-offs."
                .to_string(),
        },
        Persona {
            name: "Kenji Tanaka".to_string(),
            role: "Senior Frontend Engineer".to_string(),
            background: "9 years specializing in React and modern frontend architecture. \
                         Strong focus on component reusability and accessibility."
                .to_string(),
            communication_style: "Detail-oriented and collaborative. Thinks in terms of \
                                 components and state management. Values maintainability."
                .to_string(),
        },
        Persona {
            name: "Elena Rodriguez".to_string(),
            role: "QA Engineer".to_string(),
            background: "6 years in quality assurance for SaaS products. Expert in test \
                         automation and regression testing."
                .to_string(),
            communication_style: "Systematic and thorough. Thinks about edge cases and error \
                                 states. Asks 'what if' questions to uncover potential issues."
                .to_string(),
        },
    ];

    // Create a blueprint with pre-defined participants
    let blueprint_with_team = DialogueBlueprint {
        agenda: "1on1 Feature Planning".to_string(),
        context: "Product planning meeting for new 1on1 feature in HR SaaS platform. \
                  Focus on MVP scope, user workflows, and technical architecture."
            .to_string(),
        participants: Some(hr_saas_team), // Pre-defined team - no LLM generation
        execution_strategy: Some("broadcast".to_string()),
    };

    println!("Blueprint created with pre-defined team:");
    println!("  Agenda: {}", blueprint_with_team.agenda);
    println!(
        "  Participants: {} members",
        blueprint_with_team.participants.as_ref().unwrap().len()
    );
    println!(
        "  Strategy: {}",
        blueprint_with_team.execution_strategy.as_ref().unwrap()
    );
    println!();

    // In a real application, you would create the dialogue like this:
    println!("💡 To create a dialogue from this blueprint:");
    println!("   use llm_toolkit::agent::impls::{{ClaudeCodeAgent, ClaudeCodeJsonAgent}};");
    println!();
    println!("   let dialogue = Dialogue::from_blueprint(");
    println!("       blueprint_with_team,");
    println!("       ClaudeCodeJsonAgent::new(),  // Generator (not used for pre-defined team)");
    println!("       ClaudeCodeAgent::new(),       // LLM for dialogue execution");
    println!("   ).await?;");
    println!();
    println!("   // Run the dialogue");
    println!("   let turns = dialogue.run(\"Discuss MVP scope for 1on1 feature\").await?;");
    println!();
    println!("   // The output is Vec<DialogueTurn>, each containing:");
    println!("   //   - participant_name: String (e.g., \"Sarah Johnson\")");
    println!("   //   - content: String (the participant's response)");
    println!();
    println!("   for turn in turns {{");
    println!("       println!(\"[{{}}]: {{}}\", turn.participant_name, turn.content);");
    println!("   }}");
    println!();
    println!("---\n");

    // ============================================================================
    // USE CASE 2: LLM-Generated Participants (Automatic Team Generation)
    // ============================================================================
    //
    // When you want the LLM to dynamically create the best team for your task,
    // set participants to None. The generator_agent will analyze the context
    // and create a PersonaTeam with appropriate roles and expertise.
    //
    // This is ideal for:
    // - Exploring different team compositions
    // - Quick prototyping without manual persona creation
    // - Letting the AI determine optimal team structure
    //
    println!("## Use Case 2: LLM-Generated Participants\n");

    let blueprint_auto_generated = DialogueBlueprint {
        agenda: "Security Audit Review".to_string(),
        context: "We need to review the security architecture of our authentication system. \
                  The team should include experts in: application security, infrastructure, \
                  compliance, and development. They should identify vulnerabilities, suggest \
                  improvements, and ensure we meet SOC2 requirements."
            .to_string(),
        participants: None, // No pre-defined team - will be LLM-generated
        execution_strategy: Some("broadcast".to_string()),
    };

    println!("Blueprint created for LLM-generated team:");
    println!("  Agenda: {}", blueprint_auto_generated.agenda);
    println!("  Participants: None (will be auto-generated)");
    println!("  Context describes required expertise:");
    println!("    - Application security expert");
    println!("    - Infrastructure specialist");
    println!("    - Compliance officer");
    println!("    - Developer perspective");
    println!();

    println!("💡 When participants is None, the generator_agent creates the team:");
    println!();
    println!("   let dialogue = Dialogue::from_blueprint(");
    println!("       blueprint_auto_generated,");
    println!("       ClaudeCodeJsonAgent::new(),  // Used to generate PersonaTeam from context");
    println!("       ClaudeCodeAgent::new(),");
    println!("   ).await?;");
    println!();
    println!("   // The generator_agent will:");
    println!("   // 1. Analyze the context and required expertise");
    println!("   // 2. Generate a PersonaTeam with 4-6 appropriate personas");
    println!("   // 3. Each persona will have realistic name, role, background, and style");
    println!("   // 4. The dialogue is ready to use with the auto-generated team");
    println!();
    println!("---\n");

    // ============================================================================
    // USE CASE 3: Dynamic Participant Management
    // ============================================================================
    //
    // One key feature of Dialogue is the ability to add/remove participants
    // at runtime. This is useful for guest speakers, domain experts, or
    // stakeholders who should only be involved in specific parts of the discussion.
    //
    println!("## Use Case 3: Dynamic Participant Management\n");

    println!("Suppose we have a running dialogue with our core dev team...");
    println!();

    // Example: Adding a guest participant
    let customer_success_guest = Persona {
        name: "Jessica Martinez".to_string(),
        role: "Customer Success Manager (Guest)".to_string(),
        background: "6 years managing enterprise HR customers. Direct contact with 40+ \
                     mid-to-large companies. Aggregates feedback from quarterly reviews, \
                     support tickets, and user interviews."
            .to_string(),
        communication_style: "Customer-centric and data-backed. Speaks in terms of actual \
                             customer quotes and real usage patterns. Highlights friction \
                             points based on support tickets."
            .to_string(),
    };

    println!("💡 Adding a guest participant:");
    println!();
    println!("   // Bring in Customer Success for specific customer feedback");
    println!("   let guest = Persona {{ /* ... */ }};");
    println!("   dialogue.add_participant(guest, ClaudeCodeAgent::new());");
    println!();
    println!("   // Now the dialogue has N+1 participants");
    println!("   println!(\"Team size: {{}}\", dialogue.participant_count());");
    println!();
    println!("   // Run discussion with guest present");
    println!("   let turns = dialogue.run(\"Review feature from customer perspective\").await?;");
    println!();
    println!("   // Guest participant's turn will be included in the output");
    println!("   // Output: Vec<DialogueTurn> with all participants including the guest");
    println!();

    // Example: Removing a guest participant
    println!("💡 Removing a guest participant (by name):");
    println!();
    println!("   // After getting customer feedback, remove the guest");
    println!("   dialogue.remove_participant(\"Jessica Martinez\")?;");
    println!();
    println!("   // Back to core team");
    println!("   println!(\"Team size: {{}}\", dialogue.participant_count());");
    println!();
    println!("   // Continue with core team only");
    println!("   let next_turns = dialogue.run(\"Define implementation plan\").await?;");
    println!();

    println!("Guest persona details:");
    println!("  Name: {}", customer_success_guest.name);
    println!("  Role: {}", customer_success_guest.role);
    println!("  Background: {}", customer_success_guest.background);
    println!();
    println!("---\n");

    // ============================================================================
    // USE CASE 4: Understanding Structured Output
    // ============================================================================
    //
    // The dialogue.run() method returns Vec<DialogueTurn>, which provides
    // structured access to each participant's contribution.
    //
    println!("## Use Case 4: Structured Output\n");

    println!("💡 The output of dialogue.run() is Vec<DialogueTurn>:");
    println!();
    println!("   pub struct DialogueTurn {{");
    println!("       pub participant_name: String,");
    println!("       pub content: String,");
    println!("   }}");
    println!();
    println!("   // Example usage:");
    println!("   let turns = dialogue.run(\"Discuss API design\").await?;");
    println!();
    println!("   // In broadcast mode: all participants respond");
    println!("   for turn in &turns {{");
    println!("       println!(\"[{{}}]:\", turn.participant_name);");
    println!("       println!(\"{{}}\", turn.content);");
    println!("       println!();");
    println!("   }}");
    println!();
    println!("   // In sequential mode: only final participant's turn is returned");
    println!("   let final_output = &turns[0];");
    println!("   println!(\"Final result from {{}}:\", final_output.participant_name);");
    println!("   println!(\"{{}}\", final_output.content);");
    println!();
    println!("---\n");

    // ============================================================================
    // Summary
    // ============================================================================
    println!("## Summary\n");
    println!("The Dialogue API provides flexible multi-agent conversations:");
    println!();
    println!("1. **Pre-defined teams**: Full control over personas (Use Case 1)");
    println!("2. **LLM-generated teams**: Automatic team creation from context (Use Case 2)");
    println!("3. **Dynamic management**: Add/remove participants at runtime (Use Case 3)");
    println!("4. **Structured output**: Vec<DialogueTurn> with participant names (Use Case 4)");
    println!();
    println!("Choose the approach that best fits your use case!");

    Ok(())
}
