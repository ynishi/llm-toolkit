//! Example: Dialogue Session Resumption with History Injection
//!
//! This example demonstrates how to save and resume dialogue sessions using
//! the history injection feature. This is useful for:
//! - Persistent multi-turn conversations across process restarts
//! - Saving dialogue state for later analysis
//! - Implementing stateful chatbots with session management
//!
//! The example shows:
//! 1. Creating an initial dialogue session
//! 2. Running a conversation and saving the history
//! 3. Resuming the conversation from saved history
//! 4. Continuing the dialogue with full context

use async_trait::async_trait;
use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use std::sync::{Arc, Mutex};

// Mock agent that tracks conversation context
#[derive(Clone)]
struct MockConversationAgent {
    name: String,
    call_count: Arc<Mutex<usize>>,
}

impl MockConversationAgent {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            call_count: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait]
impl Agent for MockConversationAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock conversational agent";
        &EXPERTISE
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let input = payload.to_text();

        // Generate response based on call count (simulating conversation flow)
        let response = match *count {
            1 => format!(
                "{}: I understand we're discussing {}. Let me analyze this.",
                self.name, input
            ),
            2 => format!(
                "{}: Building on our previous discussion, here's my perspective on {}.",
                self.name, input
            ),
            3 => format!(
                "{}: Based on our conversation history, I recommend {}.",
                self.name, input
            ),
            _ => format!("{}: Continuing from turn {}: {}.", self.name, *count, input),
        };

        Ok(response)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Dialogue Session Resumption Example ===\n");

    // ============================================================================
    // Session 1: Initial Conversation
    // ============================================================================
    println!("## Session 1: Starting initial conversation\n");

    let persona1 = Persona {
        name: "Alice".to_string(),
        role: "Product Manager".to_string(),
        background: "Experienced PM with focus on user experience".to_string(),
        communication_style: "Strategic and user-focused".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let persona2 = Persona {
        name: "Bob".to_string(),
        role: "Software Engineer".to_string(),
        background: "Senior engineer specializing in backend systems".to_string(),
        communication_style: "Technical and pragmatic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut session1 = Dialogue::broadcast();
    session1.add_participant(persona1.clone(), MockConversationAgent::new("Alice"));
    session1.add_participant(persona2.clone(), MockConversationAgent::new("Bob"));

    // First discussion
    println!("💬 Prompt: 'Let's discuss the new authentication feature'\n");
    let turns1 = session1
        .run("Let's discuss the new authentication feature")
        .await?;

    for turn in &turns1 {
        println!("[{}]:", turn.speaker.name());
        println!("{}\n", turn.content);
    }

    // Save the session
    let session_file = "dialogue_session.json";
    session1.save_history(session_file)?;
    println!("✅ Session saved to {}\n", session_file);
    println!(
        "📊 History length after Session 1: {} turns\n",
        session1.history().len()
    );
    println!("---\n");

    // ============================================================================
    // Simulate process restart or session end
    // ============================================================================
    println!("⏸️  Simulating process restart...\n");
    println!("---\n");

    // ============================================================================
    // Session 2: Resume Conversation
    // ============================================================================
    println!("## Session 2: Resuming conversation from saved history\n");

    // Load saved history
    let loaded_history = Dialogue::load_history(session_file)?;
    println!("✅ Loaded history from {}", session_file);
    println!("📊 Restored {} turns\n", loaded_history.len());

    // Create new dialogue with injected history
    let mut session2 = Dialogue::broadcast().with_history(loaded_history);

    // Re-add participants (with new agent instances)
    session2.add_participant(persona1.clone(), MockConversationAgent::new("Alice"));
    session2.add_participant(persona2.clone(), MockConversationAgent::new("Bob"));

    println!("📋 Current conversation history:");
    for (i, turn) in session2.history().iter().enumerate() {
        println!(
            "  {}. [{}]: {}",
            i + 1,
            turn.speaker.name(),
            if turn.content.len() > 60 {
                format!("{}...", &turn.content[..60])
            } else {
                turn.content.clone()
            }
        );
    }
    println!();

    // Continue the conversation
    println!("💬 Prompt: 'What are the security implications?'\n");
    let turns2 = session2.run("What are the security implications?").await?;

    for turn in &turns2 {
        println!("[{}]:", turn.speaker.name());
        println!("{}\n", turn.content);
    }

    println!(
        "📊 History length after Session 2: {} turns\n",
        session2.history().len()
    );
    println!("---\n");

    // ============================================================================
    // Session 3: Continue Further
    // ============================================================================
    println!("## Session 3: Continuing the conversation\n");

    println!("💬 Prompt: 'Let's finalize the implementation plan'\n");
    let turns3 = session2
        .run("Let's finalize the implementation plan")
        .await?;

    for turn in &turns3 {
        println!("[{}]:", turn.speaker.name());
        println!("{}\n", turn.content);
    }

    // Save updated history
    session2.save_history(session_file)?;
    println!("✅ Updated session saved to {}", session_file);
    println!(
        "📊 Final history length: {} turns\n",
        session2.history().len()
    );

    // ============================================================================
    // Summary
    // ============================================================================
    println!("---\n");
    println!("## Summary\n");
    println!("Complete conversation history:");
    for (i, turn) in session2.history().iter().enumerate() {
        println!(
            "{}. [{}]: {}",
            i + 1,
            turn.speaker.name(),
            if turn.content.len() > 80 {
                format!("{}...", &turn.content[..80])
            } else {
                turn.content.clone()
            }
        );
    }
    println!();

    println!("Key Features Demonstrated:");
    println!("✅ Session persistence with save_history()");
    println!("✅ Session restoration with load_history()");
    println!("✅ History injection with with_history()");
    println!("✅ Stateful conversations across multiple sessions");
    println!("✅ Context preservation for continuous dialogue");

    // Cleanup
    std::fs::remove_file(session_file)?;
    println!("\n🧹 Cleaned up session file");

    Ok(())
}
