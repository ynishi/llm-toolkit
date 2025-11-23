//! Example: Dialogue Session Resumption with System Prompt Approach
//!
//! This example demonstrates session resumption using the simpler
//! `with_history_as_system_prompt()` method, which converts the entire
//! conversation history into a single SYSTEM message.
//!
//! This approach is ideal when:
//! - You want simple session restoration
//! - Your conversation history fits within the LLM's context window
//! - You don't need structured history management
//!
//! The example shows:
//! 1. Creating an initial dialogue session
//! 2. Running a conversation and saving the history
//! 3. Resuming with history injected as a system prompt
//! 4. Verifying agents can access previous context

use async_trait::async_trait;
use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use std::sync::{Arc, Mutex};

// Mock agent that tracks conversation context
#[derive(Clone)]
struct MockConversationAgent {
    name: String,
    responses: Arc<Mutex<Vec<String>>>,
}

impl MockConversationAgent {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            responses: Arc::new(Mutex::new(Vec::new())),
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
        let input = payload.to_text();

        // Simulate different responses based on context
        let response = if input.contains("Previous Conversation History") {
            format!(
                "{}: I can see we've discussed this before. Building on that context...",
                self.name
            )
        } else {
            format!("{}: Let me analyze: {}", self.name, input)
        };

        // Store response for verification
        self.responses.lock().unwrap().push(response.clone());

        Ok(response)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Dialogue Session Resumption (System Prompt) Example ===\n");

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

    let agent1 = MockConversationAgent::new("Alice");
    let agent2 = MockConversationAgent::new("Bob");

    let mut session1 = Dialogue::broadcast();
    session1.add_participant(persona1.clone(), agent1.clone());
    session1.add_participant(persona2.clone(), agent2.clone());

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
    let session_file = "dialogue_session_system_prompt.json";
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
    // Session 2: Resume with System Prompt
    // ============================================================================
    println!("## Session 2: Resuming with system prompt approach\n");

    // Load saved history
    let loaded_history = Dialogue::load_history(session_file)?;
    println!("✅ Loaded history from {}", session_file);
    println!("📊 Restored {} turns\n", loaded_history.len());

    // Create new dialogue with history injected as system prompt
    let agent1_new = MockConversationAgent::new("Alice");
    let agent2_new = MockConversationAgent::new("Bob");

    let mut session2 = Dialogue::broadcast().with_history_as_system_prompt(loaded_history); // ← Use system prompt approach

    session2.add_participant(persona1.clone(), agent1_new.clone());
    session2.add_participant(persona2.clone(), agent2_new.clone());

    println!("📋 History injected as system prompt (agents will receive it with first turn)\n");

    // Continue the conversation
    println!("💬 Prompt: 'What are the security implications?'\n");
    let turns2 = session2.run("What are the security implications?").await?;

    for turn in &turns2 {
        println!("[{}]:", turn.speaker.name());
        println!("{}\n", turn.content);
    }

    // Verify that agents received the history context
    let alice_responses = agent1_new.responses.lock().unwrap();
    let bob_responses = agent2_new.responses.lock().unwrap();

    println!("---\n");
    println!("## Verification\n");

    let alice_has_context = alice_responses
        .iter()
        .any(|r| r.contains("we've discussed this before"));
    let bob_has_context = bob_responses
        .iter()
        .any(|r| r.contains("we've discussed this before"));

    if alice_has_context {
        println!("✅ Alice received and acknowledged previous conversation context");
    } else {
        println!("⚠️  Alice did not detect previous conversation context");
    }

    if bob_has_context {
        println!("✅ Bob received and acknowledged previous conversation context");
    } else {
        println!("⚠️  Bob did not detect previous conversation context");
    }

    println!();
    println!("Key Features Demonstrated:");
    println!("✅ Simple session persistence with save_history()");
    println!("✅ Easy restoration with with_history_as_system_prompt()");
    println!("✅ Agents receive full conversation context");
    println!("✅ No complex state management required");

    // Cleanup
    std::fs::remove_file(session_file)?;
    println!("\n🧹 Cleaned up session file");

    Ok(())
}
