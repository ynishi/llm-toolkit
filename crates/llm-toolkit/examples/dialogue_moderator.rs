//! Example: Using Moderator for dynamic execution strategy selection
//!
//! This example demonstrates the Moderator execution model, which allows
//! a moderator agent to dynamically decide how the dialogue should execute
//! based on the current context and conversation state.
//!
//! The moderator can choose:
//! - OrderedSequential: Execute participants in a specific order
//! - OrderedBroadcast: Get input from all participants in parallel
//! - Mentioned: Let the user @mention who should speak
//!
//! This is useful for:
//! - Adaptive workflows that change based on conversation flow
//! - Complex decision-making scenarios
//! - Delegating control to an intelligent coordinator

use llm_toolkit::agent::dialogue::{
    BroadcastOrder, Dialogue, ExecutionModel, SequentialOrder,
};
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

/// A simple moderator that decides execution strategy based on keywords.
///
/// This demonstrates how a moderator can analyze the input and choose
/// the appropriate execution model dynamically.
#[derive(Clone)]
struct KeywordModerator;

#[async_trait]
impl Agent for KeywordModerator {
    type Output = ExecutionModel;

    fn expertise(&self) -> &str {
        "Decides execution strategy based on conversation context"
    }

    fn name(&self) -> String {
        "KeywordModerator".to_string()
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        // Simple keyword-based decision logic
        let text = payload
            .to_messages()
            .into_iter()
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join(" ");

        if text.contains("brainstorm") || text.contains("all ideas") {
            // For brainstorming, everyone speaks in parallel
            println!("[Moderator]: Detected brainstorming → Broadcast mode");
            Ok(ExecutionModel::OrderedBroadcast(
                BroadcastOrder::Completion,
            ))
        } else if text.contains("analyze") || text.contains("step by step") {
            // For analysis, sequential processing with Analyst first
            println!("[Moderator]: Detected analysis task → Sequential mode (Analyst → Engineer → Designer)");
            Ok(ExecutionModel::OrderedSequential(
                SequentialOrder::Explicit(vec![
                    "Analyst".to_string(),
                    "Engineer".to_string(),
                    "Designer".to_string(),
                ]),
            ))
        } else if text.contains("design") {
            // For design tasks, Designer goes first
            println!("[Moderator]: Detected design task → Sequential mode (Designer → Engineer → Analyst)");
            Ok(ExecutionModel::OrderedSequential(
                SequentialOrder::Explicit(vec![
                    "Designer".to_string(),
                    "Engineer".to_string(),
                    "Analyst".to_string(),
                ]),
            ))
        } else {
            // Default: broadcast
            println!("[Moderator]: Default → Broadcast mode");
            Ok(ExecutionModel::OrderedBroadcast(
                BroadcastOrder::Completion,
            ))
        }
    }
}

/// Mock agent that responds with a simple message
#[derive(Clone)]
struct MockAgent {
    name: String,
    role: String,
}

impl MockAgent {
    fn new(name: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            role: role.into(),
        }
    }
}

#[async_trait]
impl Agent for MockAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "Mock agent for demonstration"
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let input = payload
            .to_messages()
            .into_iter()
            .filter(|m| !m.content.contains("Conversation history"))
            .filter(|m| !m.content.contains("Available participants"))
            .filter(|m| !m.content.contains("YOU ARE A PERSONA"))
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join(" ");

        Ok(format!(
            "[{}] handling: {}",
            self.role,
            input.chars().take(50).collect::<String>()
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Moderator Execution Model Example ===\n");

    // Create personas for our team
    let analyst = Persona {
        name: "Analyst".to_string(),
        role: "Data Analyst".to_string(),
        background: "Expert in data analysis and statistical reasoning".to_string(),
        communication_style: "Analytical and data-driven".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let engineer = Persona {
        name: "Engineer".to_string(),
        role: "Software Engineer".to_string(),
        background: "Full-stack developer with system design expertise".to_string(),
        communication_style: "Technical and pragmatic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let designer = Persona {
        name: "Designer".to_string(),
        role: "UX Designer".to_string(),
        background: "User experience specialist focused on usability".to_string(),
        communication_style: "User-centric and visual".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    // Create mock agents for each persona
    let analyst_agent = MockAgent::new("Analyst", "Data Analyst");
    let engineer_agent = MockAgent::new("Engineer", "Software Engineer");
    let designer_agent = MockAgent::new("Designer", "UX Designer");

    // Create a dialogue with Moderator execution model
    let mut dialogue = Dialogue::moderator();

    // Set the moderator agent
    dialogue.with_moderator(KeywordModerator);

    // Add participants
    dialogue.add_participant(analyst, analyst_agent);
    dialogue.add_participant(engineer, engineer_agent);
    dialogue.add_participant(designer, designer_agent);

    println!("Created dialogue with {} participants", dialogue.participant_count());
    println!("Execution model: Moderator (dynamic strategy selection)\n");

    // ========================================================================
    // Scenario 1: Brainstorming task
    // ========================================================================
    println!("--- Scenario 1: Brainstorming ---");
    let turns1 = dialogue
        .run("Let's brainstorm all ideas for improving user onboarding")
        .await?;

    println!("\nResults: {} participants responded", turns1.len());
    for turn in &turns1 {
        println!("  [{}]: {}", turn.speaker.name(), turn.content);
    }

    // ========================================================================
    // Scenario 2: Analysis task
    // ========================================================================
    println!("\n--- Scenario 2: Analysis Task ---");
    let turns2 = dialogue
        .run("Analyze the performance metrics step by step")
        .await?;

    println!("\nResults: {} turn(s) (Sequential mode)", turns2.len());
    for turn in &turns2 {
        println!("  [{}]: {}", turn.speaker.name(), turn.content);
    }

    // ========================================================================
    // Scenario 3: Design task
    // ========================================================================
    println!("\n--- Scenario 3: Design Task ---");
    let turns3 = dialogue
        .run("Design a better user interface for the dashboard")
        .await?;

    println!("\nResults: {} turn(s) (Sequential mode)", turns3.len());
    for turn in &turns3 {
        println!("  [{}]: {}", turn.speaker.name(), turn.content);
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n=== Summary ===");
    println!("The Moderator execution model enables:");
    println!("✓ Dynamic strategy selection based on context");
    println!("✓ Adaptive workflow that changes per turn");
    println!("✓ Intelligent coordination without manual configuration");
    println!("\nIn this example:");
    println!("- Scenario 1 (brainstorm) → Broadcast (all participants)");
    println!("- Scenario 2 (analyze) → Sequential (Analyst → Engineer → Designer)");
    println!("- Scenario 3 (design) → Sequential (Designer → Engineer → Analyst)");

    Ok(())
}
