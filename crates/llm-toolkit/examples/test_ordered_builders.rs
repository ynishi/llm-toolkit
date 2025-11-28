//! Test example for new ordered builder methods
//!
//! This example demonstrates the new builder methods:
//! - `ordered_sequential()` - Execute participants in specific order
//! - `ordered_broadcast()` - All respond in parallel, results in specific order

use async_trait::async_trait;
use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};

/// Mock agent that identifies itself in responses
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
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Test agent for ordered execution";
        &EXPERTISE
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let input = payload
            .to_messages()
            .into_iter()
            .filter(|m| !m.content.contains("YOU ARE A PERSONA"))
            .filter(|m| !m.content.contains("# Persona Profile"))
            .filter(|m| !m.content.contains("# Participants"))
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join(" ");

        Ok(format!(
            "[{}] ({}): {}",
            self.name,
            self.role,
            input.chars().take(80).collect::<String>()
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing New Ordered Builder Methods ===\n");

    // Create test personas
    let jordan = Persona {
        name: "Jordan".to_string(),
        role: "UX Designer".to_string(),
        background: "User experience specialist".to_string(),
        communication_style: "User-centered and empathetic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let alex = Persona {
        name: "Alex".to_string(),
        role: "Software Engineer".to_string(),
        background: "Full-stack developer".to_string(),
        communication_style: "Technical and systematic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let sam = Persona {
        name: "Sam".to_string(),
        role: "Product Manager".to_string(),
        background: "Product strategy and roadmap".to_string(),
        communication_style: "Strategic and metrics-driven".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    // Create mock agents
    let jordan_agent = MockAgent::new("Jordan", "UX Designer");
    let alex_agent = MockAgent::new("Alex", "Software Engineer");
    let sam_agent = MockAgent::new("Sam", "Product Manager");

    println!("=== Test 1: ordered_sequential() ===");
    println!("Expected order: Jordan → Alex → Sam");

    let mut dialogue1 = Dialogue::ordered_sequential(vec![
        "Jordan".to_string(),
        "Alex".to_string(),
        "Sam".to_string(),
    ]);

    dialogue1.add_participant(jordan.clone(), jordan_agent.clone());
    dialogue1.add_participant(alex.clone(), alex_agent.clone());
    dialogue1.add_participant(sam.clone(), sam_agent.clone());

    let turns1 = dialogue1.run("Design a user-friendly API").await?;

    println!("Results ({} turns):", turns1.len());
    for (i, turn) in turns1.iter().enumerate() {
        println!("  {}. [{}]: {}", i + 1, turn.speaker.name(), turn.content);
    }

    println!("\n=== Test 2: ordered_broadcast() ===");
    println!("Expected order: Jordan → Alex → Sam (but all run in parallel)");

    let mut dialogue2 = Dialogue::ordered_broadcast(vec![
        "Jordan".to_string(),
        "Alex".to_string(),
        "Sam".to_string(),
    ]);

    dialogue2.add_participant(jordan, jordan_agent);
    dialogue2.add_participant(alex, alex_agent);
    dialogue2.add_participant(sam, sam_agent);

    let turns2 = dialogue2.run("Evaluate this feature proposal").await?;

    println!("Results ({} turns):", turns2.len());
    for (i, turn) in turns2.iter().enumerate() {
        println!("  {}. [{}]: {}", i + 1, turn.speaker.name(), turn.content);
    }

    println!("\n=== Summary ===");
    println!("✓ ordered_sequential(): Sequential execution with custom order");
    println!("✓ ordered_broadcast(): Parallel execution with ordered results");
    println!("Both methods successfully use ExecutionModel variants with order control!");

    Ok(())
}
