//! Example: Persona Visual Identity for Enhanced Recognition
//!
//! This example demonstrates how visual identity (icons and taglines) can enhance
//! agent recognition and improve multi-agent dialogue clarity. Visual anchoring
//! has been shown to strengthen LLM's role adherence and improve conversation quality.
//!
//! Key features demonstrated:
//! - Creating personas with visual identity (icons + taglines)
//! - Visual differentiation in multi-agent dialogues
//! - Enhanced conversation history readability
//! - Stronger agent role adherence through visual anchoring

use async_trait::async_trait;
use llm_toolkit::ToPrompt;
use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::{Persona, VisualIdentity};
use llm_toolkit::agent::{Agent, AgentError, Payload};

// Mock agent for demonstration
#[derive(Clone)]
struct MockAgent {
    name: String,
}

impl MockAgent {
    fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
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
        let input = payload.to_text();

        // Simulate persona-appropriate responses
        let response = match self.name.as_str() {
            "Alice" => format!("🎨 Alice here! From a design perspective: {}", input),
            "Bob" => format!("🔧 Bob checking in. Technical analysis: {}", input),
            "Charlie" => format!("📊 Charlie reporting. Data shows: {}", input),
            "Diana" => format!("🔒 Diana on security. Risk assessment: {}", input),
            _ => format!("{}: {}", self.name, input),
        };

        Ok(response)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Persona Visual Identity Example ===\n");
    println!("Demonstrating enhanced agent recognition through visual anchoring\n");
    println!("---\n");

    // ============================================================================
    // Create Personas with Visual Identity
    // ============================================================================
    println!("## Creating Team with Visual Identities\n");

    // Designer with visual identity
    let alice_identity = VisualIdentity::new("🎨")
        .with_tagline("User-Centered Design Advocate")
        .with_color("#FF6B6B");

    let persona_alice = Persona::new("Alice", "UI/UX Designer")
        .with_background("10 years of experience in user-centered design and interface development")
        .with_communication_style("Visual, empathetic, and user-focused")
        .with_visual_identity(alice_identity);

    println!(
        "✅ Created: {} - {}",
        persona_alice.display_name(),
        persona_alice.tagline().unwrap_or("")
    );

    // Engineer with visual identity (using convenience method)
    let persona_bob = Persona::new("Bob", "Backend Engineer")
        .with_background("Senior engineer specializing in distributed systems and performance")
        .with_communication_style("Technical, pragmatic, and detail-oriented")
        .with_icon("🔧") // Convenience method for quick icon addition
        .with_visual_identity(
            VisualIdentity::new("🔧").with_tagline("Performance & Scalability Specialist"),
        );

    println!(
        "✅ Created: {} - {}",
        persona_bob.display_name(),
        persona_bob.tagline().unwrap_or("")
    );

    // Data Analyst with visual identity
    let persona_charlie = Persona::new("Charlie", "Data Analyst")
        .with_background("Expert in data visualization and statistical analysis")
        .with_communication_style("Data-driven, analytical, and evidence-based")
        .with_visual_identity(
            VisualIdentity::new("📊")
                .with_tagline("Data Tells the Story")
                .with_color("#4ECDC4"),
        );

    println!(
        "✅ Created: {} - {}",
        persona_charlie.display_name(),
        persona_charlie.tagline().unwrap_or("")
    );

    // Security Engineer with visual identity
    let persona_diana = Persona::new("Diana", "Security Engineer")
        .with_background("Specializes in zero-trust architecture and threat modeling")
        .with_communication_style("Security-first, risk-aware, and proactive")
        .with_visual_identity(
            VisualIdentity::new("🔒")
                .with_tagline("Zero-Trust Architecture Champion")
                .with_color("#95E1D3"),
        );

    println!(
        "✅ Created: {} - {}",
        persona_diana.display_name(),
        persona_diana.tagline().unwrap_or("")
    );

    println!("\n---\n");

    // ============================================================================
    // Run Multi-Agent Dialogue with Visual Identity
    // ============================================================================
    println!("## Starting Multi-Agent Discussion\n");
    println!("Topic: 'Designing a new real-time collaboration feature'\n");

    let mut dialogue = Dialogue::broadcast();
    dialogue
        .add_participant(persona_alice.clone(), MockAgent::new("Alice"))
        .add_participant(persona_bob.clone(), MockAgent::new("Bob"))
        .add_participant(persona_charlie.clone(), MockAgent::new("Charlie"))
        .add_participant(persona_diana.clone(), MockAgent::new("Diana"));

    let turns = dialogue
        .run("We need to design a new real-time collaboration feature. What are your perspectives?")
        .await?;

    println!("💬 Discussion Results:\n");
    for turn in &turns {
        println!("[{}]:", turn.speaker.display_name());
        println!("{}\n", turn.content);
    }

    println!("---\n");

    // ============================================================================
    // Save and Display History
    // ============================================================================
    println!("## Conversation History with Visual Identity\n");

    dialogue.save_history("persona_visual_identity_session.json")?;
    println!("✅ Saved conversation history\n");

    println!("Historical records showing visual identity:");
    let history = dialogue.history();
    for (idx, turn) in history.iter().enumerate() {
        let speaker_display = match &turn.speaker {
            llm_toolkit::agent::dialogue::Speaker::System => "[System]".to_string(),
            llm_toolkit::agent::dialogue::Speaker::User { name, .. } => format!("[{}]", name),
            llm_toolkit::agent::dialogue::Speaker::Agent { name, role, icon } => match icon {
                Some(icon) => format!("[{} {} ({})]", icon, name, role),
                None => format!("[{} ({})]", name, role),
            },
        };

        println!("{}. {}", idx + 1, speaker_display);
    }

    println!("\n---\n");

    // ============================================================================
    // Benefits Summary
    // ============================================================================
    println!("## Benefits of Visual Identity\n");
    println!("✅ **Enhanced Recognition**: Icons provide visual anchors for LLMs");
    println!("✅ **Improved Clarity**: Easier to distinguish agents in conversation logs");
    println!("✅ **Stronger Adherence**: LLMs maintain role consistency better");
    println!("✅ **Human Readability**: Users quickly identify agents at a glance");
    println!("✅ **Professional**: Taglines communicate expertise clearly");
    println!("✅ **Future-Ready**: Color codes enable UI integration\n");

    println!("## Persona Details\n");
    println!("🎨 Alice: {}", persona_alice.to_prompt());
    println!("\n🔧 Bob: {}", persona_bob.to_prompt());
    println!("\n📊 Charlie: {}", persona_charlie.to_prompt());
    println!("\n🔒 Diana: {}", persona_diana.to_prompt());

    // Cleanup
    std::fs::remove_file("persona_visual_identity_session.json")?;
    println!("\n🧹 Cleaned up session file");

    Ok(())
}
