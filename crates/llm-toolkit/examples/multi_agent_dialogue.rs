use async_trait::async_trait;
use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};

// A mock LLM agent for demonstration purposes.
#[derive(Clone)]
struct MockLLMAgent {
    agent_type: String,
}

#[async_trait]
impl Agent for MockLLMAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "A mock LLM agent."
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let last_line = intent.to_text().lines().last().unwrap_or("").to_string();
        let response = format!("[{}] processed: '{}'", self.agent_type, last_line);
        Ok(response)
    }
}

#[tokio::main]
async fn main() {
    println!("--- Running Multi-Agent Dialogue Example ---");

    // --- Pattern 1: Sequential Pipeline ---
    // Use case: Processing a piece of data through a series of steps.
    println!("\n--- Pattern 1: Sequential Pipeline ---");

    // Create personas with String fields
    let summarizer_persona = Persona {
        name: "Summarizer".to_string(),
        role: "Content Summarizer".to_string(),
        background: "Expert in distilling long texts into key points.".to_string(),
        communication_style: "Concise and to the point.".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let translator_persona = Persona {
        name: "Translator".to_string(),
        role: "English to Japanese Translator".to_string(),
        background: "Professional translator with a focus on technical accuracy.".to_string(),
        communication_style: "Formal and precise.".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut sequential_dialogue = Dialogue::sequential();
    sequential_dialogue
        .add_participant(
            summarizer_persona,
            MockLLMAgent {
                agent_type: "Summarizer".to_string(),
            },
        )
        .add_participant(
            translator_persona,
            MockLLMAgent {
                agent_type: "Translator".to_string(),
            },
        );

    let initial_text = "The quick brown fox jumps over the lazy dog.";
    println!("Initial Text: '{}'", initial_text);

    match sequential_dialogue.run(initial_text.to_string()).await {
        Ok(turns) => {
            println!("Sequential Pipeline Results:");
            for turn in &turns {
                println!("  [{}]: {}", turn.speaker.name(), turn.content);
            }
        }
        Err(e) => eprintln!("Sequential dialogue failed: {}", e),
    }

    // --- Pattern 2: Broadcast ---
    // Use case: Getting multiple perspectives on a single topic.
    println!("\n--- Pattern 2: Broadcast ---");

    // Create personas for broadcast mode
    let translator_persona_broadcast = Persona {
        name: "Translator".to_string(),
        role: "English to Japanese Translator".to_string(),
        background: "Professional translator with a focus on technical accuracy.".to_string(),
        communication_style: "Formal and precise.".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let critic_persona = Persona {
        name: "Critic".to_string(),
        role: "Content Critic".to_string(),
        background: "A skeptical critic who finds potential issues.".to_string(),
        communication_style: "Direct and questioning.".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut broadcast_dialogue = Dialogue::broadcast();
    broadcast_dialogue
        .add_participant(
            translator_persona_broadcast,
            MockLLMAgent {
                agent_type: "Translator".to_string(),
            },
        )
        .add_participant(
            critic_persona,
            MockLLMAgent {
                agent_type: "Critic".to_string(),
            },
        );

    let topic = "The new API design is complete.";
    println!("Topic: '{}'", topic);

    match broadcast_dialogue.run(topic.to_string()).await {
        Ok(turns) => {
            println!("Broadcast Responses (from all agents):");
            for turn in &turns {
                println!("  [{}]: {}", turn.speaker.name(), turn.content);
            }
        }
        Err(e) => eprintln!("Broadcast dialogue failed: {}", e),
    }

    println!("\n--- Example Finished ---");
}
