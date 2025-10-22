use async_trait::async_trait;
use llm_toolkit::agent::chat::Chat;
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

// Define personas for the agents.
const SUMMARIZER_PERSONA: Persona = Persona {
    name: "Summarizer",
    role: "Content Summarizer",
    background: "Expert in distilling long texts into key points.",
    communication_style: "Concise and to the point.",
};

const TRANSLATOR_PERSONA: Persona = Persona {
    name: "Translator",
    role: "English to Japanese Translator",
    background: "Professional translator with a focus on technical accuracy.",
    communication_style: "Formal and precise.",
};

const CRITIC_PERSONA: Persona = Persona {
    name: "Critic",
    role: "Content Critic",
    background: "A skeptical critic who finds potential issues.",
    communication_style: "Direct and questioning.",
};

#[tokio::main]
async fn main() {
    println!("--- Running Multi-Agent Dialogue Example ---");

    // --- Pattern 1: Sequential Pipeline ---
    // Use case: Processing a piece of data through a series of steps.
    println!("\n--- Pattern 1: Sequential Pipeline ---");

    let summarizer = Chat::new(MockLLMAgent {
        agent_type: "Summarizer".to_string(),
    })
    .with_persona(SUMMARIZER_PERSONA)
    .with_history(false)
    .build();

    let translator = Chat::new(MockLLMAgent {
        agent_type: "Translator".to_string(),
    })
    .with_persona(TRANSLATOR_PERSONA)
    .with_history(false)
    .build();

    let mut sequential_dialogue = Dialogue::sequential();
    sequential_dialogue
        .add_participant(summarizer)
        .add_participant(translator);

    let initial_text = "The quick brown fox jumps over the lazy dog.";
    println!("Initial Text: '{}'", initial_text);

    match sequential_dialogue.run(initial_text.to_string()).await {
        Ok(final_result) => {
            println!(
                "Final Result (from last agent in chain): {:?}",
                final_result
            );
        }
        Err(e) => eprintln!("Sequential dialogue failed: {}", e),
    }

    // --- Pattern 2: Broadcast ---
    // Use case: Getting multiple perspectives on a single topic.
    println!("\n--- Pattern 2: Broadcast ---");

    let translator_for_broadcast = Chat::new(MockLLMAgent {
        agent_type: "Translator".to_string(),
    })
    .with_persona(TRANSLATOR_PERSONA)
    .with_history(false)
    .build();

    let critic = Chat::new(MockLLMAgent {
        agent_type: "Critic".to_string(),
    })
    .with_persona(CRITIC_PERSONA)
    .with_history(false)
    .build();

    let mut broadcast_dialogue = Dialogue::broadcast();
    broadcast_dialogue
        .add_participant(translator_for_broadcast)
        .add_participant(critic);

    let topic = "The new API design is complete.";
    println!("Topic: '{}'", topic);

    match broadcast_dialogue.run(topic.to_string()).await {
        Ok(responses) => {
            println!("Broadcast Responses (from all agents): {:?}", responses);
        }
        Err(e) => eprintln!("Broadcast dialogue failed: {}", e),
    }

    println!("\n--- Example Finished ---");
}
