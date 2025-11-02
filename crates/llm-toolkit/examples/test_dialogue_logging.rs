use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

#[derive(Clone)]
struct SimpleAgent {
    name: String,
}

#[async_trait]
impl Agent for SimpleAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "Simple agent"
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
        Ok(format!("{} responded", self.name))
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(true)
        .init();

    let mut dialogue = Dialogue::broadcast();

    let persona_a = Persona {
        name: "AgentA".to_string(),
        role: "TesterA".to_string(),
        background: "Test agent A".to_string(),
        communication_style: "Direct".to_string(),
    };

    let persona_b = Persona {
        name: "AgentB".to_string(),
        role: "TesterB".to_string(),
        background: "Test agent B".to_string(),
        communication_style: "Direct".to_string(),
    };

    dialogue
        .add_participant(persona_a, SimpleAgent { name: "AgentA".to_string() })
        .add_participant(persona_b, SimpleAgent { name: "AgentB".to_string() });

    println!("\n=== Turn 1 ===");
    dialogue.run("First message").await.unwrap();

    println!("\n=== Turn 2 ===");
    dialogue.run("Second message").await.unwrap();
}
