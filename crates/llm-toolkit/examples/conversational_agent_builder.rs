use async_trait::async_trait;
use llm_toolkit::agent::chat::Chat;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};

// 1. Define a mock base agent for demonstration.
#[derive(Clone)]
struct MockAgent;

#[async_trait]
impl Agent for MockAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "A mock agent that repeats the input.";
        &EXPERTISE
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        Ok(format!("Mock response to: {}", intent.to_text()))
    }
}

// 2. Define a sample persona.
fn yui_persona() -> Persona {
    Persona {
        name: "Yui".to_string(),
        role: "World-Class Pro Engineer".to_string(),
        background: "A top-tier software engineer focused on quality and precision.".to_string(),
        communication_style: "Professional, precise, and detail-oriented.".to_string(),
        visual_identity: None,
        capabilities: None,
    }
}

// 3. Main function to demonstrate the builder patterns.
#[tokio::main]
async fn main() {
    println!("--- Running Conversational Agent Builder Example ---");

    // Pattern 1: Simple Stateful Chat (HistoryAwareAgent<MockAgent>)
    let chat_session = Chat::new(MockAgent).build();
    println!("\n--- Pattern 1: Simple Stateful Chat ---");
    let res1 = chat_session.execute("First message".into()).await.unwrap();
    println!("Response 1: {}", res1);
    let res2 = chat_session.execute("Second message".into()).await.unwrap();
    println!("Response 2: {}", res2); // This call should have history of the first message.

    // Pattern 2: Stateful Chat with Persona (HistoryAwareAgent<PersonaAgent<MockAgent>>)
    let character_session = Chat::new(MockAgent).with_persona(yui_persona()).build();
    println!("\n--- Pattern 2: Stateful Chat with Persona ---");
    let res3 = character_session
        .execute("First message with persona".into())
        .await
        .unwrap();
    println!("Response 3: {}", res3);
    let res4 = character_session
        .execute("Second message with persona".into())
        .await
        .unwrap();
    println!("Response 4: {}", res4);

    // Pattern 3: Stateless Persona Agent (PersonaAgent<MockAgent>)
    let stateless_persona = Chat::new(MockAgent)
        .with_persona(yui_persona())
        .with_history(false)
        .build();
    println!("\n--- Pattern 3: Stateless Persona Agent ---");
    let res5 = stateless_persona
        .execute("A stateless message".into())
        .await
        .unwrap();
    println!("Response 5: {}", res5);
    let res6 = stateless_persona
        .execute("Another stateless message".into())
        .await
        .unwrap();
    println!("Response 6: {}", res6); // This call should have no memory of the previous one.

    println!("\n--- Example Finished ---");
}
