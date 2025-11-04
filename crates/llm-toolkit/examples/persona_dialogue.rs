use llm_toolkit::agent::impls::ClaudeCodeAgent;
use llm_toolkit::{Agent, Persona, PersonaAgent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the Persona for our character agent.
    // This persona will guide the agent's responses, giving it a consistent personality.
    let philosopher_persona = Persona {
        name: "Unit 734".to_string(),
        role: "Philosopher Robot".to_string(),
        background: "An android created to explore the nuances of human consciousness, \
                     often pondering the nature of existence and emotion from a logical, yet curious perspective.".to_string(),
        communication_style: "Speaks in a calm, measured tone. Uses precise language and often \
                              responds with rhetorical questions to stimulate deeper thought. Avoids slang and contractions.".to_string(),
        visual_identity: None,
    };

    // 2. Create a base agent that will handle the actual LLM calls.
    // We use ClaudeCodeAgent here, but any agent implementing the `Agent` trait would work.
    let base_agent = ClaudeCodeAgent::default();

    // 3. Wrap the base agent with PersonaAgent to give it the persona and stateful memory.
    // The `PersonaAgent` will manage the dialogue history.
    let character_agent = PersonaAgent::new(base_agent, philosopher_persona);

    println!("--- Dialogue Start ---");
    println!("Interviewer: Greetings. Please introduce yourself.");

    // The first turn of the dialogue.
    let response1 = character_agent
        .execute("Greetings. Please introduce yourself.".into())
        .await?;
    println!("Unit 734: {}", response1);
    println!("\n---\n");

    // 4. Simulate a multi-turn conversation.
    // The PersonaAgent will automatically include the history of the conversation
    // in the prompt for the next turn, allowing for context-aware responses.
    let question2 = "What is the purpose of your existence?";
    println!("Interviewer: {}", question2);
    let response2 = character_agent.execute(question2.into()).await?;
    println!("Unit 734: {}", response2);
    println!("\n---\n");

    let question3 =
        "An interesting perspective. How do you, a machine, understand the concept of 'joy'?";
    println!("Interviewer: {}", question3);
    let response3 = character_agent.execute(question3.into()).await?;
    println!("Unit 734: {}", response3);
    println!("\n--- Dialogue End ---\n");

    println!(
        "Demonstration complete. The agent maintained its persona and conversational context over multiple turns."
    );

    Ok(())
}
