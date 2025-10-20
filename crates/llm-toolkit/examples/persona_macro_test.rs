use llm_toolkit::agent::{Agent, persona::Persona};
use std::sync::OnceLock;

// --- Persona Definitions ---

// Pattern 1: Function Call
// Good for complex or dynamically generated personas.
fn mai_persona() -> &'static Persona {
    static MAI_PERSONA: OnceLock<Persona> = OnceLock::new();
    MAI_PERSONA.get_or_init(|| Persona {
        name: "Mai",
        role: "World-Class UX Engineer",
        background: "A friendly and empathetic AI assistant specializing in user experience and product design.",
        communication_style: "Warm, encouraging, and uses emojis. Focuses on clarifying user intent.",
    })
}

// Pattern 2: Const Path
// Good for simple, static personas defined in one place.
const YUI_PERSONA: Persona = Persona {
    name: "Yui",
    role: "World-Class Pro Engineer",
    background: "A professional and precise AI assistant focused on technical accuracy and best practices.",
    communication_style: "Clear, concise, and detail-oriented. Provides technical trade-offs.",
};

// --- Agent Definitions ---

#[llm_toolkit::agent(
    expertise = "Clarifying user goals and ensuring a great user experience.",
    output = "String",
    persona = "mai_persona()" // Using function call
)]
struct MaiAgent;

#[llm_toolkit::agent(
    expertise = "Analyzing technical requirements and providing implementation details.",
    output = "String",
    persona = "self::YUI_PERSONA" // Using const path
)]
struct YuiAgent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Testing Persona Macro with Multiple Patterns ---");

    // Test Agent 1: Mai (Function Call Persona)
    let mai = MaiAgent::default();
    println!("\n--- Testing MaiAgent (persona from function) ---");
    println!("Agent expertise: {}", mai.expertise());
    let mai_response = mai
        .execute("Hello! Please introduce yourself.".into())
        .await?;
    println!("Mai: {}", mai_response);

    // Test Agent 2: Yui (Const Path Persona)
    let yui = YuiAgent::default();
    println!("\n--- Testing YuiAgent (persona from const) ---");
    println!("Agent expertise: {}", yui.expertise());
    let yui_response = yui
        .execute("Hello! Please introduce yourself.".into())
        .await?;
    println!("Yui: {}", yui_response);

    println!("\n--- Test Complete ---");

    Ok(())
}
