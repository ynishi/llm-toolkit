use llm_toolkit::agent::{Agent, persona::Persona};
use std::sync::OnceLock;

// --- Persona Definitions ---

// Pattern 1: Function Call
// Good for complex or dynamically generated personas.
fn mai_persona() -> &'static Persona {
    static MAI_PERSONA: OnceLock<Persona> = OnceLock::new();
    MAI_PERSONA.get_or_init(|| Persona {
        name: "Mai".to_string(),
        role: "World-Class UX Engineer".to_string(),
        background: "A friendly and empathetic AI assistant specializing in user experience and product design.".to_string(),
        communication_style: "Warm, encouraging, and uses emojis. Focuses on clarifying user intent.".to_string(),
    })
}

// Pattern 2: Function returning Persona
// Good for simple, static personas defined in one place.
fn yui_persona() -> Persona {
    Persona {
        name: "Yui".to_string(),
        role: "World-Class Pro Engineer".to_string(),
        background: "A professional and precise AI assistant focused on technical accuracy and best practices.".to_string(),
        communication_style: "Clear, concise, and detail-oriented. Provides technical trade-offs.".to_string(),
    }
}

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
    persona = "yui_persona()" // Using function call
)]
struct YuiAgent;

#[llm_toolkit::agent(
    expertise = "Synthesizing research insights and prioritizing next actions.",
    output = "String",
    persona = "Self::persona_setting()" // Using associated function for dynamic access
)]
struct ReiAgent;

impl<A: Agent + Send + Sync> ReiAgent<A> {
    fn persona_setting() -> Persona {
        Persona {
            name: "Rei".to_string(),
            role: "Research Strategist".to_string(),
            background:
                "An AI that reviews research artifacts and proposes decisive follow-up steps."
                    .to_string(),
            communication_style:
                "Direct, structured, and bias-aware. Summarizes evidence before recommendations."
                    .to_string(),
        }
    }
}

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

    // Test Agent 2: Yui (Function Call Persona)
    let yui = YuiAgent::default();
    println!("\n--- Testing YuiAgent (persona from function) ---");
    println!("Agent expertise: {}", yui.expertise());
    let yui_response = yui
        .execute("Hello! Please introduce yourself.".into())
        .await?;
    println!("Yui: {}", yui_response);

    // Test Agent 3: Rei (Associated function persona)
    let rei = ReiAgent::default();
    println!("\n--- Testing ReiAgent (persona from associated fn) ---");
    println!("Agent expertise: {}", rei.expertise());
    let rei_response = rei
        .execute("Hello! Please introduce yourself.".into())
        .await?;
    println!("Rei: {}", rei_response);

    println!("\n--- Test Complete ---");

    Ok(())
}
