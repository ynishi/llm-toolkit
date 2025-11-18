// Test for #[agent] macro with custom default_inner + persona support
// The trybuild environment needs explicit imports for all crates used by the macro expansion.
extern crate tracing;

use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::agent::persona::Persona;
use serde::{Deserialize, Serialize};

// ========================================================================
// Persona definition
// ========================================================================

fn test_persona() -> Persona {
    Persona::new("Test Persona", "Mock role")
        .with_background("Used to verify persona + default_inner handling")
        .with_communication_style("Precise and concise")
}

// ========================================================================
// Custom backend (OlamaAgent)
// ========================================================================

#[derive(Default, Clone)]
struct OlamaAgent {
    model: String,
}

impl OlamaAgent {
    fn new() -> Self {
        Self {
            model: "llama3".to_string(),
        }
    }

    fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[async_trait::async_trait]
impl Agent for OlamaAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "General purpose Olama agent"
    }

    async fn execute(&self, intent: Payload) -> Result<String, AgentError> {
        let text_intent = intent.to_text();
        Ok(format!(
            "{{\"title\": \"{}\", \"body\": \"{}\"}}",
            self.model, text_intent
        ))
    }
}

// ========================================================================
// Output type
// ========================================================================

#[derive(Serialize, Deserialize, Debug, llm_toolkit_macros::ToPrompt)]
struct ArticleData {
    title: String,
    body: String,
}

// ========================================================================
// Agents
// ========================================================================

#[llm_toolkit_macros::agent(
    expertise = "Writing with persona and custom backend",
    output = "ArticleData",
    default_inner = "OlamaAgent",
    persona = "test_persona()"
)]
struct PersonaArticleWriterAgent;

#[llm_toolkit_macros::agent(
    expertise = "Baseline custom backend agent without persona",
    output = "ArticleData",
    default_inner = "OlamaAgent"
)]
struct PlainArticleWriterAgent;

fn main() {
    // Persona + default_inner should work via Default::default
    let _persona_default = PersonaArticleWriterAgent::default();

    // Custom inner injection should continue to work
    let custom = OlamaAgent::new().with_model("llama3.1");
    let _persona_custom = PersonaArticleWriterAgent::new(custom.clone());

    // Verify the persona-less variant still behaves like before
    let _plain_default = PlainArticleWriterAgent::default();
    let _plain_custom = PlainArticleWriterAgent::new(custom);
}
