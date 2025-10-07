// Test that Agent derive macro works with backend attribute
extern crate log;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, llm_toolkit::ToPrompt)]
#[prompt(mode = "full")]
pub struct TestOutput {
    pub result: String,
}

// Test default backend (claude)
#[derive(llm_toolkit_macros::Agent)]
#[agent(
    expertise = "Test agent with default backend",
    output = "TestOutput"
)]
pub struct DefaultBackendAgent;

// Test gemini backend
#[derive(llm_toolkit_macros::Agent)]
#[agent(
    expertise = "Test agent with gemini backend",
    output = "TestOutput",
    backend = "gemini"
)]
pub struct GeminiBackendAgent;

// Test gemini backend with model
#[derive(llm_toolkit_macros::Agent)]
#[agent(
    expertise = "Test agent with gemini backend and model",
    output = "TestOutput",
    backend = "gemini",
    model = "pro"
)]
pub struct GeminiWithModelAgent;

#[tokio::main]
async fn main() {}
