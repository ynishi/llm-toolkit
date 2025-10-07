// The trybuild environment needs explicit imports for all crates used by the macro expansion.
extern crate log;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, llm_toolkit::ToPrompt)]
#[prompt(mode = "full")]
pub struct ArticleDraft {
    pub title: String,
    pub body: String,
}

#[derive(llm_toolkit_macros::Agent)]
#[agent(
    expertise = "Synthesizing comprehensive articles from source material",
    output = "ArticleDraft"
)]
pub struct ContentSynthesizerAgent;

// We need a dummy main function with tokio to make the async trait implementation compile.
#[tokio::main]
async fn main() {}