// The trybuild environment needs explicit imports for all crates used by the macro expansion.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
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