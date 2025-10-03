// Test for the new attribute macro #[agent(...)]
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ArticleData {
    pub title: String,
    pub content: String,
}

// Using attribute macro - this will generate struct definition
#[llm_toolkit_macros::agent(
    expertise = "Writing articles with structured data",
    output = "ArticleData"
)]
struct WriterAgent;

#[tokio::main]
async fn main() {}
