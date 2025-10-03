// Test for agent attribute macro with custom default_inner type

use llm_toolkit::agent::{Agent, AgentError};
use serde::{Deserialize, Serialize};

// Define a custom OlamaAgent (mock implementation for testing)
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

    async fn execute(&self, intent: String) -> Result<String, AgentError> {
        // Mock implementation
        Ok(format!(
            "{{\"title\": \"Response from {}\", \"body\": \"Processed: {}\"}}",
            self.model, intent
        ))
    }
}

// Define output type
#[derive(Serialize, Deserialize, Debug)]
struct ArticleData {
    title: String,
    body: String,
}

// Using custom default_inner
#[llm_toolkit_macros::agent(
    expertise = "Writing articles with OlamaAgent backend",
    output = "ArticleData",
    default_inner = "OlamaAgent"
)]
struct ArticleWriterAgent;

// Another agent with the same OlamaAgent backend but different expertise
#[llm_toolkit_macros::agent(
    expertise = "Reviewing code with OlamaAgent backend",
    output = "ArticleData",
    default_inner = "OlamaAgent"
)]
struct CodeReviewerAgent;

fn main() {
    // This test just ensures the macro expands correctly at compile time

    // Test 1: Default construction (uses OlamaAgent::default())
    let _writer = ArticleWriterAgent::default();

    // Test 2: Custom injection
    let custom_olama = OlamaAgent::new().with_model("llama3.1");
    let _writer = ArticleWriterAgent::new(custom_olama.clone());

    // Test 3: Multiple agents sharing the same backend
    let _reviewer = CodeReviewerAgent::new(custom_olama);

    // Test 4: Each agent has different expertise
    let writer = ArticleWriterAgent::default();
    let reviewer = CodeReviewerAgent::default();
    assert_eq!(writer.expertise(), "Writing articles with OlamaAgent backend");
    assert_eq!(reviewer.expertise(), "Reviewing code with OlamaAgent backend");
}
