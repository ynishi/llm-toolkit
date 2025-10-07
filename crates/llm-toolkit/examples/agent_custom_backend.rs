//! Example demonstrating custom backend agent with default_inner attribute.
//!
//! This example shows how to use a custom Agent implementation (like OlamaAgent)
//! as the default backend for multiple specialized agents.
//!
//! Run with: cargo run --example agent_custom_backend --features agent

use llm_toolkit::ToPrompt;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use serde::{Deserialize, Serialize};

// ============================================================================
// Custom Backend: OlamaAgent
// ============================================================================

/// A custom agent implementation for Olama (example/mock)
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
        // In a real implementation, this would call the Olama API
        // For this example, we'll return mock data
        let text_intent = intent.to_text();
        println!(
            "📡 OlamaAgent (model: {}) executing: {}",
            self.model, text_intent
        );
        Ok(format!(
            r#"{{"title": "Response from {}", "content": "Processed: {}"}}"#,
            self.model, text_intent
        ))
    }
}

// ============================================================================
// Output Types
// ============================================================================

#[derive(Serialize, Deserialize, Debug, ToPrompt)]
struct ArticleDraft {
    title: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug, ToPrompt)]
struct CodeReview {
    title: String,
    content: String,
}

// ============================================================================
// Specialized Agents using OlamaAgent as default backend
// ============================================================================

/// Article writing agent backed by OlamaAgent
#[llm_toolkit_macros::agent(
    expertise = "Writing technical articles with clear explanations and examples",
    output = "ArticleDraft",
    default_inner = "OlamaAgent"
)]
struct ArticleWriterAgent;

/// Code review agent backed by OlamaAgent
#[llm_toolkit_macros::agent(
    expertise = "Reviewing Rust code for best practices, performance, and correctness",
    output = "CodeReview",
    default_inner = "OlamaAgent"
)]
struct CodeReviewerAgent;

/// Data analysis agent backed by OlamaAgent
#[llm_toolkit_macros::agent(
    expertise = "Analyzing data and generating insights with statistical methods",
    output = "ArticleDraft",
    default_inner = "OlamaAgent"
)]
struct DataAnalystAgent;

// ============================================================================
// Main
// ============================================================================

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🎯 Custom Backend Agent Example\n");
    println!("This example demonstrates using a custom agent backend (OlamaAgent)");
    println!("with multiple specialized agents, each with different expertise.\n");

    // ========================================================================
    // Method 1: Using Default
    // ========================================================================
    println!("📌 Method 1: Using Default");
    println!("   Each agent uses OlamaAgent::default() as backend\n");

    let writer = ArticleWriterAgent::default();
    let reviewer = CodeReviewerAgent::default();

    println!("   Writer expertise: {}", writer.expertise());
    println!("   Reviewer expertise: {}\n", reviewer.expertise());

    // ========================================================================
    // Method 2: Inject shared custom agent
    // ========================================================================
    println!("📌 Method 2: Inject shared custom OlamaAgent");
    println!("   Multiple agents can share the same backend instance\n");

    // Create a custom configured Olama agent
    let olama = OlamaAgent::new().with_model("llama3.1");

    // Inject it into multiple specialized agents
    let writer = ArticleWriterAgent::new(olama.clone());
    let reviewer = CodeReviewerAgent::new(olama.clone());
    let analyst = DataAnalystAgent::new(olama);

    println!("   ✅ Created 3 specialized agents sharing the same OlamaAgent backend");
    println!("   Writer expertise: {}", writer.expertise());
    println!("   Reviewer expertise: {}", reviewer.expertise());
    println!("   Analyst expertise: {}\n", analyst.expertise());

    // ========================================================================
    // Method 3: Different configurations for different agents
    // ========================================================================
    println!("📌 Method 3: Different OlamaAgent configurations");
    println!("   Each agent can use a different model/configuration\n");

    let writer = ArticleWriterAgent::new(OlamaAgent::new().with_model("llama3"));
    let reviewer = CodeReviewerAgent::new(OlamaAgent::new().with_model("codellama"));
    let analyst = DataAnalystAgent::new(OlamaAgent::new().with_model("llama3.1"));

    println!("   ✅ Each agent configured with different models");

    // Execute example tasks
    println!("\n📝 Executing sample tasks:\n");

    let article = writer
        .execute("Write about Rust async/await".to_string().into())
        .await
        .unwrap();
    println!("   Article: {:?}", article);

    let review = reviewer
        .execute(
            "Review this function for performance issues"
                .to_string()
                .into(),
        )
        .await
        .unwrap();
    println!("   Review: {:?}", review);

    let analysis = analyst
        .execute("Analyze user engagement metrics".to_string().into())
        .await
        .unwrap();
    println!("   Analysis: {:?}\n", analysis);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("🎉 Key Benefits:");
    println!("   ✅ One OlamaAgent implementation, multiple specialized agents");
    println!("   ✅ Each agent has unique expertise");
    println!("   ✅ Flexible: share backend or use different configurations");
    println!("   ✅ Type-safe: compile-time verification");
    println!("   ✅ Testable: inject mock agents for testing\n");
}
