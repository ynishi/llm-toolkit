//! Orchestrator example with mock agents for testing without external dependencies.
//!
//! This example demonstrates:
//! - Creating custom mock agents for testing
//! - Injecting mock internal agents for strategy generation
//! - How the orchestrator coordinates multiple agents without requiring claude CLI
//!
//! Run with: cargo run --example orchestrator_with_mock --features agent,derive

use async_trait::async_trait;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator, StrategyMap, StrategyStep};

/// A mock agent that simulates work without requiring external LLM calls.
#[derive(Debug)]
struct MockAgent {
    name: String,
    expertise: String,
    response_prefix: String,
}

impl MockAgent {
    fn new(name: &str, expertise: &str, response_prefix: &str) -> Self {
        Self {
            name: name.to_string(),
            expertise: expertise.to_string(),
            response_prefix: response_prefix.to_string(),
        }
    }
}

#[async_trait]
impl Agent for MockAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        &self.expertise
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        // Return a mock response
        let text_intent = intent.to_text();
        Ok(format!(
            "{}: Processed request - {}",
            self.response_prefix,
            text_intent.chars().take(50).collect::<String>()
        ))
    }
}

/// A mock JSON agent that returns a simple StrategyMap for testing.
#[derive(Debug)]
struct MockJsonAgent;

#[async_trait]
impl Agent for MockJsonAgent {
    type Output = StrategyMap;

    fn expertise(&self) -> &str {
        "Mock strategy generator for testing"
    }

    fn name(&self) -> String {
        "MockJsonAgent".to_string()
    }

    async fn execute(&self, _intent: Payload) -> Result<Self::Output, AgentError> {
        // Return a simple mock strategy
        let mut strategy = StrategyMap::new("Complete the mock task".to_string());

        strategy.add_step(StrategyStep::new(
            "step_1".to_string(),
            "Process data".to_string(),
            "DataProcessor".to_string(),
            "Process the input data: {user_request}".to_string(),
            "Processed data result".to_string(),
        ));

        strategy.add_step(StrategyStep::new(
            "step_2".to_string(),
            "Generate report".to_string(),
            "ReportGenerator".to_string(),
            "Generate report from: {previous_output}".to_string(),
            "Final report".to_string(),
        ));

        Ok(strategy)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Orchestrator Mock Agent Example\n");
    println!("✨ This example runs completely offline without requiring claude CLI\n");

    // Define a simple workflow
    let blueprint = BlueprintWorkflow::new(
        r#"
        Data Processing Pipeline:
        1. Extract data from source
        2. Transform and clean the data
        3. Analyze patterns and insights
        4. Generate summary report
        "#
        .to_string(),
    );

    // Create orchestrator with mock internal agents
    let mut orchestrator = Orchestrator::with_internal_agents(
        blueprint,
        Box::new(MockAgent::new(
            "MockInternalAgent",
            "Mock internal agent for intent generation and redesign",
            "🤖 Internal",
        )),
        Box::new(MockJsonAgent),
    );

    // Add mock agents with different specializations
    // Generic add_agent() automatically wraps them
    orchestrator.add_agent(MockAgent::new(
        "DataProcessor",
        "Expert at processing and transforming data",
        "📥 Processed",
    ));

    orchestrator.add_agent(MockAgent::new(
        "ReportGenerator",
        "Generates clear and comprehensive reports from analyzed data",
        "📝 Generated",
    ));

    println!("📋 Registered {} agents", orchestrator.list_agents().len());
    println!("🔧 Using mock internal agents (no claude CLI required)\n");

    let task = "Process customer feedback data and generate insights report";

    println!("🚀 Executing workflow: {}\n", task);

    let result = orchestrator.execute(task).await;

    // E2E Test Assertions
    println!("🔍 Validating results...\n");

    // Assert: Status should be Success
    if result.status != llm_toolkit::orchestrator::OrchestrationStatus::Success {
        eprintln!("❌ ASSERTION FAILED: Expected Success status");
        if let Some(error) = result.error_message {
            eprintln!("   Error: {}", error);
        }
        std::process::exit(1);
    }

    // Assert: Should have executed 2 steps
    if result.steps_executed != 2 {
        eprintln!(
            "❌ ASSERTION FAILED: Expected 2 steps executed, got {}",
            result.steps_executed
        );
        std::process::exit(1);
    }

    // Assert: Should have 0 redesigns (no errors)
    if result.redesigns_triggered != 0 {
        eprintln!(
            "❌ ASSERTION FAILED: Expected 0 redesigns, got {}",
            result.redesigns_triggered
        );
        std::process::exit(1);
    }

    // Assert: Final output should exist
    let output = result
        .final_output
        .expect("❌ ASSERTION FAILED: Expected final output");

    // Assert: Output should be a string containing the mock response prefix
    let output_str = serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string());
    if !output_str.contains("📝 Generated") {
        eprintln!(
            "❌ ASSERTION FAILED: Expected output to contain '📝 Generated'\n   Got: {}",
            output_str
        );
        std::process::exit(1);
    }

    // All assertions passed!
    println!("✅ All assertions passed!\n");
    println!("📊 Execution Summary:");
    println!("  Status: {:?}", result.status);
    println!("  Steps executed: {}", result.steps_executed);
    println!("  Redesigns triggered: {}", result.redesigns_triggered);
    println!("\n📄 Final Result:\n{}\n", output_str);

    println!("✨ E2E Test: PASSED");

    Ok(())
}
