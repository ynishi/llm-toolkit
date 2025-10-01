//! Orchestrator example with mock agents for testing without external dependencies.
//!
//! This example demonstrates:
//! - Creating custom mock agents for testing
//! - Injecting mock internal agents for strategy generation
//! - How the orchestrator coordinates multiple agents without requiring claude CLI
//!
//! Run with: cargo run --example orchestrator_with_mock --features agent,derive

use async_trait::async_trait;
use llm_toolkit::agent::{Agent, AgentError};
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

    async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
        // Return a mock response
        Ok(format!(
            "{}: Processed request - {}",
            self.response_prefix,
            intent.chars().take(50).collect::<String>()
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

    async fn execute(&self, _intent: String) -> Result<Self::Output, AgentError> {
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
    orchestrator.add_agent(Box::new(MockAgent::new(
        "DataProcessor",
        "Expert at processing and transforming data",
        "📥 Processed",
    )));

    orchestrator.add_agent(Box::new(MockAgent::new(
        "ReportGenerator",
        "Generates clear and comprehensive reports from analyzed data",
        "📝 Generated",
    )));

    println!("📋 Registered {} agents", orchestrator.list_agents().len());
    println!("🔧 Using mock internal agents (no claude CLI required)\n");

    let task = "Process customer feedback data and generate insights report";

    println!("🚀 Executing workflow: {}\n", task);

    let result = orchestrator.execute(task).await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            println!("\n✅ Workflow completed!\n");
            if let Some(output) = result.final_output {
                println!("📄 Final Result:\n{}\n", output);
            }
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            if let Some(error) = result.error_message {
                eprintln!("\n❌ Workflow failed: {}\n", error);
            } else {
                eprintln!("\n❌ Workflow failed\n");
            }
            std::process::exit(1);
        }
    }

    Ok(())
}
