//! Example demonstrating Orchestrator's get_typed_output() functionality.
//!
//! This example shows:
//! - Creating agents with TypeMarker-enabled output types
//! - Using Orchestrator to execute a simple 1-step workflow
//! - Retrieving typed results using get_typed_output()
//! - Verifying __type field is preserved during serialization
//!
//! Run with: cargo run --example orchestrator_get_typed_output --features agent,derive

use async_trait::async_trait;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator, StrategyMap, StrategyStep};
use llm_toolkit::{ToPrompt, TypeMarker, type_marker};
use serde::{Deserialize, Serialize};

// Define output types with TypeMarker
// Using #[type_marker] attribute macro to automatically add __type field
// IMPORTANT: #[type_marker] must be placed FIRST (before #[derive])
// The __type field is:
// - Automatically added by the macro (no manual definition needed)
// - Automatically excluded from LLM prompts (schema generation skips __type)
// - Preserved during JSON serialization (for Orchestrator type-based retrieval)
// - Automatically filled during deserialization via #[serde(default)]
#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct DataAnalysisResult {
    pub summary: String,
    pub insights: Vec<String>,
    pub score: i32,
}

#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct ReportResult {
    pub title: String,
    pub content: String,
    pub conclusion: String,
}

/// Mock agent that returns TypeMarker-enabled structured data
#[derive(Debug)]
struct MockTypedAgent {
    name: String,
    expertise: String,
}

impl MockTypedAgent {
    fn new(name: &str, expertise: &str) -> Self {
        Self {
            name: name.to_string(),
            expertise: expertise.to_string(),
        }
    }
}

#[async_trait]
impl Agent for MockTypedAgent {
    type Output = DataAnalysisResult;

    fn expertise(&self) -> &str {
        &self.expertise
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        println!(
            "🔍 {} executing with intent: {}",
            self.name,
            intent.to_text().chars().take(80).collect::<String>()
        );

        // Return mock structured data
        // The __type field is added automatically by #[type_marker] but must be specified when creating instances
        Ok(DataAnalysisResult {
            __type: DataAnalysisResult::TYPE_NAME.to_string(),
            summary: "Analysis completed successfully".to_string(),
            insights: vec![
                "Key finding 1: Data quality is high".to_string(),
                "Key finding 2: Trends are positive".to_string(),
                "Key finding 3: No anomalies detected".to_string(),
            ],
            score: 95,
        })
    }
}

/// Mock strategy agent that returns a simple 1-step strategy
#[derive(Debug)]
struct MockStrategyAgent;

#[async_trait]
impl Agent for MockStrategyAgent {
    type Output = StrategyMap;

    fn expertise(&self) -> &str {
        "Mock strategy generator"
    }

    fn name(&self) -> String {
        "MockStrategyAgent".to_string()
    }

    async fn execute(&self, _intent: Payload) -> Result<Self::Output, AgentError> {
        let mut strategy = StrategyMap::new("Analyze data and extract insights".to_string());

        strategy.add_step(StrategyStep::new(
            "analysis".to_string(),
            "Analyze the input data".to_string(),
            "DataAnalyzer".to_string(),
            "Analyze this data: {user_request}".to_string(),
            "Analysis result with insights".to_string(),
        ));

        Ok(strategy)
    }
}

/// Mock internal agent for intent generation
#[derive(Debug)]
struct MockInternalAgent;

#[async_trait]
impl Agent for MockInternalAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "Mock internal agent"
    }

    fn name(&self) -> String {
        "MockInternalAgent".to_string()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        // Just pass through the intent
        Ok(intent.to_text())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Orchestrator get_typed_output() Example\n");
    println!("This example demonstrates type-based output retrieval from Orchestrator\n");

    // Create a simple 1-step workflow
    let blueprint = BlueprintWorkflow::new(
        "Analyze customer feedback data and extract key insights".to_string(),
    );

    // Create orchestrator with mock internal agents
    let mut orchestrator = Orchestrator::with_internal_agents(
        blueprint,
        Box::new(MockInternalAgent),
        Box::new(MockStrategyAgent),
    );

    // Add the typed agent
    orchestrator.add_agent(MockTypedAgent::new(
        "DataAnalyzer",
        "Expert at analyzing data and extracting insights",
    ));

    println!("📋 Setup complete:");
    println!(
        "   - Registered {} agents",
        orchestrator.list_agents().len()
    );
    println!("   - Using mock agents (no external dependencies)\n");

    let task = "Analyze Q4 customer feedback data";
    println!("🚀 Executing task: {}\n", task);

    // Execute the workflow
    let result = orchestrator.execute(task).await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            println!("✅ Workflow completed successfully!\n");

            // Display all step outputs in context
            println!("📦 All step outputs in context:");
            let all_outputs = orchestrator.get_all_step_outputs();
            for (step_id, output) in &all_outputs {
                println!(
                    "   - {}: {}",
                    step_id,
                    serde_json::to_string(output)
                        .unwrap_or_default()
                        .chars()
                        .take(100)
                        .collect::<String>()
                );

                // Check if __type field exists
                if let Some(type_value) = output.get("__type") {
                    println!("      ✅ __type field present: {}", type_value);
                } else {
                    println!("      ❌ __type field MISSING!");
                }
            }

            println!("\n🔍 Testing get_typed_output()...\n");

            // Try to retrieve the typed output
            match orchestrator.get_typed_output::<DataAnalysisResult>() {
                Ok(analysis_result) => {
                    println!("✅ Successfully retrieved DataAnalysisResult by type!");
                    println!("\n📊 Retrieved Data:");
                    println!("   __type: {}", analysis_result.__type);
                    println!("   Summary: {}", analysis_result.summary);
                    println!("   Score: {}", analysis_result.score);
                    println!("   Insights:");
                    for (i, insight) in analysis_result.insights.iter().enumerate() {
                        println!("      {}. {}", i + 1, insight);
                    }

                    println!("\n✅ VERIFICATION PASSED:");
                    println!("   - __type field was preserved during serialization");
                    println!("   - Orchestrator successfully stored typed output in context");
                    println!("   - get_typed_output() successfully retrieved by type");
                }
                Err(e) => {
                    eprintln!("❌ FAILED to retrieve DataAnalysisResult: {}", e);
                    eprintln!("\nThis indicates the __type field was not preserved!");
                    std::process::exit(1);
                }
            }

            // Try to retrieve a type that doesn't exist (should fail)
            println!("\n🔍 Testing get_typed_output() with non-existent type...");
            match orchestrator.get_typed_output::<ReportResult>() {
                Ok(_) => {
                    eprintln!("❌ UNEXPECTED: Found ReportResult when it shouldn't exist!");
                    std::process::exit(1);
                }
                Err(e) => {
                    println!("✅ Correctly failed to find ReportResult: {}", e);
                    println!("   This is expected behavior - type doesn't exist in context");
                }
            }

            println!("\n🎉 All tests passed!");
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            if let Some(error) = result.error_message {
                eprintln!("❌ Workflow failed: {}", error);
            }
            std::process::exit(1);
        }
    }

    Ok(())
}
