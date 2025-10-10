//! Production-ready example demonstrating Orchestrator with TypeMarker and real Claude LLM.
//!
//! This example shows:
//! - Using #[derive(TypeMarker)] for type-based retrieval
//! - Real Claude LLM calls via ClaudeCodeAgent
//! - Orchestrator executing a 1-step workflow
//! - Type-based output retrieval with get_typed_output()
//! - Complete production-ready setup
//!
//! Prerequisites:
//! - Claude CLI must be installed and available in PATH
//! - Install with: npm install -g @anthropic-ai/claude-code-cli
//!
//! Run with: cargo run --example orchestrator_type_marker_production --features agent,derive

use llm_toolkit::agent::impls::ClaudeCodeAgent;
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator, StrategyMap, StrategyStep};
use llm_toolkit::{Agent, ToPrompt, type_marker};
use serde::{Deserialize, Serialize};

// Define output type using #[type_marker] attribute macro
// This automatically:
// - Adds __type field with proper default function
// - Implements TypeMarker trait
// - Excludes __type from LLM prompts (via ToPrompt)
// - Preserves __type during JSON serialization (for Orchestrator retrieval)
// IMPORTANT: #[type_marker] must be placed FIRST (before #[derive])
#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct DataAnalysisResult {
    /// Brief summary of the analysis
    pub summary: String,

    /// Key insights discovered from the data
    pub insights: Vec<String>,

    /// Quality score from 0 to 100
    pub score: i32,
}

// Define the agent using #[derive(Agent)] macro
// This automatically implements Agent trait with proper JSON parsing
#[derive(Agent)]
#[agent(
    name = "DataAnalyzer",
    expertise = "Expert data analyst capable of extracting insights from customer feedback and rating quality",
    output = "DataAnalysisResult"
)]
struct DataAnalyzerAgent;

// Mock strategy agent to provide a simple 1-step workflow
// In production, this would also use real LLM
#[derive(Debug)]
struct SimpleStrategyAgent;

#[async_trait::async_trait]
impl llm_toolkit::agent::Agent for SimpleStrategyAgent {
    type Output = StrategyMap;

    fn expertise(&self) -> &str {
        "Strategy generator"
    }

    fn name(&self) -> String {
        "SimpleStrategyAgent".to_string()
    }

    async fn execute(
        &self,
        _intent: llm_toolkit::agent::Payload,
    ) -> Result<Self::Output, llm_toolkit::agent::AgentError> {
        let mut strategy = StrategyMap::new("Analyze customer feedback data".to_string());

        strategy.add_step(StrategyStep::new(
            "analysis".to_string(),
            "Analyze the customer feedback data and extract insights".to_string(),
            "DataAnalyzerAgent".to_string(),  // Must match the struct name
            "Analyze the customer feedback provided in the task and extract key insights, provide a summary, and rate the overall quality with a score from 0 to 100.".to_string(),
            "Structured analysis with insights and quality score".to_string(),
        ));

        Ok(strategy)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("🎯 Production Orchestrator with TypeMarker Example\n");
    println!("This example demonstrates real Claude LLM integration with type-based retrieval\n");

    // Check if Claude CLI is available
    if !ClaudeCodeAgent::is_available() {
        eprintln!("❌ Claude CLI not found in PATH");
        eprintln!("\n💡 Please install the Claude CLI:");
        eprintln!("   npm install -g @anthropic-ai/claude-code-cli");
        eprintln!("   or visit: https://github.com/anthropics/anthropic-sdk-typescript\n");
        std::process::exit(1);
    }

    println!("✅ Claude CLI is available\n");

    // Create a simple blueprint
    let blueprint = BlueprintWorkflow::new(
        "Analyze customer feedback and extract actionable insights".to_string(),
    );

    // Create orchestrator with ClaudeCodeAgent for intent generation
    // and SimpleStrategyAgent for workflow generation
    let mut orchestrator = Orchestrator::with_internal_agents(
        blueprint,
        Box::new(ClaudeCodeAgent::default()), // Real Claude for intent generation
        Box::new(SimpleStrategyAgent),        // Mock for strategy (could also be real)
    );

    // Add the data analyzer agent (uses real Claude)
    let data_analyzer = DataAnalyzerAgent;

    // Debug: Show agent expertise (includes JSON schema)
    println!("🔍 DEBUG: Agent expertise:");
    println!("{}", "=".repeat(70));
    println!("{}", llm_toolkit::agent::Agent::expertise(&data_analyzer));
    println!("{}", "=".repeat(70));
    println!();

    orchestrator.add_agent(data_analyzer);

    println!("📋 Setup complete:");
    println!(
        "   - Registered {} agents",
        orchestrator.list_agents().len()
    );

    // Debug: Show registered agent names
    println!("   - Agent names:");
    for agent_name in orchestrator.list_agents() {
        println!("      • {}", agent_name);
    }

    println!("   - Using real Claude LLM for agent execution\n");

    let task = r#"
Analyze the following customer feedback:

"The product quality is excellent, but the delivery time was longer than expected.
Customer support was very helpful when I had questions. Overall satisfied but
shipping needs improvement."

Please provide:
1. A brief summary
2. Key insights about what's working and what needs improvement
3. An overall quality score (0-100)
"#;

    println!("🚀 Executing task with real Claude LLM...\n");
    println!("📝 Task: {}\n", task.lines().next().unwrap_or(""));

    // Execute the workflow
    let result = orchestrator.execute(task).await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            println!("✅ Workflow completed successfully!\n");

            // Display all step outputs in context
            println!("📦 Step outputs in context:");
            let all_outputs = orchestrator.get_all_step_outputs();
            for (step_id, output) in &all_outputs {
                println!("   Step: {}", step_id);

                // Check if __type field exists
                if let Some(type_value) = output.get("__type") {
                    println!("   ✅ __type field present: {}", type_value);
                } else {
                    println!("   ❌ __type field MISSING!");
                }

                // Show pretty JSON
                println!(
                    "   JSON:\n{}\n",
                    serde_json::to_string_pretty(output).unwrap_or_default()
                );
            }

            println!("🔍 Retrieving typed output using get_typed_output()...\n");

            // Try to retrieve the typed output
            match orchestrator.get_typed_output::<DataAnalysisResult>() {
                Ok(analysis_result) => {
                    println!("✅ Successfully retrieved DataAnalysisResult by type!\n");
                    println!("📊 Analysis Results:");
                    println!("{}", "=".repeat(70));
                    println!("\n📝 Summary:");
                    println!("   {}\n", analysis_result.summary);

                    println!("💡 Key Insights:");
                    for (i, insight) in analysis_result.insights.iter().enumerate() {
                        println!("   {}. {}", i + 1, insight);
                    }

                    println!("\n⭐ Quality Score: {}/100\n", analysis_result.score);
                    println!("{}", "=".repeat(70));

                    println!("\n✅ VERIFICATION PASSED:");
                    println!("   ✓ #[type_marker] macro automatically added __type field");
                    println!("   ✓ __type field preserved during JSON serialization");
                    println!("   ✓ Orchestrator stored typed output in context correctly");
                    println!("   ✓ get_typed_output() successfully retrieved by type");
                    println!("   ✓ Real Claude LLM generated structured output");
                    println!("\n🎉 Production-ready TypeMarker workflow complete!");
                }
                Err(e) => {
                    eprintln!("❌ FAILED to retrieve DataAnalysisResult: {}", e);
                    eprintln!("\nDEBUG: Check if __type field was preserved in serialization");
                    eprintln!("This might indicate a bug in the TypeMarker implementation");
                    std::process::exit(1);
                }
            }
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            if let Some(error) = result.error_message {
                eprintln!("❌ Workflow failed: {}", error);
            } else {
                eprintln!("❌ Workflow failed with unknown error");
            }
            std::process::exit(1);
        }
    }

    Ok(())
}
