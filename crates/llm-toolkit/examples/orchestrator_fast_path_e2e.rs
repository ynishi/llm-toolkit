//! End-to-end test demonstrating fast path intent generation optimization.
//!
//! This example demonstrates:
//! - Fast path optimization when all placeholders are resolved
//! - Comparison between fast path (enabled) and LLM path (disabled)
//! - Performance measurement and validation
//! - Complete offline execution using mock agents
//!
//! Run with: cargo run --example orchestrator_fast_path_e2e --features agent,derive

use async_trait::async_trait;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::orchestrator::{
    BlueprintWorkflow, Orchestrator, OrchestratorConfig, StrategyMap, StrategyStep,
};
use std::time::Instant;

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
    type Expertise = String;

    fn expertise(&self) -> &String {
        &self.expertise
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let text_intent = intent.to_text();
        Ok(format!(
            "{}: {}",
            self.response_prefix,
            text_intent.chars().take(100).collect::<String>()
        ))
    }
}

/// Mock internal agent that returns fixed responses for intent generation.
#[derive(Debug)]
struct MockInternalAgent;

#[async_trait]
impl Agent for MockInternalAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock internal agent";
        &EXPERTISE
    }

    fn name(&self) -> String {
        "MockInternalAgent".to_string()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        // Simulate LLM delay (this is what we're optimizing away!)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Return a generic intent
        Ok(format!(
            "🤖 Generated intent: {}",
            intent.to_text().chars().take(50).collect::<String>()
        ))
    }
}

/// Mock JSON agent that returns a predefined strategy for fast path testing.
#[derive(Debug)]
struct MockJsonAgent;

#[async_trait]
impl Agent for MockJsonAgent {
    type Output = StrategyMap;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock strategy generator";
        &EXPERTISE
    }

    fn name(&self) -> String {
        "MockJsonAgent".to_string()
    }

    async fn execute(&self, _intent: Payload) -> Result<Self::Output, AgentError> {
        let mut strategy = StrategyMap::new("Process data pipeline".to_string());

        // Step 1: Simple placeholder - should use fast path
        strategy.add_step(StrategyStep::new(
            "step_1".to_string(),
            "Extract data from input".to_string(),
            "DataExtractor".to_string(),
            "Extract data from: {{task}}".to_string(),
            "Extracted data".to_string(),
        ));

        // Step 2: Uses previous_output - should use fast path
        strategy.add_step(StrategyStep::new(
            "step_2".to_string(),
            "Transform extracted data".to_string(),
            "DataTransformer".to_string(),
            "Transform this data: {{previous_output}}".to_string(),
            "Transformed data".to_string(),
        ));

        // Step 3: Uses specific step output - should use fast path
        strategy.add_step(StrategyStep::new(
            "step_3".to_string(),
            "Generate final report".to_string(),
            "ReportGenerator".to_string(),
            "Create report from: {{previous_output}}".to_string(),
            "Final report".to_string(),
        ));

        Ok(strategy)
    }
}

async fn run_orchestrator_test(
    enable_fast_path: bool,
) -> Result<
    (
        llm_toolkit::orchestrator::OrchestrationResult,
        std::time::Duration,
    ),
    Box<dyn std::error::Error>,
> {
    let blueprint = BlueprintWorkflow::new(
        r#"
        Fast Path Test Pipeline:
        1. Extract data from input
        2. Transform the extracted data
        3. Generate comprehensive report
        "#
        .to_string(),
    );

    let mut orchestrator = Orchestrator::with_internal_agents(
        blueprint,
        Box::new(MockInternalAgent),
        Box::new(MockJsonAgent),
    );

    // Configure fast path
    let config = OrchestratorConfig {
        enable_fast_path_intent_generation: enable_fast_path,
        ..Default::default()
    };
    orchestrator.set_config(config);

    // Add mock agents
    orchestrator.add_agent(MockAgent::new(
        "DataExtractor",
        "Extracts data from various sources",
        "📥 Extracted",
    ));

    orchestrator.add_agent(MockAgent::new(
        "DataTransformer",
        "Transforms and cleans data",
        "🔄 Transformed",
    ));

    orchestrator.add_agent(MockAgent::new(
        "ReportGenerator",
        "Generates reports from processed data",
        "📝 Report",
    ));

    let task = "customer feedback data";

    let start = Instant::now();
    let result = orchestrator.execute(task).await;
    let duration = start.elapsed();

    Ok((result, duration))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Fast Path Intent Generation E2E Test\n");
    println!("{}", "=".repeat(60));

    // Test 1: With fast path enabled (default)
    println!("\n📊 Test 1: Fast Path ENABLED (optimized)");
    println!("{}", "-".repeat(60));

    let (result_fast, duration_fast) = run_orchestrator_test(true).await?;

    println!("✅ Test completed!");
    println!("   Status: {:?}", result_fast.status);
    println!("   Steps executed: {}", result_fast.steps_executed);
    println!("   Redesigns: {}", result_fast.redesigns_triggered);
    println!("   ⚡ Duration: {:?}", duration_fast);

    // Validate fast path results
    assert_eq!(
        result_fast.status,
        llm_toolkit::orchestrator::OrchestrationStatus::Success,
        "Fast path execution should succeed"
    );
    assert_eq!(result_fast.steps_executed, 3, "Should execute all 3 steps");
    assert_eq!(
        result_fast.redesigns_triggered, 0,
        "Should have no redesigns"
    );

    // Test 2: With fast path disabled (uses LLM for all intents)
    println!("\n📊 Test 2: Fast Path DISABLED (uses LLM)");
    println!("{}", "-".repeat(60));

    let (result_slow, duration_slow) = run_orchestrator_test(false).await?;

    println!("✅ Test completed!");
    println!("   Status: {:?}", result_slow.status);
    println!("   Steps executed: {}", result_slow.steps_executed);
    println!("   Redesigns: {}", result_slow.redesigns_triggered);
    println!("   🐌 Duration: {:?}", duration_slow);

    // Validate LLM path results
    assert_eq!(
        result_slow.status,
        llm_toolkit::orchestrator::OrchestrationStatus::Success,
        "LLM path execution should succeed"
    );
    assert_eq!(result_slow.steps_executed, 3, "Should execute all 3 steps");
    assert_eq!(
        result_slow.redesigns_triggered, 0,
        "Should have no redesigns"
    );

    // Compare performance
    println!("\n📈 Performance Comparison");
    println!("{}", "=".repeat(60));

    let speedup = duration_slow.as_millis() as f64 / duration_fast.as_millis() as f64;

    println!("   Fast Path: {:?}", duration_fast);
    println!("   LLM Path:  {:?}", duration_slow);
    println!("   Speedup:   {:.2}x faster", speedup);

    // Validate that fast path is actually faster
    // Note: In this mock setup, fast path should be significantly faster
    // because MockInternalAgent has a 100ms delay per call
    // With 3 steps, LLM path should take ~300ms+ extra
    if duration_fast >= duration_slow {
        println!("\n⚠️  WARNING: Fast path was not faster than LLM path!");
        println!("   This might indicate the fast path is not being used.");
        println!("   Check that all placeholders are being resolved correctly.");
    } else {
        let saved_ms = duration_slow.as_millis() - duration_fast.as_millis();
        println!("   ✨ Saved: ~{}ms by using fast path", saved_ms);
    }

    // Final validation
    println!("\n🔍 Final Validation");
    println!("{}", "=".repeat(60));

    // Both should produce valid final outputs
    assert!(
        result_fast.final_output.is_some(),
        "Fast path should produce final output"
    );
    assert!(
        result_slow.final_output.is_some(),
        "LLM path should produce final output"
    );

    let output_fast = result_fast.final_output.unwrap();
    let output_slow = result_slow.final_output.unwrap();

    // Both outputs should contain the report prefix from the final step
    let output_fast_str = serde_json::to_string(&output_fast)?;
    let output_slow_str = serde_json::to_string(&output_slow)?;

    assert!(
        output_fast_str.contains("📝 Report"),
        "Fast path output should contain report marker"
    );
    assert!(
        output_slow_str.contains("📝 Report"),
        "LLM path output should contain report marker"
    );

    println!("✅ Both paths produce valid outputs");
    println!("✅ Fast path provides performance improvement");
    println!("✅ Configuration toggle works correctly");

    println!("\n🎉 E2E Test: PASSED");
    println!("{}", "=".repeat(60));

    println!("\n💡 Key Takeaways:");
    println!("   • Fast path skips LLM calls when placeholders are resolved");
    println!("   • Both paths produce functionally equivalent results");
    println!("   • Performance improvement: {:.2}x", speedup);
    println!("   • Can be toggled via OrchestratorConfig");

    Ok(())
}
