#### Basic Orchestrator Usage


```rust
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator};
use llm_toolkit::agent::impls::ClaudeCodeAgent;

#[tokio::main]
async fn main() {
    // Define workflow in natural language
    let blueprint = BlueprintWorkflow::new(r#"
        Technical Article Workflow:
        1. Analyze the topic and create an outline
        2. Research key concepts
        3. Write the main content
        4. Generate title and summary
        5. Review and refine
    "#.to_string());

    // Create orchestrator (InnerValidatorAgent is automatically registered)
    let mut orchestrator = Orchestrator::new(blueprint);
    orchestrator.add_agent(Box::new(ClaudeCodeAgent::new()));

    // Execute workflow - the orchestrator will:
    // - Generate an optimal execution strategy
    // - Assign agents to each step
    // - Handle errors with adaptive redesign
    let result = orchestrator.execute(
        "Write a beginner-friendly article about Rust ownership"
    ).await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            println!("✅ Workflow completed!");
            println!("Steps executed: {}", result.steps_executed);
            println!("Redesigns triggered: {}", result.redesigns_triggered);
            if let Some(output) = result.final_output {
                println!("\nFinal output:\n{}", output);
            }
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            eprintln!("❌ Workflow failed: {:?}", result.error_message);
        }
    }
}
```

