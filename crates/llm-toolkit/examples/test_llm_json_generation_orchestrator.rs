//! Example: Test LLM JSON generation with TypeScript schema via Orchestrator
//!
//! This example tests that LLMs can correctly generate JSON from TypeScript-style schemas
//! when using the Orchestrator and Agent framework.
//! Run with: cargo run --example test_llm_json_generation_orchestrator --features agent,derive

use llm_toolkit::ToPrompt;
use llm_toolkit::agent::impls::ClaudeCodeAgent;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::extract_json;
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Test enum with descriptions
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, PartialEq)]
enum Priority {
    /// Urgent tasks that need immediate attention
    Critical,
    /// High priority tasks
    High,
    /// Regular priority tasks
    Medium,
    /// Low priority tasks
    Low,
}

/// Test struct with nested types
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
struct Task {
    /// The task title
    title: String,
    /// The task description
    description: String,
    /// Priority level
    priority: Priority,
    /// Is the task completed?
    completed: bool,
}

/// Agent that generates Task objects with custom expertise including nested enum schemas
struct TaskGeneratorAgent {
    inner: ClaudeCodeAgent,
}

impl Default for TaskGeneratorAgent {
    fn default() -> Self {
        Self {
            inner: ClaudeCodeAgent::new(),
        }
    }
}

impl TaskGeneratorAgent {
    fn get_expertise() -> &'static String {
        static EXPERTISE: OnceLock<String> = OnceLock::new();
        EXPERTISE.get_or_init(|| {
            let priority_schema = Priority::prompt_schema();
            let task_schema = Task::prompt_schema();

            format!(
                r#"Generate task objects based on user requirements.

IMPORTANT: Use the exact enum values shown in the Priority type definition (e.g., "High", not "HIGH").

Type Definitions:

{}

{}

IMPORTANT: Respond with valid JSON matching the Task schema above."#,
                priority_schema, task_schema
            )
        })
    }
}

#[async_trait::async_trait]
impl Agent for TaskGeneratorAgent {
    type Output = Task;
    type Expertise = String;

    fn expertise(&self) -> &String {
        Self::get_expertise()
    }

    async fn execute(&self, payload: Payload) -> Result<Task, AgentError> {
        let intent_with_schema = format!("{}\n\n{}", Self::get_expertise(), payload.to_text());
        let response = self
            .inner
            .execute(Payload::text(&intent_with_schema))
            .await?;
        let json_str = extract_json(&response).map_err(|e| AgentError::ParseError {
            message: format!("Failed to extract JSON: {}", e),
            reason: llm_toolkit::agent::error::ParseErrorReason::MarkdownExtractionFailed,
        })?;
        serde_json::from_str(&json_str).map_err(|e| {
            let reason = if e.is_eof() {
                llm_toolkit::agent::error::ParseErrorReason::UnexpectedEof
            } else if e.is_syntax() {
                llm_toolkit::agent::error::ParseErrorReason::InvalidJson
            } else {
                llm_toolkit::agent::error::ParseErrorReason::SchemaMismatch
            };
            AgentError::ParseError {
                message: format!("Failed to parse agent output: {}", e),
                reason,
            }
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🧪 Testing LLM JSON Generation via Orchestrator\n");

    // Check if claude CLI is available
    if !llm_toolkit::agent::impls::ClaudeCodeAgent::is_available() {
        eprintln!("❌ claude CLI not found in PATH");
        eprintln!("\n💡 This test requires Claude CLI. Skipping...");
        println!("✅ Test skipped (no Claude CLI available)");
        return Ok(());
    }

    println!("✅ claude CLI found\n");

    // Display generated TypeScript schemas
    let task_schema = Task::prompt_schema();
    let priority_schema = Priority::prompt_schema();

    println!("📋 Generated Schemas:\n");
    println!("--- Priority Enum ---");
    println!("{}\n", priority_schema);
    println!("--- Task Struct ---");
    println!("{}\n", task_schema);

    // Verify that Agent includes nested enum schemas in expertise
    let agent = TaskGeneratorAgent::default();
    let expertise = agent.expertise();

    println!("🤖 Agent Expertise (with auto-included schema):\n");
    println!("{}\n", expertise);

    // Verify both schemas are included
    assert!(
        expertise.contains("type Task = {"),
        "Agent expertise should include Task schema"
    );
    assert!(
        expertise.contains("type Priority ="),
        "Agent expertise should include Priority enum schema"
    );

    println!("✅ Both Task and Priority schemas are included in agent expertise\n");

    // Create orchestrator with simple workflow
    let blueprint = BlueprintWorkflow::new(
        r#"
        Task Generation Workflow:
        1. Generate a task object based on user requirements
        "#
        .to_string(),
    );

    let mut orchestrator = Orchestrator::new(blueprint);
    orchestrator.add_agent(TaskGeneratorAgent::default());

    println!("🔄 Executing workflow via Orchestrator...\n");

    // Execute via orchestrator
    let result = orchestrator
        .execute("Generate a task about 'Fix the login bug'")
        .await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            if let Some(json_value) = &result.final_output {
                println!("📥 Orchestrator Result:");
                println!("{}\n", serde_json::to_string_pretty(json_value)?);

                // Try to deserialize from JSON Value
                println!("🔍 Attempting to deserialize...");
                match serde_json::from_value::<Task>(json_value.clone()) {
                    Ok(task) => {
                        println!("✅ Successfully deserialized!");
                        println!("\n📊 Parsed Task:");
                        println!("  Title: {}", task.title);
                        println!("  Description: {}", task.description);
                        println!("  Priority: {:?}", task.priority);
                        println!("  Completed: {}", task.completed);

                        // Validate enum format
                        println!("\n✅ Enum deserialization successful!");
                        println!("   Priority value: {:?}", task.priority);

                        println!(
                            "\n🎉 TEST PASSED: Orchestrator + Agent generated valid JSON from TypeScript schema"
                        );
                    }
                    Err(e) => {
                        eprintln!("❌ Deserialization failed: {}", e);
                        eprintln!("\n💔 TEST FAILED: LLM output doesn't match expected format");
                        std::process::exit(1);
                    }
                }
            } else {
                eprintln!("❌ No output from orchestrator");
                eprintln!("\n💔 TEST FAILED: Orchestrator returned no output");
                std::process::exit(1);
            }
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            eprintln!("❌ Orchestrator failed: {:?}", result.error_message);
            eprintln!("\n💔 TEST FAILED: Orchestrator execution failed");
            std::process::exit(1);
        }
    }

    Ok(())
}
