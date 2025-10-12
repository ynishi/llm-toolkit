//! Example: Test LLM JSON generation with TypeScript schema (Direct Prompt)
//!
//! This example tests that LLMs can correctly generate JSON from TypeScript-style schemas
//! by directly constructing prompts and calling the LLM.
//! Run with: cargo run --example test_llm_json_generation_direct --features agent,derive

use llm_toolkit::agent::impls::ClaudeCodeAgent;
use llm_toolkit::agent::payload::Payload;
use llm_toolkit::agent::Agent;
use llm_toolkit::extract_json;
use llm_toolkit::ToPrompt;
use serde::{Deserialize, Serialize};

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

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🧪 Testing LLM JSON Generation with TypeScript Schema\n");

    // Check if claude CLI is available
    if !ClaudeCodeAgent::is_available() {
        eprintln!("❌ claude CLI not found in PATH");
        eprintln!("\n💡 This test requires Claude CLI. Skipping...");
        println!("✅ Test skipped (no Claude CLI available)");
        return Ok(());
    }

    println!("✅ claude CLI found\n");

    // Generate TypeScript schema
    let task_schema = Task::prompt_schema();
    let priority_schema = Priority::prompt_schema();

    println!("📋 Generated Schemas:\n");
    println!("--- Priority Enum ---");
    println!("{}\n", priority_schema);
    println!("--- Task Struct ---");
    println!("{}\n", task_schema);

    // Create prompt for LLM
    let prompt = format!(
        r#"Generate a JSON object matching these TypeScript type definitions:

{}

{}

Generate a task about "Fix the login bug". Return ONLY the JSON, no markdown code blocks."#,
        priority_schema, task_schema
    );

    println!("🤖 Calling Claude to generate JSON...\n");

    let agent = ClaudeCodeAgent::new();
    let payload = Payload::from(prompt);
    let response = agent.execute(payload).await;

    match response {
        Ok(json_str) => {
            println!("📥 LLM Response:");
            println!("{}\n", json_str);

            // Extract JSON from markdown code blocks if present
            let extracted = extract_json(&json_str)?;

            // Try to parse as JSON
            println!("🔍 Attempting to deserialize...");
            match serde_json::from_str::<Task>(&extracted) {
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

                    println!("\n🎉 TEST PASSED: LLM generated valid JSON from TypeScript schema");
                }
                Err(e) => {
                    eprintln!("❌ Deserialization failed: {}", e);
                    eprintln!("\n💔 TEST FAILED: LLM output doesn't match expected format");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ LLM call failed: {}", e);
            eprintln!("\n💔 TEST FAILED: Could not get response from LLM");
            std::process::exit(1);
        }
    }

    Ok(())
}
