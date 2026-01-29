//! Basic example demonstrating LlamaCppServerAgent usage.
//!
//! This example shows how to use LlamaCppServerAgent for local LLM inference
//! with llama-server (llama.cpp HTTP server).
//!
//! # Prerequisites
//!
//! 1. Build llama.cpp: https://github.com/ggerganov/llama.cpp
//! 2. Download a GGUF model (e.g., from HuggingFace)
//! 3. Start the server:
//!    ```bash
//!    llama-server -m model.gguf --host 0.0.0.0 --port 8080
//!    ```
//!
//! # Run
//!
//! ```bash
//! cargo run --example llama_cpp_server_basic --features llama-cpp-server
//! ```

use llm_toolkit::agent::impls::{ChatTemplate, LlamaCppServerAgent};
use llm_toolkit::agent::Agent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("=== llama-server API Agent Example ===\n");

    // =========================================================================
    // 1. Default Configuration
    // =========================================================================
    println!("1. Default Configuration (localhost:8080, Llama3 template)");

    let agent = LlamaCppServerAgent::new();
    println!("   Endpoint: {}", agent.endpoint());

    // Check if server is running
    if agent.is_healthy().await {
        println!("   Status: llama-server is healthy\n");
    } else {
        println!("   Status: llama-server is NOT running");
        println!("   Please start: llama-server -m model.gguf --port 8080\n");
        return;
    }

    // =========================================================================
    // 2. Check Available Slots
    // =========================================================================
    println!("2. Server Slots (Concurrent Request Capacity)");

    match agent.available_slots().await {
        Ok(slots) => println!("   Available slots: {}\n", slots),
        Err(e) => println!("   Could not get slots: {}\n", e),
    }

    // =========================================================================
    // 3. Different Chat Templates
    // =========================================================================
    println!("3. Chat Template Examples");

    let templates = [
        ("Llama3", ChatTemplate::Llama3),
        ("Qwen", ChatTemplate::Qwen),
        ("Mistral", ChatTemplate::Mistral),
        ("None (raw)", ChatTemplate::None),
    ];

    for (name, template) in templates {
        let formatted = template.format("Hello");
        println!("   {}: {}", name, formatted.replace('\n', "\\n"));
    }
    println!();

    // =========================================================================
    // 4. Custom Configuration
    // =========================================================================
    println!("4. Custom Configuration");

    let custom_agent = LlamaCppServerAgent::new()
        .with_endpoint("http://localhost:8080")
        .with_chat_template(ChatTemplate::Llama3)
        .with_max_tokens(256)
        .with_temperature(0.7)
        .with_system_prompt("You are a helpful assistant. Be concise.");

    println!("   Max tokens: 256");
    println!("   Temperature: 0.7");
    println!("   System prompt: Set\n");

    // =========================================================================
    // 5. Execute a Prompt
    // =========================================================================
    println!("5. Execute a Prompt");
    println!("   Prompt: 'What is Rust? Answer in one sentence.'\n");

    match custom_agent
        .execute("What is Rust? Answer in one sentence.".into())
        .await
    {
        Ok(response) => {
            println!("   Response: {}\n", response.trim());
        }
        Err(e) => {
            println!("   Error: {}\n", e);
        }
    }

    // =========================================================================
    // 6. JSON Generation with System Prompt
    // =========================================================================
    println!("6. JSON Generation");

    let json_agent = LlamaCppServerAgent::new()
        .with_chat_template(ChatTemplate::Llama3)
        .with_system_prompt(
            "You are a JSON generator. Always respond with valid JSON only, no explanation.",
        )
        .with_temperature(0.3); // Lower temperature for more deterministic output

    println!("   Prompt: 'Generate a user object with name and age'\n");

    match json_agent
        .execute("Generate a user object with name and age".into())
        .await
    {
        Ok(response) => {
            println!("   Response: {}\n", response.trim());
        }
        Err(e) => {
            println!("   Error: {}\n", e);
        }
    }

    // =========================================================================
    // 7. Environment Variable Configuration
    // =========================================================================
    println!("7. Environment Variable Configuration");

    let env_agent = LlamaCppServerAgent::from_env();
    println!(
        "   Endpoint (from LLAMA_SERVER_ENDPOINT or default): {}",
        env_agent.endpoint()
    );
    println!("   Variables: LLAMA_SERVER_ENDPOINT, LLAMA_SERVER_MAX_TOKENS, LLAMA_SERVER_TEMPERATURE\n");

    println!("=== Example Complete ===");
}
