//! Basic example demonstrating OllamaApiAgent usage.
//!
//! This example shows how to use OllamaApiAgent for local LLM inference
//! with Ollama server.
//!
//! # Prerequisites
//!
//! 1. Install Ollama: https://ollama.ai/download
//! 2. Pull a model: `ollama pull llama3`
//! 3. Start the server: `ollama serve` (usually runs automatically)
//!
//! # Run
//!
//! ```bash
//! cargo run --example ollama_basic --features ollama-api
//! ```

use llm_toolkit::agent::Agent;
use llm_toolkit::agent::impls::OllamaApiAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("=== Ollama API Agent Example ===\n");

    // =========================================================================
    // 1. Default Configuration
    // =========================================================================
    println!("1. Default Configuration (localhost:11434, llama3)");

    let agent = OllamaApiAgent::new();
    println!("   Model: {}", agent.model());
    println!("   Endpoint: {}", agent.endpoint());

    // Check if Ollama server is running
    if agent.is_healthy().await {
        println!("   Status: Ollama server is healthy\n");
    } else {
        println!("   Status: Ollama server is NOT running");
        println!("   Please start Ollama: ollama serve\n");
        return;
    }

    // =========================================================================
    // 2. List Available Models
    // =========================================================================
    println!("2. Available Models on Server");

    match agent.list_models().await {
        Ok(models) => {
            for model in &models {
                println!("   - {}", model);
            }
            println!();
        }
        Err(e) => {
            println!("   Error listing models: {}\n", e);
        }
    }

    // =========================================================================
    // 3. Custom Configuration
    // =========================================================================
    println!("3. Custom Configuration");

    let custom_agent = OllamaApiAgent::new()
        .with_model("llama3")
        .with_system_prompt("You are a helpful assistant. Be concise.");

    println!("   Model: {}", custom_agent.model());
    println!("   System prompt: Set\n");

    // =========================================================================
    // 4. Environment Variable Configuration
    // =========================================================================
    println!("4. Environment Variable Configuration");

    let env_agent = OllamaApiAgent::from_env();
    println!(
        "   Model (from OLLAMA_MODEL or default): {}",
        env_agent.model()
    );
    println!(
        "   Endpoint (from OLLAMA_HOST or default): {}\n",
        env_agent.endpoint()
    );

    // =========================================================================
    // 5. Execute a Prompt
    // =========================================================================
    println!("5. Execute a Prompt");
    println!("   Prompt: 'What is Rust programming language? Answer in one sentence.'\n");

    match agent
        .execute("What is Rust programming language? Answer in one sentence.".into())
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
    // 6. With System Prompt
    // =========================================================================
    println!("6. With System Prompt");

    let json_agent = OllamaApiAgent::new()
        .with_model("llama3")
        .with_system_prompt("You are a JSON generator. Always respond with valid JSON only.");

    println!("   Prompt: 'Generate a user object with name and age fields'\n");

    match json_agent
        .execute("Generate a user object with name and age fields".into())
        .await
    {
        Ok(response) => {
            println!("   Response: {}\n", response.trim());
        }
        Err(e) => {
            println!("   Error: {}\n", e);
        }
    }

    println!("=== Example Complete ===");
}
