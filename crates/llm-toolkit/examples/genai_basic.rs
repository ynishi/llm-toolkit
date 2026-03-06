//! Basic example demonstrating GenaiAgent - unified multi-provider LLM access.
//!
//! GenaiAgent uses the `genai` crate to talk to any supported provider
//! (OpenAI, Anthropic, Gemini, xAI, Groq, DeepSeek, Ollama, Cohere, ...)
//! through a single, normalized API. The provider is auto-resolved from
//! the model name.
//!
//! # Prerequisites
//!
//! Set the API key for the provider you want to use:
//! - OpenAI: `OPENAI_API_KEY`
//! - Anthropic: `ANTHROPIC_API_KEY`
//! - Gemini: `GEMINI_API_KEY`
//! - etc.
//!
//! # Run
//!
//! ```bash
//! cargo run --example genai_basic --features genai-api
//! ```

use llm_toolkit::agent::Agent;
use llm_toolkit::agent::impls::GenaiAgent;

/// Models to demonstrate. Each resolves to a different provider automatically.
const DEMO_MODELS: &[(&str, &str)] = &[
    ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ("gpt-4o-mini", "OPENAI_API_KEY"),
    ("gemini-2.5-flash", "GEMINI_API_KEY"),
];

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("=== GenaiAgent: Unified Multi-Provider Example ===\n");

    let question = "What is Rust's ownership model? Answer in exactly one sentence.";

    // =========================================================================
    // 1. Try each provider that has an API key configured
    // =========================================================================
    println!("Question: {}\n", question);

    let mut any_succeeded = false;

    for (model, env_var) in DEMO_MODELS {
        if std::env::var(env_var).is_err() {
            println!("[SKIP] {} ({} not set)", model, env_var);
            continue;
        }

        let agent = GenaiAgent::new(*model);
        print!("[{}] ", model);

        match agent.execute(question.into()).await {
            Ok(response) => {
                println!("{}", response.trim());
                any_succeeded = true;
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    if !any_succeeded {
        println!("\nNo API keys found. Set at least one of:");
        for (_, env_var) in DEMO_MODELS {
            println!("  export {}=your-key-here", env_var);
        }
        println!();
        return;
    }

    // =========================================================================
    // 2. Builder pattern with options
    // =========================================================================
    println!("\n--- With system prompt and options ---\n");

    // Pick the first available model
    let model = DEMO_MODELS
        .iter()
        .find(|(_, env)| std::env::var(env).is_ok())
        .map(|(m, _)| *m)
        .expect("at least one key should be set");

    let agent = GenaiAgent::new(model)
        .with_system("You are a concise technical writer. Use bullet points.")
        .with_temperature(0.3)
        .with_max_tokens(256);

    match agent
        .execute("List 3 benefits of Rust's type system.".into())
        .await
    {
        Ok(response) => {
            println!("[{}] {}", model, response.trim());
        }
        Err(e) => {
            println!("[{}] Error: {}", model, e);
        }
    }

    // =========================================================================
    // 3. Model switching at runtime
    // =========================================================================
    println!("\n--- Runtime model switching ---\n");

    let available: Vec<&str> = DEMO_MODELS
        .iter()
        .filter(|(_, env)| std::env::var(env).is_ok())
        .map(|(m, _)| *m)
        .collect();

    if available.len() >= 2 {
        let prompt = "What is 2 + 2? Answer with just the number.";
        for m in &available[..2] {
            let agent = GenaiAgent::new(*m);
            match agent.execute(prompt.into()).await {
                Ok(r) => println!("[{}] {}", m, r.trim()),
                Err(e) => println!("[{}] Error: {}", m, e),
            }
        }
    } else {
        println!("(Need 2+ API keys to demo model switching)");
    }

    println!("\n=== Example Complete ===");
}
