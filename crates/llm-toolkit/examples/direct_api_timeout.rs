//! Example showing per-request timeout configuration for Direct API agents.
//!
//! Run with: cargo run --example direct_api_timeout --features agent

use llm_toolkit::agent::impls::{AnthropicApiAgent, GeminiApiAgent, OpenAIApiAgent};

fn main() {
    // Configure API agents with explicit request timeouts.
    let _openai = OpenAIApiAgent::new("dummy-key", "gpt-5")
        .with_max_tokens(1024)
        .with_timeout_secs(30);

    let _gemini = GeminiApiAgent::new("dummy-key", "gemini-2.5-flash")
        .with_google_search(false)
        .with_timeout_secs(45);

    let _anthropic = AnthropicApiAgent::new("dummy-key", "claude-sonnet-4-6")
        .with_system("You are concise and precise.")
        .with_timeout_secs(60);

    println!("Configured Direct API agents with explicit request timeouts.");
}
