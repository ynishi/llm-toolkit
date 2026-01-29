//! Basic example demonstrating LlamaCppNativeAgent usage.
//!
//! This example shows how to use LlamaCppNativeAgent for local LLM inference
//! with native llama.cpp binding (no HTTP server required).
//!
//! # Prerequisites
//!
//! Models are automatically downloaded from HuggingFace Hub on first use.
//!
//! # Run
//!
//! ```bash
//! cargo run --example llama_cpp_native_basic --features llama-cpp-native
//! ```
//!
//! # With GPU acceleration (macOS Metal)
//!
//! ```bash
//! cargo run --example llama_cpp_native_basic --features "llama-cpp-native,metal"
//! ```

use llm_toolkit::agent::Agent;
use llm_toolkit::agent::impls::{LlamaCppNativeAgent, LlamaCppNativeConfig, NativeChatTemplate};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("llm_toolkit=info".parse().unwrap()),
        )
        .init();

    println!("=== llama.cpp Native Agent Example ===\n");

    // =========================================================================
    // 1. Available Model Presets
    // =========================================================================
    println!("1. Available Model Presets\n");

    let presets = [
        ("lfm2_1b", "LiquidAI LFM2.5 1.2B (Q4_K_M)"),
        ("qwen_0_5b", "Qwen 2.5 0.5B (Q4_K_M) - Ultra light"),
        ("qwen_1_5b", "Qwen 2.5 1.5B (Q4_K_M)"),
        ("qwen_3b", "Qwen 2.5 3B (Q4_K_M)"),
        ("llama3_1b", "Llama 3.2 1B (Q4_K_M)"),
        ("llama3_3b", "Llama 3.2 3B (Q4_K_M)"),
        ("phi3_mini", "Microsoft Phi-3 Mini (Q4_K_M)"),
    ];

    for (name, desc) in presets {
        println!("   {:12} - {}", name, desc);
    }
    println!();

    // =========================================================================
    // 2. Chat Templates
    // =========================================================================
    println!("2. Chat Template Examples\n");

    let templates = [
        ("Llama3", NativeChatTemplate::Llama3),
        ("Qwen", NativeChatTemplate::Qwen),
        ("LFM2", NativeChatTemplate::Lfm2),
        ("Mistral", NativeChatTemplate::Mistral),
        ("None", NativeChatTemplate::None),
    ];

    for (name, template) in templates {
        let formatted = template.format("Hello");
        println!(
            "   {:8}: {}",
            name,
            formatted
                .replace('\n', "\\n")
                .chars()
                .take(60)
                .collect::<String>()
        );
    }
    println!();

    // =========================================================================
    // 3. Create Agent with Smallest Model (Qwen 0.5B)
    // =========================================================================
    println!("3. Creating Agent (Qwen 2.5 0.5B - smallest, fastest)");
    println!("   Note: First run will download the model from HuggingFace\n");

    let config = LlamaCppNativeConfig::qwen_0_5b()
        .with_max_tokens(128)
        .with_temperature(0.7);

    println!("   Model: {}", config.display_name());
    println!("   Max tokens: {}", config.max_tokens);
    println!("   Temperature: {}\n", config.temperature);

    let agent = match LlamaCppNativeAgent::try_new(config) {
        Ok(agent) => {
            println!("   Agent created successfully!\n");
            agent
        }
        Err(e) => {
            println!("   Failed to create agent: {}\n", e);
            println!("   This may happen if:");
            println!("   - Network connection issues (for first-time download)");
            println!("   - Insufficient disk space");
            println!("   - llama.cpp compilation issues\n");
            return;
        }
    };

    // =========================================================================
    // 4. Execute a Simple Prompt
    // =========================================================================
    println!("4. Execute a Prompt");
    println!("   Prompt: 'What is Rust? Answer in one sentence.'\n");

    match agent
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
    // 5. Custom Configuration with System Prompt
    // =========================================================================
    println!("5. Custom Configuration with System Prompt");

    let custom_config = LlamaCppNativeConfig::qwen_0_5b()
        .with_max_tokens(64)
        .with_temperature(0.3)
        .with_system_prompt("You are a helpful assistant. Be very concise.");

    let custom_agent = match LlamaCppNativeAgent::try_new(custom_config) {
        Ok(agent) => agent,
        Err(e) => {
            println!("   Failed to create custom agent: {}\n", e);
            return;
        }
    };

    println!("   Prompt: 'List 3 programming languages'\n");

    match custom_agent
        .execute("List 3 programming languages".into())
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
    // 6. GPU Acceleration Info
    // =========================================================================
    println!("6. GPU Acceleration\n");
    println!("   To enable GPU acceleration, use feature flags:\n");
    println!("   macOS (Metal):");
    println!("     cargo run --features \"llama-cpp-native,metal\"\n");
    println!("   NVIDIA (CUDA):");
    println!("     cargo run --features \"llama-cpp-native,cuda\"\n");
    println!("   Then use .with_gpu_layers(N) in config:");
    println!("     let config = LlamaCppNativeConfig::qwen_3b().with_gpu_layers(32);\n");

    println!("=== Example Complete ===");
}
