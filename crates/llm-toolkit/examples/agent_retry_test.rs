//! Example demonstrating automatic retry behavior for Agent API.
//!
//! This shows how agents automatically retry on retryable errors (ParseError,
//! ProcessError, IoError) with configurable max_retries.
//!
//! Run with: cargo run --example agent_retry_test --features agent

use async_trait::async_trait;
use llm_toolkit::Agent;
use llm_toolkit::agent::{Agent as AgentTrait, AgentError, Payload};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Serialize, Deserialize, Debug, llm_toolkit::ToPrompt)]
#[prompt(mode = "full")]
struct TestOutput {
    message: String,
}

// Mock agent that fails N times before succeeding
#[allow(dead_code)]
struct FailingMockAgent {
    fail_count: Arc<AtomicU32>,
    failures_remaining: u32,
}

#[allow(dead_code)]
impl FailingMockAgent {
    fn new(failures_before_success: u32) -> Self {
        Self {
            fail_count: Arc::new(AtomicU32::new(0)),
            failures_remaining: failures_before_success,
        }
    }
}

#[async_trait]
impl AgentTrait for FailingMockAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock agent for testing retry behavior";
        &EXPERTISE
    }

    async fn execute(&self, _intent: Payload) -> Result<Self::Output, AgentError> {
        let current = self.fail_count.fetch_add(1, Ordering::SeqCst);

        if current < self.failures_remaining {
            println!("  ⚠️  Mock agent failing (attempt {})", current + 1);
            Err(AgentError::ParseError {
                message: "Simulated parse error".to_string(),
                reason: llm_toolkit::agent::error::ParseErrorReason::MarkdownExtractionFailed,
            })
        } else {
            println!("  ✅ Mock agent succeeded (attempt {})", current + 1);
            Ok(r#"{"message": "Success after retries!"}"#.to_string())
        }
    }

    fn name(&self) -> String {
        "FailingMockAgent".to_string()
    }
}

// Test Case 1: Default retry (max_retries = 3)
#[allow(dead_code)]
#[derive(Agent)]
#[agent(
    expertise = "Test agent with default retry settings",
    output = "TestOutput"
)]
struct DefaultRetryAgent;

// Test Case 2: Custom retry count
#[allow(dead_code)]
#[derive(Agent)]
#[agent(
    expertise = "Test agent with custom retry count",
    output = "TestOutput",
    max_retries = 5
)]
struct CustomRetryAgent;

// Test Case 3: No retry
#[allow(dead_code)]
#[derive(Agent)]
#[agent(
    expertise = "Test agent with retry disabled",
    output = "TestOutput",
    max_retries = 0
)]
struct NoRetryAgent;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("🔬 Agent Retry Strategy Test\n");
    println!("{}", "=".repeat(70));

    // Note: This example demonstrates the retry *structure* but cannot actually
    // test retries without a real LLM backend. The macro-generated code includes
    // retry logic that will activate when real agents encounter retryable errors.

    println!("\n📊 Test Case 1: Default Retry (max_retries = 3)");
    println!("{}", "-".repeat(70));
    println!("Configuration:");
    println!("  • max_retries = 3 (default)");
    println!("  • Retryable errors: ParseError, ProcessError, IoError");
    println!("  • Delay: 100ms * attempt_number");
    println!("\nBehavior:");
    println!("  • Attempt 1 fails → wait 100ms → retry");
    println!("  • Attempt 2 fails → wait 200ms → retry");
    println!("  • Attempt 3 fails → wait 300ms → retry");
    println!("  • Attempt 4 fails → return error (exhausted)");

    println!("\n{}", "=".repeat(70));
    println!("\n📊 Test Case 2: Custom Retry (max_retries = 5)");
    println!("{}", "-".repeat(70));
    println!("Configuration:");
    println!("  • max_retries = 5 (custom)");
    println!("  • Can retry up to 5 times");

    println!("\n{}", "=".repeat(70));
    println!("\n📊 Test Case 3: No Retry (max_retries = 0)");
    println!("{}", "-".repeat(70));
    println!("Configuration:");
    println!("  • max_retries = 0 (disabled)");
    println!("  • Fails immediately without retry");

    println!("\n{}", "=".repeat(70));
    println!("\n💡 Design Philosophy");
    println!("{}", "-".repeat(70));
    println!("Agent-level retries are intentionally simple:");
    println!("  • Limited retries (2-3 attempts)");
    println!("  • Fixed delay (100ms * attempt)");
    println!("  • Only retryable errors (ParseError, ProcessError, IoError)");
    println!("\nWhy simple?");
    println!("  • Fail fast and report to orchestrator");
    println!("  • Orchestrator has broader context for recovery:");
    println!("    - Try different agents");
    println!("    - Redesign strategy");
    println!("    - Escalate to human");
    println!("  • System-wide stability over local complexity");

    println!("\n{}", "=".repeat(70));
    println!("\n📝 Error Classification");
    println!("{}", "-".repeat(70));
    println!("Retryable (auto-retry enabled):");
    println!("  ✓ ParseError     - LLM output malformed");
    println!("  ✓ ProcessError   - Process communication issues");
    println!("  ✓ IoError        - Temporary I/O failures");
    println!("\nNon-retryable (fail immediately):");
    println!("  ✗ ExecutionFailed      - LLM logical errors");
    println!("  ✗ SerializationFailed  - Code-level issues");
    println!("  ✗ Other                - Unknown errors");

    println!("\n{}", "=".repeat(70));
    println!("\n🎉 Retry logic is built into the macro-generated code!");
    println!("When using real LLM backends, retries happen automatically.");
}
