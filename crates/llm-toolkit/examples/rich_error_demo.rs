//! Demonstrates rich error metadata system for enhanced debugging and observability.
//!
//! This example shows how to use the rich error system to create errors with
//! detailed contextual information including agent name, operation, custom context,
//! and error chaining.

use llm_toolkit::agent::{AgentError, ErrorMetadata, ParseErrorReason};
use serde_json::json;

fn main() {
    println!("=== Rich Error System Demo ===\n");

    // Example 1: Simple error (backward compatible)
    println!("1. Simple Error (Backward Compatible):");
    let simple_err = AgentError::ExecutionFailed("Task failed".to_string());
    println!("   {}", simple_err);
    simple_err.trace_error(); // Basic logging
    println!();

    // Example 2: Rich error with builder
    println!("2. Rich Error with Builder API:");
    let rich_err = AgentError::execution_failed_rich("Model timeout after 30 seconds")
        .agent("GeminiAgent")
        .expertise("Fast inference agent")
        .operation("execute")
        .context("model", json!("gemini-2.5-flash"))
        .context("timeout_ms", json!(30000))
        .context("input_tokens", json!(1024))
        .build();

    println!("   {}", rich_err);
    println!("   Metadata available: {}", rich_err.metadata().is_some());
    if let Some(metadata) = rich_err.metadata() {
        println!("   - Agent: {:?}", metadata.agent_name);
        println!("   - Operation: {:?}", metadata.operation);
        println!("   - Context: {:?}", metadata.context);
    }
    rich_err.trace_error(); // Structured logging with all metadata
    println!();

    // Example 3: Error chaining
    println!("3. Error Chaining (Caused By):");
    let inner_err = AgentError::IoError(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "config.json not found",
    ));

    let outer_err = AgentError::execution_failed_rich("Failed to initialize agent")
        .agent("ConfigLoader")
        .operation("load_config")
        .caused_by(&inner_err)
        .build();

    println!("   {}", outer_err);
    if let Some(metadata) = outer_err.metadata() {
        println!("   Caused by: {:?}", metadata.caused_by_description);
    }
    println!();

    // Example 4: Converting simple error to rich
    println!("4. Converting Simple -> Rich:");
    let simple = AgentError::ExecutionFailed("Generic failure".to_string());
    println!("   Before: {}", simple);
    println!("   Has metadata: {}", simple.metadata().is_some());

    let upgraded = simple.with_metadata(
        ErrorMetadata::new()
            .with_agent("UpgradedAgent")
            .with_operation("upgrade_demo")
            .with_context("upgraded", json!(true)),
    );
    println!("   After: {}", upgraded);
    println!("   Has metadata: {}", upgraded.metadata().is_some());
    println!();

    // Example 5: Parse error with rich metadata
    println!("5. Rich Parse Error:");
    let parse_err = AgentError::ParseErrorRich {
        message: "Failed to parse JSON response".to_string(),
        reason: ParseErrorReason::InvalidJson,
        metadata: ErrorMetadata::new()
            .with_agent("ClaudeCodeAgent")
            .with_operation("parse_json_output")
            .with_context("expected_schema", json!("ReviewResult"))
            .with_context("actual_output_preview", json!("Not a JSON...")),
    };

    println!("   {}", parse_err);
    println!("   Is retryable: {}", parse_err.is_retryable());
    parse_err.trace_error();
    println!();

    // Example 6: Process error with retry information
    println!("6. Rich Process Error (Rate Limiting):");
    let process_err = AgentError::ProcessErrorRich {
        status_code: Some(429),
        message: "Rate limit exceeded".to_string(),
        is_retryable: true,
        retry_after: Some(std::time::Duration::from_secs(60)),
        metadata: ErrorMetadata::new()
            .with_agent("GeminiAgent")
            .with_operation("api_request")
            .with_context("endpoint", json!("/v1/models/generate"))
            .with_context("requests_today", json!(1000)),
    };

    println!("   {}", process_err);
    println!("   Is retryable: {}", process_err.is_retryable());
    println!("   Retry delay (attempt 1): {:?}", process_err.retry_delay(1));
    process_err.trace_error();
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Key Features Demonstrated:");
    println!("  ✓ Backward compatibility (simple errors still work)");
    println!("  ✓ Rich error builder with fluent API");
    println!("  ✓ Error chaining for root cause analysis");
    println!("  ✓ Simple -> Rich conversion");
    println!("  ✓ Structured logging with trace_error()");
    println!("  ✓ All error types support metadata");
}
