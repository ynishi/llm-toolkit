use llm_toolkit::define_intent;
use llm_toolkit::intent::{IntentExtractionError, IntentExtractor};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[define_intent]
#[intent(
    prompt = r#"You are an AI assistant analyzing user intent.

Context: {{ context }}
User Query: {{ user_query }}

{{ intents_doc }}

Please identify the user's intent based on the query and context provided.
Respond with the appropriate intent wrapped in <analysis> tags."#,
    extractor_tag = "analysis"
)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// AI Assistant Intents
enum AssistantIntent {
    /// User wants to ask a question
    AskQuestion,
    /// User wants to execute a command
    ExecuteCommand,
    /// User wants to get help or documentation
    GetHelp,
    /// User wants to provide feedback
    ProvideFeedback,
    /// User wants to report an issue
    ReportIssue,
}

impl FromStr for AssistantIntent {
    type Err = IntentExtractionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "AskQuestion" => Ok(AssistantIntent::AskQuestion),
            "ExecuteCommand" => Ok(AssistantIntent::ExecuteCommand),
            "GetHelp" => Ok(AssistantIntent::GetHelp),
            "ProvideFeedback" => Ok(AssistantIntent::ProvideFeedback),
            "ReportIssue" => Ok(AssistantIntent::ReportIssue),
            _ => Err(IntentExtractionError::ParseFailed {
                value: format!("Unknown intent: {}", s),
            }),
        }
    }
}

fn main() {
    println!("=== Testing Comprehensive Intent Definition ===\n");

    // Test 1: Build prompt with multiple parameters
    let context = "User is working on a Rust project";
    let user_query = "How do I implement a trait for my struct?";

    let prompt = build_assistant_intent_prompt(context, user_query);
    println!("Generated Prompt:\n{}\n", prompt);
    println!("---\n");

    // Test 2: Verify the extractor tag
    let extractor = AssistantIntentExtractor;
    println!(
        "Extractor Tag: {}\n",
        AssistantIntentExtractor::EXTRACTOR_TAG
    );

    // Test 3: Extract various intents
    let test_cases = vec![
        (
            "<analysis>AskQuestion</analysis> The user is asking about Rust traits.",
            AssistantIntent::AskQuestion,
        ),
        (
            "<analysis>ExecuteCommand</analysis> User wants to run a command.",
            AssistantIntent::ExecuteCommand,
        ),
        (
            "<analysis>GetHelp</analysis> User needs documentation.",
            AssistantIntent::GetHelp,
        ),
    ];

    for (response, expected) in test_cases {
        match extractor.extract_intent(response) {
            Ok(intent) => {
                assert_eq!(intent, expected);
                println!("✓ Successfully extracted: {:?}", intent);
            }
            Err(e) => {
                println!("✗ Failed to extract from '{}': {:?}", response, e);
            }
        }
    }

    println!("\n---\n");

    // Test 4: Error handling
    println!("Testing error cases:");

    // Missing tag
    let bad_response = "No tag here, just text";
    match extractor.extract_intent(bad_response) {
        Ok(intent) => println!("✗ Unexpected success: {:?}", intent),
        Err(IntentExtractionError::TagNotFound { tag }) => {
            println!("✓ Correctly caught missing tag error for '{}'", tag);
        }
        Err(e) => println!("✗ Unexpected error: {:?}", e),
    }

    // Invalid intent value
    let bad_intent = "<analysis>InvalidIntent</analysis>";
    match extractor.extract_intent(bad_intent) {
        Ok(intent) => println!("✗ Unexpected success: {:?}", intent),
        Err(IntentExtractionError::ParseFailed { value }) => {
            println!("✓ Correctly caught parse error: {}", value);
        }
        Err(e) => println!("✗ Unexpected error: {:?}", e),
    }

    println!("\n=== All tests completed successfully! ===");
}
