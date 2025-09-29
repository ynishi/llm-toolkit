use llm_toolkit::{IntentError, IntentExtractor, IntentFrame, define_intent};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[define_intent]
#[intent(
    prompt = r#"
You are a helpful AI assistant. Analyze the user's query and determine their intent.

User Query: {{ user_query }}

Based on the query above, classify the user's intent into one of the following categories:

{{ intents_doc }}

Respond with your classification wrapped in the appropriate tags.
"#,
    extractor_tag = "intent"
)]
enum UserIntent {
    /// The user wants to search for information or find something specific
    SearchQuery,
    /// The user wants to create or generate new content
    CreateContent,
    /// The user needs help with a problem or wants assistance
    RequestHelp,
    /// The user wants to analyze or evaluate data
    AnalyzeData,
    /// The user is just greeting or making small talk
    Greeting,
}

// Implement FromStr for the enum (required by the IntentExtractor trait)
impl FromStr for UserIntent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SearchQuery" => Ok(UserIntent::SearchQuery),
            "CreateContent" => Ok(UserIntent::CreateContent),
            "RequestHelp" => Ok(UserIntent::RequestHelp),
            "AnalyzeData" => Ok(UserIntent::AnalyzeData),
            "Greeting" => Ok(UserIntent::Greeting),
            _ => Err(format!("Unknown UserIntent variant: {}", s)),
        }
    }
}

#[test]
fn test_generated_prompt_function() {
    // Test that the auto-generated build_user_intent_prompt function works correctly
    let prompt = build_user_intent_prompt("How can I learn Rust programming?");

    // Assert that the prompt contains the user query
    assert!(prompt.contains("How can I learn Rust programming?"));

    // Assert that the prompt contains the enum documentation
    assert!(prompt.contains("The user wants to search for information"));
    assert!(prompt.contains("SearchQuery"));
    assert!(prompt.contains("The user wants to create or generate new content"));
    assert!(prompt.contains("CreateContent"));
    assert!(prompt.contains("The user needs help with a problem"));
    assert!(prompt.contains("RequestHelp"));
    assert!(prompt.contains("Greeting"));

    // Assert that the prompt has the correct structure
    assert!(prompt.contains("User Query:"));
    assert!(prompt.contains("classify the user's intent"));
}

#[test]
fn test_generated_extractor_success() {
    // Test that the auto-generated UserIntentExtractor works correctly
    let extractor = UserIntentExtractor;

    // Test case 1: SearchQuery intent
    let mock_response = r#"
Based on the user's query about learning Rust programming, I can see they are looking for information and resources.

<intent>SearchQuery</intent>

This is clearly a search for educational information.
"#;

    let result = extractor.extract_intent(mock_response);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), UserIntent::SearchQuery);

    // Test case 2: CreateContent intent
    let mock_response_2 = r#"
The user wants to generate something new.
<intent>CreateContent</intent>
"#;

    let result_2 = extractor.extract_intent(mock_response_2);
    assert!(result_2.is_ok());
    assert_eq!(result_2.unwrap(), UserIntent::CreateContent);

    // Test case 3: RequestHelp intent
    let mock_response_3 = "<intent>RequestHelp</intent>";
    let result_3 = extractor.extract_intent(mock_response_3);
    assert!(result_3.is_ok());
    assert_eq!(result_3.unwrap(), UserIntent::RequestHelp);

    // Test case 4: Greeting intent
    let mock_response_4 =
        "Let me analyze this... <intent>Greeting</intent> Yes, this is just a greeting.";
    let result_4 = extractor.extract_intent(mock_response_4);
    assert!(result_4.is_ok());
    assert_eq!(result_4.unwrap(), UserIntent::Greeting);
}

#[test]
fn test_generated_extractor_failure() {
    let extractor = UserIntentExtractor;

    // Test case 1: Missing tag entirely
    let mock_response = "This response doesn't contain the required tag at all.";
    let result = extractor.extract_intent(mock_response);
    assert!(result.is_err());
    if let Err(err) = result {
        // Verify it's an IntentError related to missing tag
        match err {
            IntentError::TagNotFound { tag } => {
                assert_eq!(tag, "intent");
            }
            _ => panic!("Expected TagNotFound error, got: {:?}", err),
        }
    }

    // Test case 2: Wrong tag name
    let mock_response_2 = "<wrong_tag>SearchQuery</wrong_tag>";
    let result_2 = extractor.extract_intent(mock_response_2);
    assert!(result_2.is_err());

    // Test case 3: Invalid enum variant
    let mock_response_3 = "<intent>InvalidVariant</intent>";
    let result_3 = extractor.extract_intent(mock_response_3);
    assert!(result_3.is_err());
    if let Err(err) = result_3 {
        // This should fail when trying to parse the enum from string
        match err {
            IntentError::ParseFailed { .. } => {
                // Expected
            }
            _ => panic!("Expected ParseFailed error, got: {:?}", err),
        }
    }

    // Test case 4: Empty tag content
    let mock_response_4 = "<intent></intent>";
    let result_4 = extractor.extract_intent(mock_response_4);
    assert!(result_4.is_err());
}

#[test]
fn test_intent_frame_integration() {
    // Test that the generated intent works with IntentFrame
    let frame = IntentFrame::new("input", "intent");

    let mock_response =
        "Analysis complete. <intent>AnalyzeData</intent> The user wants to analyze data.";
    let result: Result<UserIntent, _> = frame.extract_intent(mock_response);

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), UserIntent::AnalyzeData);
}

#[test]
fn test_enum_variant_serialization() {
    // Test that the enum variants can be properly serialized/deserialized
    let intent = UserIntent::SearchQuery;

    // Test to JSON
    let json = serde_json::to_string(&intent).unwrap();
    assert_eq!(json, "\"SearchQuery\"");

    // Test from JSON
    let deserialized: UserIntent = serde_json::from_str("\"CreateContent\"").unwrap();
    assert_eq!(deserialized, UserIntent::CreateContent);
}

#[test]
fn test_enum_variant_from_str() {
    // Test that FromStr is implemented correctly for the enum
    assert_eq!(
        UserIntent::from_str("SearchQuery").unwrap(),
        UserIntent::SearchQuery
    );
    assert_eq!(
        UserIntent::from_str("CreateContent").unwrap(),
        UserIntent::CreateContent
    );
    assert_eq!(
        UserIntent::from_str("RequestHelp").unwrap(),
        UserIntent::RequestHelp
    );
    assert_eq!(
        UserIntent::from_str("AnalyzeData").unwrap(),
        UserIntent::AnalyzeData
    );
    assert_eq!(
        UserIntent::from_str("Greeting").unwrap(),
        UserIntent::Greeting
    );

    // Test invalid variant
    assert!(UserIntent::from_str("InvalidVariant").is_err());
}

#[test]
fn test_multiple_intents_in_same_module() {
    // Define another intent enum to ensure multiple can coexist
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[define_intent]
    #[intent(
        prompt = "Classify the sentiment: {{ text }}\n\n{{ intents_doc }}",
        extractor_tag = "sentiment"
    )]
    enum SentimentIntent {
        /// Positive sentiment
        Positive,
        /// Negative sentiment
        Negative,
        /// Neutral sentiment
        Neutral,
    }

    impl FromStr for SentimentIntent {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "Positive" => Ok(SentimentIntent::Positive),
                "Negative" => Ok(SentimentIntent::Negative),
                "Neutral" => Ok(SentimentIntent::Neutral),
                _ => Err(format!("Unknown SentimentIntent variant: {}", s)),
            }
        }
    }

    // Test the second intent's generated functions
    let prompt = build_sentiment_intent_prompt("This is amazing!");
    assert!(prompt.contains("This is amazing!"));
    assert!(prompt.contains("Positive sentiment"));

    let extractor = SentimentIntentExtractor;
    let response = "<sentiment>Positive</sentiment>";
    assert_eq!(
        extractor.extract_intent(response).unwrap(),
        SentimentIntent::Positive
    );
}
