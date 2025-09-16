//! 'llm-toolkit' - A low-level Rust toolkit for the LLM last mile problem.
//!
//! This library provides a set of sharp, reliable, and unopinionated "tools"
//! for building robust LLM-powered applications in Rust. It focuses on solving
//! the common and frustrating problems that occur at the boundary between a
//! strongly-typed Rust application and the unstructured, often unpredictable
//! string-based responses from LLM APIs.

pub mod extract;
pub mod intent;
pub mod prompt;

pub use extract::FlexibleExtractor;
pub use intent::{IntentError, IntentExtractor, PromptBasedExtractor};
pub use prompt::ToPrompt;

use extract::ParseError;

/// Extracts a JSON string from a raw LLM response string.
///
/// This function uses a `FlexibleExtractor` with its standard strategies
/// to find and extract a JSON object from a string that may contain extraneous
/// text, such as explanations or Markdown code blocks.
///
/// For more advanced control over extraction strategies, see the `extract::FlexibleExtractor` struct.
///
/// # Returns
///
/// A `Result` containing the extracted JSON `String` on success, or a `ParseError`
/// if no JSON could be extracted.
pub fn extract_json(text: &str) -> Result<String, ParseError> {
    let extractor = FlexibleExtractor::new();
    // Note: The standard strategies in the copied code are TaggedContent("answer"), JsonBrackets, FirstJsonObject.
    // We will add a markdown strategy later during refactoring.
    extractor.extract(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_extraction() {
        let input = "Some text before {\"key\": \"value\"} and after.";
        assert_eq!(extract_json(input).unwrap(), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_standard_extraction_from_tagged_content() {
        let text = "<answer>{\"type\": \"success\"}</answer>";
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "{\"type\": \"success\"}");
    }
}
