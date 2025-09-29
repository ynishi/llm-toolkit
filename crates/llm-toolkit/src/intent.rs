//! Traits and implementations for extracting structured intents from LLM responses.

pub mod frame;

use self::frame::IntentFrame;
use std::str::FromStr;
use thiserror::Error;

/// An error type for intent extraction failures.
#[derive(Debug, Error)]
pub enum IntentError {
    #[error("Extraction failed: Tag '{tag}' not found in response")]
    TagNotFound { tag: String },

    #[error("Parsing failed: Could not parse '{value}' into a valid intent")]
    ParseFailed { value: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// A generic trait for extracting a structured intent of type `T` from a string response.
///
/// Type `T` is typically an enum representing the possible intents.
pub trait IntentExtractor<T>
where
    T: FromStr,
{
    /// Extracts and parses an intent from the given text.
    fn extract_intent(&self, text: &str) -> Result<T, IntentError>;
}

/// A classic, prompt-based implementation of `IntentExtractor`.
///
/// This extractor uses `FlexibleExtractor` to find content within a specific
/// XML-like tag (e.g., `<intent>...<intent>`) and then parses that content
/// into the target intent type `T`.
#[deprecated(
    since = "0.8.0",
    note = "Please use `IntentFrame` instead for better safety and clarity."
)]
pub struct PromptBasedExtractor {
    frame: IntentFrame,
}

#[allow(deprecated)]
impl PromptBasedExtractor {
    /// Creates a new extractor that looks for the specified tag.
    pub fn new(tag: &str) -> Self {
        Self {
            frame: IntentFrame::new(tag, tag),
        }
    }
}

#[allow(deprecated)]
impl<T> IntentExtractor<T> for PromptBasedExtractor
where
    T: FromStr,
{
    fn extract_intent(&self, text: &str) -> Result<T, IntentError> {
        self.frame.extract_intent(text)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    enum TestIntent {
        Login,
        Logout,
    }

    impl FromStr for TestIntent {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "Login" => Ok(TestIntent::Login),
                "Logout" => Ok(TestIntent::Logout),
                _ => Err(format!("Unknown intent: {}", s)),
            }
        }
    }

    #[test]
    fn test_extract_intent_success() {
        let extractor = PromptBasedExtractor::new("intent");
        let text = "<intent>Login</intent>";
        let result: Result<TestIntent, _> = IntentExtractor::extract_intent(&extractor, text);
        assert_eq!(result.unwrap(), TestIntent::Login);
    }

    #[test]
    fn test_extract_intent_tag_not_found() {
        let extractor = PromptBasedExtractor::new("intent");
        let text = "No intent tag here";
        let result: Result<TestIntent, _> = IntentExtractor::extract_intent(&extractor, text);

        match result {
            Err(IntentError::TagNotFound { tag }) => {
                assert_eq!(tag, "intent");
            }
            _ => panic!("Expected TagNotFound error"),
        }
    }

    #[test]
    fn test_extract_intent_parse_failed() {
        let extractor = PromptBasedExtractor::new("intent");
        let text = "<intent>Invalid</intent>";
        let result: Result<TestIntent, _> = IntentExtractor::extract_intent(&extractor, text);

        match result {
            Err(IntentError::ParseFailed { value }) => {
                assert_eq!(value, "Invalid");
            }
            _ => panic!("Expected ParseFailed error"),
        }
    }
}
