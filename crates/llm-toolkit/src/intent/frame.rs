use crate::extract::FlexibleExtractor;
use crate::extract::core::ContentExtractor;
use crate::intent::{IntentError, IntentExtractor};
use std::str::FromStr;

/// A frame-based intent extractor that uses separate tags for input wrapping and extraction.
pub struct IntentFrame {
    input_tag: String,
    extractor_tag: String,
}

impl IntentFrame {
    /// Creates a new `IntentFrame` with specified input and extractor tags.
    pub fn new(input_tag: &str, extractor_tag: &str) -> Self {
        Self {
            input_tag: input_tag.to_string(),
            extractor_tag: extractor_tag.to_string(),
        }
    }

    /// Wraps the given text with the input tag.
    pub fn wrap(&self, text: &str) -> String {
        format!("<{0}>{1}</{0}>", self.input_tag, text)
    }
}

impl<T> IntentExtractor<T> for IntentFrame
where
    T: FromStr,
{
    fn extract_intent(&self, text: &str) -> Result<T, IntentError> {
        // Use FlexibleExtractor to get the string inside the extractor tag
        let extractor = FlexibleExtractor::new();
        let extracted_str = extractor
            .extract_tagged(text, &self.extractor_tag)
            .ok_or_else(|| IntentError::TagNotFound {
                tag: self.extractor_tag.clone(),
            })?;

        // Parse the string into the user's type
        T::from_str(&extracted_str).map_err(|_| IntentError::ParseFailed {
            value: extracted_str.to_string(),
        })
    }
}

#[cfg(test)]
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
    fn test_wrap_method() {
        let frame = IntentFrame::new("input", "output");
        let wrapped = frame.wrap("test content");
        assert_eq!(wrapped, "<input>test content</input>");
    }

    #[test]
    fn test_extract_intent_success() {
        let frame = IntentFrame::new("input", "intent");
        let text = "<intent>Login</intent>";
        let result: Result<TestIntent, _> = IntentExtractor::extract_intent(&frame, text);
        assert_eq!(result.unwrap(), TestIntent::Login);
    }

    #[test]
    fn test_extract_intent_tag_not_found() {
        let frame = IntentFrame::new("input", "intent");
        let text = "No intent tag here";
        let result: Result<TestIntent, _> = IntentExtractor::extract_intent(&frame, text);

        match result {
            Err(IntentError::TagNotFound { tag }) => {
                assert_eq!(tag, "intent");
            }
            _ => panic!("Expected TagNotFound error"),
        }
    }

    #[test]
    fn test_extract_intent_parse_failed() {
        let frame = IntentFrame::new("input", "intent");
        let text = "<intent>Invalid</intent>";
        let result: Result<TestIntent, _> = IntentExtractor::extract_intent(&frame, text);

        match result {
            Err(IntentError::ParseFailed { value }) => {
                assert_eq!(value, "Invalid");
            }
            _ => panic!("Expected ParseFailed error"),
        }
    }
}
