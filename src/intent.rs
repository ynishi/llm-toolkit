//! Traits and implementations for extracting structured intents from LLM responses.

use crate::extract::FlexibleExtractor;
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
pub struct PromptBasedExtractor {
    extractor: FlexibleExtractor,
    tag: String,
}

impl PromptBasedExtractor {
    /// Creates a new extractor that looks for the specified tag.
    pub fn new(tag: &str) -> Self {
        Self {
            extractor: FlexibleExtractor::new(),
            tag: tag.to_string(),
        }
    }
}

impl<T> IntentExtractor<T> for PromptBasedExtractor
where
    T: FromStr,
{
    fn extract_intent(&self, text: &str) -> Result<T, IntentError> {
        // 1. Use FlexibleExtractor to get the string inside the tag
        let extracted_str = self
            .extractor
            .extract_tagged(text, &self.tag)
            .ok_or_else(|| IntentError::TagNotFound {
                tag: self.tag.clone(),
            })?;

        // 2. Parse the string into the user's enum
        T::from_str(&extracted_str).map_err(|_| IntentError::ParseFailed {
            value: extracted_str.to_string(),
        })
    }
}
