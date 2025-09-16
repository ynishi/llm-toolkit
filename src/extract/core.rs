use super::error::ParseError;
use serde::{Deserialize, Serialize};

/// Core trait for response parsing
pub trait ResponseParser<T> {
    /// Parse response content into target type
    fn parse(&self, content: &str) -> Result<T, ParseError>;

    /// Extract content using configured strategies
    fn extract_content(&self, text: &str) -> String;

    /// Fallback parsing when primary parsing fails
    fn fallback_parse(&self, content: &str, error: &ParseError) -> Result<T, ParseError>;
}

/// Trait for extracting tagged or structured content
pub trait ContentExtractor {
    /// Extract content within specified tags
    fn extract_tagged(&self, text: &str, tag: &str) -> Option<String>;

    /// Extract JSON-like content
    fn extract_json_like(&self, text: &str) -> Option<String>;

    /// Extract using custom pattern
    fn extract_pattern(&self, text: &str, pattern: &str) -> Option<String>;
}

/// Extraction strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionStrategy {
    /// Extract content within XML-like tags: <tag>content</tag>
    TaggedContent(String),

    /// Extract content within JSON braces: {...}
    JsonBrackets,

    /// Find first complete JSON object
    FirstJsonObject,

    /// Search for specific keywords and determine type
    KeywordSearch(Vec<String>),

    /// Use regex pattern for extraction
    RegexPattern(String),

    /// Return original text as-is
    OriginalText,
}

/// Configuration for response parsing
#[derive(Debug, Clone)]
pub struct ParsingConfig {
    /// Primary tag to look for (e.g., "answer", "response")
    pub primary_tag: String,

    /// Ordered list of extraction strategies to try
    pub extraction_strategies: Vec<ExtractionStrategy>,

    /// Whether to enable debug logging
    pub debug_mode: bool,

    /// Maximum content length to process
    pub max_content_length: Option<usize>,
}

impl Default for ParsingConfig {
    fn default() -> Self {
        Self {
            primary_tag: "answer".to_string(),
            extraction_strategies: vec![
                ExtractionStrategy::TaggedContent("answer".to_string()),
                ExtractionStrategy::JsonBrackets,
                ExtractionStrategy::OriginalText,
            ],
            debug_mode: false,
            max_content_length: Some(50_000), // 50KB limit
        }
    }
}

impl ParsingConfig {
    /// Create new config with custom tag
    pub fn with_tag(tag: &str) -> Self {
        Self {
            primary_tag: tag.to_string(),
            extraction_strategies: vec![ExtractionStrategy::TaggedContent(tag.to_string())],
            ..Default::default()
        }
    }

    /// Add extraction strategy
    pub fn add_strategy(mut self, strategy: ExtractionStrategy) -> Self {
        self.extraction_strategies.push(strategy);
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self) -> Self {
        self.debug_mode = true;
        self
    }
}
