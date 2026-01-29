use super::core::{ContentExtractor, ExtractionStrategy};

use super::error::ParseError;
use fuzzy_parser::sanitize_json;
use regex::Regex;

// Conditional debug logging macro
#[cfg(feature = "log")]
macro_rules! debug_log {
    ($($arg:tt)*) => { log::debug!($($arg)*) }
}

#[cfg(not(feature = "log"))]
macro_rules! debug_log {
    ($($arg:tt)*) => { }
}

/// Flexible content extractor with multiple strategies
pub struct FlexibleExtractor {
    debug_mode: bool,
}

impl FlexibleExtractor {
    pub fn new() -> Self {
        Self { debug_mode: false }
    }

    pub fn with_debug(mut self) -> Self {
        self.debug_mode = true;
        self
    }

    pub fn standard_extraction_strategies() -> Vec<ExtractionStrategy> {
        vec![
            ExtractionStrategy::TaggedContent("answer".to_string()),
            ExtractionStrategy::JsonBrackets,
            ExtractionStrategy::FirstJsonObject,
        ]
    }

    /// Standard extraction
    pub fn extract(&self, text: &str) -> Result<String, ParseError> {
        if self.debug_mode {
            debug_log!("Extracting content from text: {}", text);
        }
        self.extract_with_strategies(text, &Self::standard_extraction_strategies())
    }

    /// Extract content using specified strategy
    pub fn extract_with_strategy(
        &self,
        text: &str,
        strategy: &ExtractionStrategy,
    ) -> Option<String> {
        if self.debug_mode {
            debug_log!("Trying extraction strategy: {:?}", strategy);
        }

        match strategy {
            ExtractionStrategy::TaggedContent(tag) => self.extract_tagged(text, tag),
            ExtractionStrategy::JsonBrackets => self.extract_json_like(text),
            ExtractionStrategy::FirstJsonObject => self.extract_first_json_object(text),
            ExtractionStrategy::KeywordSearch(keywords) => self.extract_by_keywords(text, keywords),
            ExtractionStrategy::RegexPattern(pattern) => self.extract_pattern(text, pattern),
            ExtractionStrategy::OriginalText => Some(text.to_string()),
        }
    }

    /// Try multiple extraction strategies in order
    pub fn extract_with_strategies(
        &self,
        text: &str,
        strategies: &[ExtractionStrategy],
    ) -> Result<String, ParseError> {
        let mut errors = Vec::new();

        for strategy in strategies {
            if let Some(result) = self.extract_with_strategy(text, strategy) {
                if self.debug_mode {
                    debug_log!("Successfully extracted with strategy: {:?}", strategy);
                }
                return Ok(result);
            } else {
                errors.push(format!("Strategy {:?} failed", strategy));
            }
        }

        Err(ParseError::AllStrategiesFailed(errors))
    }

    /// Extract first complete JSON entity (object or array) from text
    fn extract_first_json_entity(&self, text: &str) -> Option<String> {
        let mut bracket_count = 0;
        let mut start_pos = None;
        let mut in_string = false;
        let mut escape_next = false;
        let mut opening_char = None;

        for (i, ch) in text.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => {
                    if bracket_count == 0 {
                        start_pos = Some(i);
                        opening_char = Some(ch);
                    }
                    bracket_count += 1;
                }
                '}' | ']' if !in_string => {
                    bracket_count -= 1;
                    if bracket_count == 0
                        && let Some(p) = start_pos
                        && let Some(opening) = opening_char
                    {
                        // Verify matching brackets
                        let is_valid =
                            (opening == '{' && ch == '}') || (opening == '[' && ch == ']');
                        if is_valid {
                            return Some(text[p..=i].to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Extract first complete JSON object from text
    fn extract_first_json_object(&self, text: &str) -> Option<String> {
        self.extract_first_json_entity(text)
            .map(|json| sanitize_json(&json))
    }

    /// Extract content based on keyword matching
    fn extract_by_keywords(&self, text: &str, keywords: &[String]) -> Option<String> {
        let lower_text = text.to_lowercase();

        for keyword in keywords {
            if lower_text.contains(&keyword.to_lowercase()) {
                // Return the keyword as the extracted content
                return Some(keyword.clone());
            }
        }

        None
    }
}

impl Default for FlexibleExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentExtractor for FlexibleExtractor {
    fn extract_tagged(&self, text: &str, tag: &str) -> Option<String> {
        // Create regex pattern for XML-like tags
        let pattern = format!(r"(?s)<{tag}>(.*?)</{tag}>", tag = regex::escape(tag));

        if let Ok(regex) = Regex::new(&pattern)
            && let Some(captures) = regex.captures(text)
            && let Some(content) = captures.get(1)
        {
            return Some(content.as_str().trim().to_string());
        }

        if self.debug_mode {
            debug_log!("Failed to extract tagged content with tag: {}", tag);
        }

        None
    }

    fn extract_json_like(&self, text: &str) -> Option<String> {
        // Delegate to extract_first_json_entity for proper handling
        let result = self
            .extract_first_json_entity(text)
            .map(|json| sanitize_json(&json));

        if result.is_none() && self.debug_mode {
            debug_log!("Failed to extract JSON-like content");
        }

        result
    }

    fn extract_pattern(&self, text: &str, pattern: &str) -> Option<String> {
        if let Ok(regex) = Regex::new(pattern)
            && let Some(captures) = regex.captures(text)
        {
            // Return the first capture group, or the whole match if no groups
            if captures.len() > 1 {
                return captures.get(1).map(|m| m.as_str().to_string());
            } else {
                return captures.get(0).map(|m| m.as_str().to_string());
            }
        }

        if self.debug_mode {
            debug_log!("Failed to extract with pattern: {}", pattern);
        }

        None
    }
}

/// Extractor for Markdown code blocks
pub struct MarkdownCodeBlockExtractor {
    /// Optional language to filter by (e.g., "rust", "python")
    pub language: Option<String>,
}

impl Default for MarkdownCodeBlockExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownCodeBlockExtractor {
    /// Create a new extractor for any code block
    pub fn new() -> Self {
        Self { language: None }
    }

    /// Create a new extractor for a specific language
    pub fn with_language(language: String) -> Self {
        Self {
            language: Some(language),
        }
    }

    /// Extract content from a markdown code block
    pub fn extract(&self, text: &str) -> Result<String, ParseError> {
        let pattern = if let Some(ref lang) = self.language {
            // Match code block with specific language
            format!(
                r"(?m)^\s*```\s*{}\s*\n((?:.*\n)*?)^\s*```\s*$",
                regex::escape(lang)
            )
        } else {
            // Match any code block (with or without language specifier)
            r"(?m)^\s*```[^\n]*\n((?:.*\n)*?)^\s*```\s*$".to_string()
        };

        let regex = Regex::new(&pattern)
            .map_err(|e| ParseError::InvalidFormat(format!("Failed to compile regex: {}", e)))?;

        if let Some(captures) = regex.captures(text)
            && let Some(content) = captures.get(1)
        {
            // Trim surrounding newlines but preserve internal formatting
            let extracted = content.as_str().trim_end();
            return Ok(extracted.to_string());
        }

        Err(ParseError::TagExtractionFailed(format!(
            "No markdown code block found{}",
            if let Some(ref lang) = self.language {
                format!(" with language '{}'", lang)
            } else {
                String::new()
            }
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tagged_content() {
        let extractor = FlexibleExtractor::new();

        let text = "<answer>Hello World</answer>";
        let result = extractor.extract_tagged(text, "answer");
        assert_eq!(result, Some("Hello World".to_string()));

        let text_with_whitespace = "<answer>\n  Hello World  \n</answer>";
        let result = extractor.extract_tagged(text_with_whitespace, "answer");
        assert_eq!(result, Some("Hello World".to_string()));
    }

    #[test]
    fn test_extract_json_like() {
        let extractor = FlexibleExtractor::new();

        let text = "Here is some JSON: {\"key\": \"value\"} and more text";
        let result = extractor.extract_json_like(text);
        assert_eq!(result, Some("{\"key\": \"value\"}".to_string()));
    }

    #[test]
    fn test_extract_first_json_object() {
        let extractor = FlexibleExtractor::new();

        let text = "Some text {\"first\": \"object\"} more text {\"second\": \"object\"}";
        let result = extractor.extract_first_json_object(text);
        assert_eq!(result, Some("{\"first\": \"object\"}".to_string()));
    }

    #[test]
    fn test_extract_json_array() {
        let extractor = FlexibleExtractor::new();

        let text = "Here is an array: [{\"key\": \"value\"}] and more text";
        let result = extractor.extract_first_json_object(text);
        assert_eq!(result, Some("[{\"key\": \"value\"}]".to_string()));

        // Test via extract_json_like as well
        let result2 = extractor.extract_json_like(text);
        assert_eq!(result2, Some("[{\"key\": \"value\"}]".to_string()));
    }

    #[test]
    fn test_extract_by_keywords() {
        let extractor = FlexibleExtractor::new();
        let keywords = vec!["Comfort".to_string(), "Debug".to_string()];

        let text = "This is about comfort and support";
        let result = extractor.extract_by_keywords(text, &keywords);
        assert_eq!(result, Some("Comfort".to_string()));
    }

    #[test]
    fn test_extraction_strategies() {
        let extractor = FlexibleExtractor::new();

        let strategies = vec![
            ExtractionStrategy::TaggedContent("answer".to_string()),
            ExtractionStrategy::JsonBrackets,
            ExtractionStrategy::OriginalText,
        ];

        let text = "<answer>{\"type\": \"success\"}</answer>";
        let result = extractor.extract_with_strategies(text, &strategies);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "{\"type\": \"success\"}");
    }

    #[test]
    fn test_clean_json_trailing_commas_object() {
        let extractor = FlexibleExtractor::new();

        // Test trailing comma in object
        let text = r#"{"name": "Alice", "age": 30,}"#;
        let result = extractor.extract_first_json_object(text);
        assert_eq!(result, Some(r#"{"name": "Alice", "age": 30}"#.to_string()));

        // Test trailing comma with whitespace
        let text2 = r#"{"name": "Bob", "age": 25, }"#;
        let result2 = extractor.extract_first_json_object(text2);
        assert_eq!(result2, Some(r#"{"name": "Bob", "age": 25 }"#.to_string()));
    }

    #[test]
    fn test_clean_json_trailing_commas_array() {
        let extractor = FlexibleExtractor::new();

        // Test trailing comma in array
        let text = r#"["apple", "banana", "cherry",]"#;
        let result = extractor.extract_first_json_object(text);
        assert_eq!(result, Some(r#"["apple", "banana", "cherry"]"#.to_string()));

        // Test trailing comma with whitespace
        let text2 = r#"[1, 2, 3, ]"#;
        let result2 = extractor.extract_first_json_object(text2);
        assert_eq!(result2, Some(r#"[1, 2, 3 ]"#.to_string()));
    }

    #[test]
    fn test_clean_json_trailing_commas_nested() {
        let extractor = FlexibleExtractor::new();

        // Test nested structures with trailing commas
        let text = r#"{"items": [{"a": 1,}, {"b": 2,},], "count": 2,}"#;
        let result = extractor.extract_first_json_object(text);
        assert_eq!(
            result,
            Some(r#"{"items": [{"a": 1}, {"b": 2}], "count": 2}"#.to_string())
        );
    }

    #[test]
    fn test_clean_json_preserves_commas_in_strings() {
        let extractor = FlexibleExtractor::new();

        // Commas inside strings should be preserved
        let text = r#"{"message": "Hello, world", "items": "a, b, c"}"#;
        let result = extractor.extract_first_json_object(text);
        // The commas in strings should remain
        assert_eq!(
            result,
            Some(r#"{"message": "Hello, world", "items": "a, b, c"}"#.to_string())
        );

        // Test with trailing comma but commas in string values
        let text2 = r#"{"msg": "test, data", "val": 1,}"#;
        let result2 = extractor.extract_first_json_object(text2);
        assert_eq!(
            result2,
            Some(r#"{"msg": "test, data", "val": 1}"#.to_string())
        );
    }

    #[test]
    fn test_clean_json_valid_json_unchanged() {
        let extractor = FlexibleExtractor::new();

        // Valid JSON without trailing commas should remain unchanged
        let text = r#"{"name": "Alice", "age": 30}"#;
        let result = extractor.extract_first_json_object(text);
        assert_eq!(result, Some(text.to_string()));

        let text2 = r#"["a", "b", "c"]"#;
        let result2 = extractor.extract_first_json_object(text2);
        assert_eq!(result2, Some(text2.to_string()));
    }

    #[test]
    fn test_extract_json_like_with_trailing_commas() {
        let extractor = FlexibleExtractor::new();

        // extract_json_like should also clean trailing commas
        let text = "Here's the data: {\"result\": \"success\", \"code\": 200,}";
        let result = extractor.extract_json_like(text);
        assert_eq!(
            result,
            Some(r#"{"result": "success", "code": 200}"#.to_string())
        );
    }
}
