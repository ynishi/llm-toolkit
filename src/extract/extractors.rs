use super::core::{ContentExtractor, ExtractionStrategy};

use super::error::ParseError;
use regex::Regex;
use tracing::debug;

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
            debug!("Extracting content from text: {}", text);
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
            debug!("Trying extraction strategy: {:?}", strategy);
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
                    debug!("Successfully extracted with strategy: {:?}", strategy);
                }
                return Ok(result);
            } else {
                errors.push(format!("Strategy {:?} failed", strategy));
            }
        }
        
        Err(ParseError::AllStrategiesFailed(errors))
    }

    /// Extract first complete JSON object from text
    fn extract_first_json_object(&self, text: &str) -> Option<String> {
        let mut brace_count = 0;
        let mut start_pos = None;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in text.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' if !in_string => {
                    if brace_count == 0 {
                        start_pos = Some(i);
                    }
                    brace_count += 1;
                }
                '}' if !in_string => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        if let Some(p) = start_pos {
                            return Some(text[p..=i].to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        None
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
        let pattern = format!(
            r"(?s)<{tag}>(.*?)</{tag}>",
            tag = regex::escape(tag)
        );

        if let Ok(regex) = Regex::new(&pattern) {
            if let Some(captures) = regex.captures(text) {
                if let Some(content) = captures.get(1) {
                    return Some(content.as_str().trim().to_string());
                }
            }
        }

        if self.debug_mode {
            debug!("Failed to extract tagged content with tag: {}", tag);
        }

        None
    }

    fn extract_json_like(&self, text: &str) -> Option<String> {
        // Find JSON-like content within braces
        if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                if end > start {
                    return Some(text[start..=end].to_string());
                }
            }
        }

        if self.debug_mode {
            debug!("Failed to extract JSON-like content");
        }

        None
    }

    fn extract_pattern(&self, text: &str, pattern: &str) -> Option<String> {
        if let Ok(regex) = Regex::new(pattern) {
            if let Some(captures) = regex.captures(text) {
                // Return the first capture group, or the whole match if no groups
                if captures.len() > 1 {
                    return captures.get(1).map(|m| m.as_str().to_string());
                } else {
                    return captures.get(0).map(|m| m.as_str().to_string());
                }
            }
        }

        if self.debug_mode {
            debug!("Failed to extract with pattern: {}", pattern);
        }

        None
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
}