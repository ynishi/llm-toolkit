//! `llm-toolkit` - A low-level Rust toolkit for the LLM last mile problem.
//!
//! This library provides a set of sharp, reliable, and unopinionated "tools"
//! for building robust LLM-powered applications in Rust. It focuses on solving
//! the common and frustrating problems that occur at the boundary between a
//! strongly-typed Rust application and the unstructured, often unpredictable
//! string-based responses from LLM APIs.

/// Extracts a JSON string from a raw LLM response string.
///
/// This function applies a series of strategies to find and extract a JSON object
/// from a string that may contain extraneous text, such as explanations or
/// Markdown code blocks.
///
/// # Strategies
///
/// 1.  **Markdown Code Block:** It first looks for a JSON object enclosed in a
///     Markdown code block (e.g., ` ```json ... ``` `).
/// 2.  **JSON Brackets:** If not found, it looks for the first `{` and the last `}`
///     and returns the content between them.
/// 3.  **Fallback:** If neither of the above strategies succeeds, it returns the
///     original trimmed string.
///
/// # Example
///
/// ````
/// use llm_toolkit::extract_json;
///
/// let response_with_markdown = concat!(
///     "Here is the JSON you requested:\n",
///     "```json\n",
///     "{\n",
///     "  \"name\": \"Elara\",\n",
///     "  \"value\": 42\n",
///     "}\n",
///     "```\n",
///     "I hope this helps!"
/// );
///
/// let extracted = extract_json(response_with_markdown);
/// assert_eq!(extracted, "{\n  \"name\": \"Elara\",\n  \"value\": 42\n}");
/// ````
pub fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // Strategy 1: Check for markdown JSON code block
    if let Some(start_idx) = trimmed.find("```json") {
        let json_start = start_idx + 7; // length of "```json"
        if let Some(end_idx) = trimmed[json_start..].find("```") {
            let extracted = &trimmed[json_start..json_start + end_idx].trim();
            // Always return what's inside the block, even if empty.
            return extracted;
        }
    }

    // Strategy 2: Find JSON object boundaries
    if let Some(first_brace) = trimmed.find('{') {
        if let Some(last_brace) = trimmed.rfind('}') {
            if last_brace >= first_brace {
                return &trimmed[first_brace..=last_brace];
            }
        }
    }

    // Strategy 3: Return original trimmed text as fallback
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_with_markdown() {
        let input = "Here is the JSON: ```json\n{\"key\": \"value\"}\n```";
        assert_eq!(extract_json(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_extract_with_json_brackets() {
        let input = "Some text before {\"key\": \"value\"} and after.";
        assert_eq!(extract_json(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_extract_pure_json() {
        let input = "  {\"key\": \"value\"}  ";
        assert_eq!(extract_json(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_no_json_returns_trimmed() {
        let input = "  Just some plain text.  ";
        assert_eq!(extract_json(input), "Just some plain text.");
    }

    #[test]
    fn test_empty_input() {
        let input = "";
        assert_eq!(extract_json(input), "");
    }

    #[test]
    fn test_empty_markdown_block() {
        let input = "```json\n```";
        assert_eq!(extract_json(input), "");
    }
}
