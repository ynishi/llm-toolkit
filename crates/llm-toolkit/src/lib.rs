//! 'llm-toolkit' - A low-level Rust toolkit for the LLM last mile problem.
//!
//! This library provides a set of sharp, reliable, and unopinionated "tools"
//! for building robust LLM-powered applications in Rust. It focuses on solving
//! the common and frustrating problems that occur at the boundary between a
//! strongly-typed Rust application and the unstructured, often unpredictable
//! string-based responses from LLM APIs.

// Allow the crate to reference itself by name, which is needed for proc macros
// to work correctly in examples, tests, and bins
extern crate self as llm_toolkit;

/// A derive macro to implement the `ToPrompt` trait for structs.
///
/// This macro is available only when the `derive` feature is enabled.
/// See the [crate-level documentation](index.html#2-structured-prompts-with-derivetoprompt) for usage examples.
#[cfg(feature = "derive")]
pub use llm_toolkit_macros::ToPrompt;

/// A derive macro to implement the `ToPromptSet` trait for structs.
///
/// This macro is available only when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use llm_toolkit_macros::ToPromptSet;

/// A derive macro to implement the `ToPromptFor` trait for structs.
///
/// This macro is available only when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use llm_toolkit_macros::ToPromptFor;

/// A macro for creating examples sections in prompts.
///
/// This macro is available only when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use llm_toolkit_macros::examples_section;

/// A procedural attribute macro for defining intent enums with automatic prompt and extractor generation.
///
/// This macro is available only when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use llm_toolkit_macros::define_intent;

/// A derive macro to implement the `Agent` trait for structs.
///
/// This macro is available only when the `agent` feature is enabled.
/// It automatically generates an Agent implementation that uses ClaudeCodeAgent
/// internally and deserializes responses into a structured output type.
///
/// # Example
///
/// ```ignore
/// use llm_toolkit_macros::Agent;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct MyOutput {
///     result: String,
/// }
///
/// #[derive(Agent)]
/// #[agent(expertise = "My expertise", output = "MyOutput")]
/// struct MyAgent;
/// ```
#[cfg(feature = "agent")]
pub use llm_toolkit_macros::Agent;

/// An attribute macro to define agent structs with automatic trait implementations.
///
/// This macro is available only when the `agent` feature is enabled.
#[cfg(feature = "agent")]
pub use llm_toolkit_macros::agent;

/// A derive macro to implement the `TypeMarker` trait for structs.
///
/// This macro is available only when the `agent` feature is enabled.
/// It automatically generates a TypeMarker implementation that provides
/// a type identifier string for type-based orchestrator output retrieval.
///
/// # Example
///
/// ```ignore
/// use llm_toolkit::TypeMarker;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize, TypeMarker)]
/// struct MyResponse {
///     #[serde(default = "default_type")]
///     __type: String,
///     result: String,
/// }
///
/// fn default_type() -> String {
///     "MyResponse".to_string()
/// }
/// ```
#[cfg(feature = "agent")]
pub use llm_toolkit_macros::{TypeMarker, type_marker};

pub mod extract;
pub mod intent;
pub mod multimodal;
pub mod prompt;

#[cfg(feature = "agent")]
pub mod agent;

#[cfg(feature = "agent")]
pub mod orchestrator;

pub use extract::{FlexibleExtractor, MarkdownCodeBlockExtractor};
pub use intent::frame::IntentFrame;
#[allow(deprecated)]
pub use intent::{IntentError, IntentExtractor, PromptBasedExtractor};
pub use multimodal::ImageData;
pub use prompt::{PromptPart, PromptSetError, ToPrompt, ToPromptFor, ToPromptSet};

#[cfg(feature = "agent")]
pub use agent::{Agent, AgentError};

#[cfg(feature = "agent")]
pub use orchestrator::{
    BlueprintWorkflow, Orchestrator, OrchestratorError, StrategyMap, TypeMarker,
};

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
    // Try markdown code block first (common LLM output format)
    if let Ok(content) = extract_markdown_block_with_lang(text, "json") {
        return Ok(content);
    }

    // Also try generic markdown block (might contain JSON without language hint)
    if let Ok(content) = extract_markdown_block(text) {
        // Verify it's actually JSON by trying to extract JSON from it
        let extractor = FlexibleExtractor::new();
        if let Ok(json) = extractor.extract(&content) {
            return Ok(json);
        }
    }

    // Fall back to standard extraction strategies
    let extractor = FlexibleExtractor::new();
    extractor.extract(text)
}

/// Extracts content from any Markdown code block in the text.
///
/// This function searches for the first code block (delimited by triple backticks)
/// and returns its content. The code block can have any language specifier or none at all.
///
/// # Returns
///
/// A `Result` containing the extracted code block content on success, or a `ParseError`
/// if no code block is found.
pub fn extract_markdown_block(text: &str) -> Result<String, ParseError> {
    let extractor = MarkdownCodeBlockExtractor::new();
    extractor.extract(text)
}

/// Extracts content from a Markdown code block with a specific language.
///
/// This function searches for a code block with the specified language hint
/// (e.g., ```rust, ```python) and returns its content.
///
/// # Arguments
///
/// * `text` - The text containing the markdown code block
/// * `lang` - The language specifier to match (e.g., "rust", "python")
///
/// # Returns
///
/// A `Result` containing the extracted code block content on success, or a `ParseError`
/// if no code block with the specified language is found.
pub fn extract_markdown_block_with_lang(text: &str, lang: &str) -> Result<String, ParseError> {
    let extractor = MarkdownCodeBlockExtractor::with_language(lang.to_string());
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

    #[test]
    fn test_markdown_extraction() {
        // Test simple code block with no language
        let text1 = "Here is some code:\n```\nlet x = 42;\n```\nAnd some text after.";
        let result1 = extract_markdown_block(text1);
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), "let x = 42;");

        // Test code block with specific language (rust)
        let text2 = "Here's Rust code:\n```rust\nfn main() {
    println!(\"Hello\");
}
```";
        let result2 = extract_markdown_block_with_lang(text2, "rust");
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), "fn main() {\n    println!(\"Hello\");\n}");

        // Test extracting rust block when json block is also present
        let text3 = r#"\nFirst a JSON block:
```json
{"key": "value"}
```

Then a Rust block:
```rust
let data = vec![1, 2, 3];
```
"#;
        let result3 = extract_markdown_block_with_lang(text3, "rust");
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap(), "let data = vec![1, 2, 3];");

        // Test case where no code block is found
        let text4 = "This text has no code blocks at all.";
        let result4 = extract_markdown_block(text4);
        assert!(result4.is_err());

        // Test with messy surrounding text and newlines
        let text5 = r#"\nLots of text before...


   ```python
def hello():
    print("world")
    return True
   ```


And more text after with various spacing.
"#;
        let result5 = extract_markdown_block_with_lang(text5, "python");
        assert!(result5.is_ok());
        assert_eq!(
            result5.unwrap(),
            "def hello():\n    print(\"world\")\n    return True"
        );
    }

    #[test]
    fn test_extract_json_from_json_markdown_block() {
        // Test extraction from JSON markdown block (highest priority)
        let text = r#"Here's the response:
```json
{"status": "success", "count": 42}
```
That's the data you requested."#;
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"status": "success", "count": 42}"#);
    }

    #[test]
    fn test_extract_json_from_generic_markdown_block() {
        // Test extraction from generic markdown block containing JSON
        let text = r#"The output is:
```
{"result": "ok", "value": 123}
```
End of output."#;
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"result": "ok", "value": 123}"#);
    }

    #[test]
    fn test_extract_json_priority_json_block_over_inline() {
        // When both JSON markdown block and inline JSON exist, JSON block should be preferred
        let text = r#"Some inline {"inline": "data"} here.
```json
{"block": "data"}
```
More text."#;
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"block": "data"}"#);
    }

    #[test]
    fn test_extract_json_priority_json_block_over_generic_block() {
        // JSON markdown block should be preferred over generic block
        let text = r#"First a generic block:
```
{"generic": "block"}
```

Then a JSON block:
```json
{"json": "block"}
```"#;
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"json": "block"}"#);
    }

    #[test]
    fn test_extract_json_fallback_from_non_json_markdown_block() {
        // When markdown block contains non-JSON, fallback to inline extraction
        let text = r#"Here's some code:
```
This is not JSON at all
```
But this is JSON: {"fallback": "value"}"#;
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"fallback": "value"}"#);
    }

    #[test]
    fn test_extract_json_from_rust_block_fallback() {
        // When only non-JSON markdown blocks exist, fallback to inline extraction
        let text = r#"```rust
let x = 42;
```
The result is {"data": "inline"}"#;
        let result = extract_json(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"data": "inline"}"#);
    }

    #[test]
    fn test_extract_json_multiline_in_markdown_block() {
        // Test extraction of multiline JSON from markdown block
        let text = r#"Response:
```json
{
  "name": "test",
  "values": [1, 2, 3],
  "nested": {
    "key": "value"
  }
}
```"#;
        let result = extract_json(text);
        assert!(result.is_ok());
        let json = result.unwrap();
        // Verify it contains the expected structure
        assert!(json.contains("\"name\": \"test\""));
        assert!(json.contains("\"values\": [1, 2, 3]"));
        assert!(json.contains("\"nested\""));
    }
}
