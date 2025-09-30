//! 'llm-toolkit' - A low-level Rust toolkit for the LLM last mile problem.
//!
//! This library provides a set of sharp, reliable, and unopinionated "tools"
//! for building robust LLM-powered applications in Rust. It focuses on solving
//! the common and frustrating problems that occur at the boundary between a
//! strongly-typed Rust application and the unstructured, often unpredictable
//! string-based responses from LLM APIs.

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
pub use orchestrator::{BlueprintWorkflow, Orchestrator, OrchestratorError, StrategyMap};

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
}
