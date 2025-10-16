//! A trait and macros for powerful, type-safe prompt generation.

use minijinja::Environment;
use serde::Serialize;

/// Represents a part of a multimodal prompt.
///
/// This enum allows prompts to contain different types of content,
/// such as text and images, enabling multimodal LLM interactions.
#[derive(Debug, Clone)]
pub enum PromptPart {
    /// Text content in the prompt.
    Text(String),
    /// Image content with media type and binary data.
    Image {
        /// The MIME media type (e.g., "image/jpeg", "image/png").
        media_type: String,
        /// The raw image data.
        data: Vec<u8>,
    },
    // Future variants like Audio or Video can be added here
}

/// A trait for converting any type into a string suitable for an LLM prompt.
///
/// This trait provides a standard interface for converting various types
/// into strings that can be used as prompts for language models.
///
/// # Example
///
/// ```
/// use llm_toolkit::prompt::ToPrompt;
///
/// // Common types have ToPrompt implementations
/// let number = 42;
/// assert_eq!(number.to_prompt(), "42");
///
/// let text = "Hello, LLM!";
/// assert_eq!(text.to_prompt(), "Hello, LLM!");
/// ```
///
/// # Custom Implementation
///
/// You can also implement `ToPrompt` directly for your own types:
///
/// ```
/// use llm_toolkit::prompt::{ToPrompt, PromptPart};
/// use std::fmt;
///
/// struct CustomType {
///     value: String,
/// }
///
/// impl fmt::Display for CustomType {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "{}", self.value)
///     }
/// }
///
/// // By implementing ToPrompt directly, you can control the conversion.
/// impl ToPrompt for CustomType {
///     fn to_prompt_parts(&self) -> Vec<PromptPart> {
///         vec![PromptPart::Text(self.to_string())]
///     }
///
///     fn to_prompt(&self) -> String {
///         self.to_string()
///     }
/// }
///
/// let custom = CustomType { value: "custom".to_string() };
/// assert_eq!(custom.to_prompt(), "custom");
/// ```
pub trait ToPrompt {
    /// Converts the object into a vector of `PromptPart`s based on a mode.
    ///
    /// This is the core method that `derive(ToPrompt)` will implement.
    /// The `mode` argument allows for different prompt representations, such as:
    /// - "full": A comprehensive prompt with schema and examples.
    /// - "schema_only": Just the data structure's schema.
    /// - "example_only": Just a concrete example.
    ///
    /// The default implementation ignores the mode and calls `to_prompt_parts`
    /// for backward compatibility with manual implementations.
    fn to_prompt_parts_with_mode(&self, mode: &str) -> Vec<PromptPart> {
        // Default implementation for backward compatibility
        let _ = mode; // Unused in default impl
        self.to_prompt_parts()
    }

    /// Converts the object into a prompt string based on a mode.
    ///
    /// This method extracts only the text portions from `to_prompt_parts_with_mode()`.
    fn to_prompt_with_mode(&self, mode: &str) -> String {
        self.to_prompt_parts_with_mode(mode)
            .iter()
            .filter_map(|part| match part {
                PromptPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Converts the object into a vector of `PromptPart`s using the default "full" mode.
    ///
    /// This method enables multimodal prompt generation by returning
    /// a collection of prompt parts that can include text, images, and
    /// other media types.
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        self.to_prompt_parts_with_mode("full")
    }

    /// Converts the object into a prompt string using the default "full" mode.
    ///
    /// This method provides backward compatibility by extracting only
    /// the text portions from `to_prompt_parts()` and joining them.
    fn to_prompt(&self) -> String {
        self.to_prompt_with_mode("full")
    }

    /// Returns a schema-level prompt for the type itself.
    ///
    /// For enums, this returns all possible variants with their descriptions.
    /// For structs, this returns the field schema.
    ///
    /// Unlike instance methods like `to_prompt()`, this is a type-level method
    /// that doesn't require an instance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Enum: get all variants
    /// let schema = MyEnum::prompt_schema();
    ///
    /// // Struct: get field schema
    /// let schema = MyStruct::prompt_schema();
    /// ```
    fn prompt_schema() -> String {
        String::new() // Default implementation returns empty string
    }
}

// Add implementations for common types

impl ToPrompt for String {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(self.clone())]
    }

    fn to_prompt(&self) -> String {
        self.clone()
    }
}

impl ToPrompt for &str {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(self.to_string())]
    }

    fn to_prompt(&self) -> String {
        self.to_string()
    }
}

impl ToPrompt for bool {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(self.to_string())]
    }

    fn to_prompt(&self) -> String {
        self.to_string()
    }
}

impl ToPrompt for char {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(self.to_string())]
    }

    fn to_prompt(&self) -> String {
        self.to_string()
    }
}

macro_rules! impl_to_prompt_for_numbers {
    ($($t:ty),*) => {
        $(
            impl ToPrompt for $t {
                fn to_prompt_parts(&self) -> Vec<PromptPart> {
                    vec![PromptPart::Text(self.to_string())]
                }

                fn to_prompt(&self) -> String {
                    self.to_string()
                }
            }
        )*
    };
}

impl_to_prompt_for_numbers!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);

// Implement ToPrompt for Vec<T> where T: ToPrompt
impl<T: ToPrompt> ToPrompt for Vec<T> {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(self.to_prompt())]
    }

    fn to_prompt(&self) -> String {
        format!(
            "[{}]",
            self.iter()
                .map(|item| item.to_prompt())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Renders a prompt from a template string and a serializable context.
///
/// This is the underlying function for the `prompt!` macro.
pub fn render_prompt<T: Serialize>(template: &str, context: T) -> Result<String, minijinja::Error> {
    let mut env = Environment::new();
    env.add_template("prompt", template)?;
    let tmpl = env.get_template("prompt")?;
    tmpl.render(context)
}

/// Creates a prompt string from a template and key-value pairs.
///
/// This macro provides a `println!`-like experience for building prompts
/// from various data sources. It leverages `minijinja` for templating.
///
/// # Example
///
/// ```
/// use llm_toolkit::prompt;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct User {
///     name: &'static str,
///     role: &'static str,
/// }
///
/// let user = User { name: "Mai", role: "UX Engineer" };
/// let task = "designing a new macro";
///
/// let p = prompt!(
///     "User {{user.name}} ({{user.role}}) is currently {{task}}.",
///     user = user,
///     task = task
/// ).unwrap();
///
/// assert_eq!(p, "User Mai (UX Engineer) is currently designing a new macro.");
/// ```
#[macro_export]
macro_rules! prompt {
    ($template:expr, $($key:ident = $value:expr),* $(,)?) => {
        $crate::prompt::render_prompt($template, minijinja::context!($($key => $value),*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::fmt::Display;

    enum TestEnum {
        VariantA,
        VariantB,
    }

    impl Display for TestEnum {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestEnum::VariantA => write!(f, "Variant A"),
                TestEnum::VariantB => write!(f, "Variant B"),
            }
        }
    }

    impl ToPrompt for TestEnum {
        fn to_prompt_parts(&self) -> Vec<PromptPart> {
            vec![PromptPart::Text(self.to_string())]
        }

        fn to_prompt(&self) -> String {
            self.to_string()
        }
    }

    #[test]
    fn test_to_prompt_for_enum() {
        let variant = TestEnum::VariantA;
        assert_eq!(variant.to_prompt(), "Variant A");
    }

    #[test]
    fn test_to_prompt_for_enum_variant_b() {
        let variant = TestEnum::VariantB;
        assert_eq!(variant.to_prompt(), "Variant B");
    }

    #[test]
    fn test_to_prompt_for_string() {
        let s = "hello world";
        assert_eq!(s.to_prompt(), "hello world");
    }

    #[test]
    fn test_to_prompt_for_number() {
        let n = 42;
        assert_eq!(n.to_prompt(), "42");
    }

    #[derive(Serialize)]
    struct SystemInfo {
        version: &'static str,
        os: &'static str,
    }

    #[test]
    fn test_prompt_macro_simple() {
        let user = "Yui";
        let task = "implementation";
        let prompt = prompt!(
            "User {{user}} is working on the {{task}}.",
            user = user,
            task = task
        )
        .unwrap();
        assert_eq!(prompt, "User Yui is working on the implementation.");
    }

    #[test]
    fn test_prompt_macro_with_struct() {
        let sys = SystemInfo {
            version: "0.1.0",
            os: "Rust",
        };
        let prompt = prompt!("System: {{sys.version}} on {{sys.os}}", sys = sys).unwrap();
        assert_eq!(prompt, "System: 0.1.0 on Rust");
    }

    #[test]
    fn test_prompt_macro_mixed() {
        let user = "Mai";
        let sys = SystemInfo {
            version: "0.1.0",
            os: "Rust",
        };
        let prompt = prompt!(
            "User {{user}} is using {{sys.os}} v{{sys.version}}.",
            user = user,
            sys = sys
        )
        .unwrap();
        assert_eq!(prompt, "User Mai is using Rust v0.1.0.");
    }

    #[test]
    fn test_to_prompt_for_vec_of_strings() {
        let items = vec!["apple", "banana", "cherry"];
        assert_eq!(items.to_prompt(), "[apple, banana, cherry]");
    }

    #[test]
    fn test_to_prompt_for_vec_of_numbers() {
        let numbers = vec![1, 2, 3, 42];
        assert_eq!(numbers.to_prompt(), "[1, 2, 3, 42]");
    }

    #[test]
    fn test_to_prompt_for_empty_vec() {
        let empty: Vec<String> = vec![];
        assert_eq!(empty.to_prompt(), "[]");
    }

    #[test]
    fn test_to_prompt_for_nested_vec() {
        let nested = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(nested.to_prompt(), "[[1, 2], [3, 4]]");
    }

    #[test]
    fn test_to_prompt_parts_for_vec() {
        let items = vec!["a", "b", "c"];
        let parts = items.to_prompt_parts();
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            PromptPart::Text(text) => assert_eq!(text, "[a, b, c]"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_prompt_macro_no_args() {
        let prompt = prompt!("This is a static prompt.",).unwrap();
        assert_eq!(prompt, "This is a static prompt.");
    }

    #[test]
    fn test_render_prompt_with_json_value_dot_notation() {
        use serde_json::json;

        let context = json!({
            "user": {
                "name": "Alice",
                "age": 30,
                "profile": {
                    "role": "Developer"
                }
            }
        });

        let template =
            "{{ user.name }} is {{ user.age }} years old and works as {{ user.profile.role }}";
        let result = render_prompt(template, &context).unwrap();

        assert_eq!(result, "Alice is 30 years old and works as Developer");
    }

    #[test]
    fn test_render_prompt_with_hashmap_json_value() {
        use serde_json::json;
        use std::collections::HashMap;

        let mut context = HashMap::new();
        context.insert(
            "step_1_output".to_string(),
            json!({
                "result": "success",
                "data": {
                    "count": 42
                }
            }),
        );
        context.insert("task".to_string(), json!("analysis"));

        let template = "Task: {{ task }}, Result: {{ step_1_output.result }}, Count: {{ step_1_output.data.count }}";
        let result = render_prompt(template, &context).unwrap();

        assert_eq!(result, "Task: analysis, Result: success, Count: 42");
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PromptSetError {
    #[error("Target '{target}' not found. Available targets: {available:?}")]
    TargetNotFound {
        target: String,
        available: Vec<String>,
    },
    #[error("Failed to render prompt for target '{target}': {source}")]
    RenderFailed {
        target: String,
        source: minijinja::Error,
    },
}

/// A trait for types that can generate multiple named prompt targets.
///
/// This trait enables a single data structure to produce different prompt formats
/// for various use cases (e.g., human-readable vs. machine-parsable formats).
///
/// # Example
///
/// ```ignore
/// use llm_toolkit::prompt::{ToPromptSet, PromptPart};
/// use serde::Serialize;
///
/// #[derive(ToPromptSet, Serialize)]
/// #[prompt_for(name = "Visual", template = "## {{title}}\n\n> {{description}}")]
/// struct Task {
///     title: String,
///     description: String,
///
///     #[prompt_for(name = "Agent")]
///     priority: u8,
///
///     #[prompt_for(name = "Agent", rename = "internal_id")]
///     id: u64,
///
///     #[prompt_for(skip)]
///     is_dirty: bool,
/// }
///
/// let task = Task {
///     title: "Implement feature".to_string(),
///     description: "Add new functionality".to_string(),
///     priority: 1,
///     id: 42,
///     is_dirty: false,
/// };
///
/// // Generate visual prompt
/// let visual_prompt = task.to_prompt_for("Visual")?;
///
/// // Generate agent prompt
/// let agent_prompt = task.to_prompt_for("Agent")?;
/// ```
pub trait ToPromptSet {
    /// Generates multimodal prompt parts for the specified target.
    fn to_prompt_parts_for(&self, target: &str) -> Result<Vec<PromptPart>, PromptSetError>;

    /// Generates a text prompt for the specified target.
    ///
    /// This method extracts only the text portions from `to_prompt_parts_for()`
    /// and joins them together.
    fn to_prompt_for(&self, target: &str) -> Result<String, PromptSetError> {
        let parts = self.to_prompt_parts_for(target)?;
        let text = parts
            .iter()
            .filter_map(|part| match part {
                PromptPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(text)
    }
}

/// A trait for generating a prompt for a specific target type.
///
/// This allows a type (e.g., a `Tool`) to define how it should be represented
/// in a prompt when provided with a target context (e.g., an `Agent`).
pub trait ToPromptFor<T> {
    /// Generates a prompt for the given target, using a specific mode.
    fn to_prompt_for_with_mode(&self, target: &T, mode: &str) -> String;

    /// Generates a prompt for the given target using the default "full" mode.
    ///
    /// This method provides backward compatibility by calling the `_with_mode`
    /// variant with a default mode.
    fn to_prompt_for(&self, target: &T) -> String {
        self.to_prompt_for_with_mode(target, "full")
    }
}
