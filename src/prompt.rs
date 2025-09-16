//! A trait and macro for converting types into prompt strings.

use minijinja::Environment;
use serde::Serialize;
use std::fmt::Display;

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
/// // Any type implementing Display automatically gets ToPrompt
/// let number = 42;
/// assert_eq!(number.to_prompt(), "42");
///
/// let text = "Hello, LLM!";
/// assert_eq!(text.to_prompt(), "Hello, LLM!");
/// ```
///
/// # Custom Implementation
///
/// While a blanket implementation is provided for all types that implement
/// `Display`, you can provide custom implementations for your own types:
///
/// ```
/// use llm_toolkit::prompt::ToPrompt;
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
/// // The blanket implementation provides ToPrompt automatically
/// let custom = CustomType { value: "custom".to_string() };
/// assert_eq!(custom.to_prompt(), "custom");
/// ```
pub trait ToPrompt {
    /// Converts the object into a prompt string.
    fn to_prompt(&self) -> String;
}

/// A blanket implementation of `ToPrompt` for any type that implements `Display`.
///
/// This provides automatic `ToPrompt` functionality for all standard library
/// types and custom types that implement `Display`.
impl<T: Display> ToPrompt for T {
    fn to_prompt(&self) -> String {
        self.to_string()
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
    fn test_prompt_macro_no_args() {
        let prompt = prompt!("This is a static prompt.",).unwrap();
        assert_eq!(prompt, "This is a static prompt.");
    }
}
