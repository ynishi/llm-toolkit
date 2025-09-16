//! A trait for converting types into prompt strings.

use std::fmt::Display;

/// A trait for converting any type into a string suitable for an LLM prompt.
///
/// # Example: Generating a selection prompt from an Enum
///
/// While `ToPrompt` is implemented for individual enum variants (via `Display`),
/// you can use it in combination with a crate like `strum` to generate a prompt
/// for selecting from all variants of an enum.
///
/// ```
/// use llm_toolkit::prompt::ToPrompt;
/// use strum::{EnumIter, IntoEnumIterator, Display};
///
/// #[derive(Debug, EnumIter, Display)]
/// enum TaskCategory {
///     Analysis,
///     Refactoring,
///     Testing,
/// }
///
/// let selection_prompt = format!(
///     "Select one of the following categories: {}",
///     TaskCategory::iter()
///         .map(|variant| variant.to_prompt())
///         .collect::<Vec<_>>()
///         .join(", ")
/// );
///
/// assert_eq!(
///     selection_prompt,
///     "Select one of the following categories: Analysis, Refactoring, Testing"
/// );
/// ```
pub trait ToPrompt {
    /// Converts the object into a prompt string.
    fn to_prompt(&self) -> String;
}

/// A blanket implementation of `ToPrompt` for any type that implements `Display`.
///
/// This allows any type that can be formatted as a string (including standard
/// types like numbers, and enums that derive `strum::Display`) to be used
/// as a prompt component automatically.
impl<T: Display> ToPrompt for T {
    fn to_prompt(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}