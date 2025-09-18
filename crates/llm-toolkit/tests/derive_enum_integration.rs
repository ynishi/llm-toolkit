#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;

    #[derive(ToPrompt, Debug, PartialEq)]
    enum TestEnum {
        /// First variant with doc
        First,
        #[prompt("Custom description")]
        Second,
        #[prompt(skip)]
        Third,
        Fourth,
    }

    #[test]
    fn test_enum_prompt_generation() {
        let instance = TestEnum::First;
        let prompt = instance.to_prompt();

        // Check that the prompt contains expected elements
        assert!(prompt.contains("TestEnum:"));
        assert!(prompt.contains("Possible values:"));
        assert!(prompt.contains("- First: First variant with doc"));
        assert!(prompt.contains("- Second: Custom description"));
        assert!(prompt.contains("- Fourth"));

        // Check that skipped variant is not included
        assert!(!prompt.contains("Third"));
    }

    #[test]
    fn test_all_fallback_priorities() {
        #[derive(ToPrompt)]
        enum PriorityTest {
            #[prompt(skip)]
            SkipMe,

            #[prompt("Custom override")]
            /// This doc comment should be ignored
            CustomOverride,

            /// Uses doc comment
            WithDoc,

            PlainName,
        }

        let instance = PriorityTest::WithDoc;
        let prompt = instance.to_prompt();

        // Verify skip works
        assert!(!prompt.contains("SkipMe"));

        // Verify custom description overrides doc comment
        assert!(prompt.contains("- CustomOverride: Custom override"));
        assert!(!prompt.contains("This doc comment should be ignored"));

        // Verify doc comment is used
        assert!(prompt.contains("- WithDoc: Uses doc comment"));

        // Verify plain name is shown
        assert!(prompt.contains("- PlainName"));
    }

    #[test]
    fn test_empty_enum() {
        #[derive(ToPrompt)]
        enum EmptyEnum {}

        // This should compile and generate a prompt with no values
        // (Though an empty enum isn't very useful in practice)
    }

    #[test]
    fn test_all_variants_skipped() {
        #[derive(ToPrompt)]
        enum AllSkipped {
            #[prompt(skip)]
            A,
            #[prompt(skip)]
            B,
            #[prompt(skip)]
            C,
        }

        let instance = AllSkipped::A;
        let prompt = instance.to_prompt();

        // Should have header but no variants
        assert!(prompt.contains("AllSkipped:"));
        assert!(prompt.contains("Possible values:"));
        assert!(!prompt.contains("- A"));
        assert!(!prompt.contains("- B"));
        assert!(!prompt.contains("- C"));
    }
}
