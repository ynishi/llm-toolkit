#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;

    #[derive(ToPrompt, Debug, PartialEq)]
    #[allow(dead_code)]
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
        // Test instance prompt (single variant)
        let instance = TestEnum::First;
        let prompt = instance.to_prompt();
        assert_eq!(prompt, "First: First variant with doc");

        // Test schema prompt (all variants)
        let schema = TestEnum::prompt_schema();
        assert!(schema.contains("TestEnum:"));
        assert!(schema.contains("Possible values:"));
        assert!(schema.contains("- First: First variant with doc"));
        assert!(schema.contains("- Second: Custom description"));
        assert!(schema.contains("- Fourth"));

        // Check that skipped variant is not included
        assert!(!schema.contains("Third"));
    }

    #[test]
    fn test_all_fallback_priorities() {
        #[derive(ToPrompt)]
        #[allow(dead_code)]
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

        // Test instance prompt
        let instance = PriorityTest::WithDoc;
        let prompt = instance.to_prompt();
        assert_eq!(prompt, "WithDoc: Uses doc comment");

        // Test schema prompt (all variants)
        let schema = PriorityTest::prompt_schema();

        // Verify skip works
        assert!(!schema.contains("SkipMe"));

        // Verify custom description overrides doc comment
        assert!(schema.contains("- CustomOverride: Custom override"));
        assert!(!schema.contains("This doc comment should be ignored"));

        // Verify doc comment is used
        assert!(schema.contains("- WithDoc: Uses doc comment"));

        // Verify plain name is shown
        assert!(schema.contains("- PlainName"));
    }

    #[test]
    fn test_empty_enum() {
        #[derive(ToPrompt)]
        #[allow(dead_code)]
        enum EmptyEnum {}

        // This should compile and generate a prompt with no values
        // (Though an empty enum isn't very useful in practice)
    }

    #[test]
    fn test_all_variants_skipped() {
        #[derive(ToPrompt)]
        #[allow(dead_code)]
        enum AllSkipped {
            #[prompt(skip)]
            A,
            #[prompt(skip)]
            B,
            #[prompt(skip)]
            C,
        }

        // Test instance prompt (even skipped variants have names)
        let instance = AllSkipped::A;
        let prompt = instance.to_prompt();
        assert_eq!(prompt, "A");

        // Test schema prompt (all variants are skipped, so schema shows header but no variants)
        let schema = AllSkipped::prompt_schema();
        assert!(schema.contains("AllSkipped:"));
        assert!(schema.contains("Possible values:"));
        assert!(!schema.contains("- A"));
        assert!(!schema.contains("- B"));
        assert!(!schema.contains("- C"));
    }

    #[test]
    fn test_struct_default_prompt_generation() {
        #[derive(ToPrompt)]
        struct TestStruct {
            name: String,
            age: u32,
            active: bool,
        }

        let instance = TestStruct {
            name: "Yui".to_string(),
            age: 28,
            active: true,
        };

        let prompt = instance.to_prompt();
        let expected_prompt = "name: Yui\nage: 28\nactive: true";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_struct_skip_attribute() {
        #[derive(ToPrompt)]
        #[allow(dead_code)]
        struct UserProfile {
            username: String,
            display_name: String,
            #[prompt(skip)]
            internal_id: u64,
        }

        let profile = UserProfile {
            username: "yui".to_string(),
            display_name: "Yui".to_string(),
            internal_id: 12345,
        };

        let prompt = profile.to_prompt();
        let expected_prompt = "username: yui\ndisplay_name: Yui";

        assert_eq!(prompt, expected_prompt);
        assert!(!prompt.contains("internal_id"));
    }

    #[test]
    fn test_struct_priority_based_keys() {
        #[derive(ToPrompt)]
        #[allow(dead_code)]
        struct PriorityTestStruct {
            // Priority 3: Field name fallback
            plain_field: String,

            // Priority 2: Doc comment
            /// User's full name
            documented_field: String,

            // Priority 1: Rename attribute (highest priority)
            #[prompt(rename = "custom_key")]
            renamed_field: String,

            // Combined: rename takes priority over doc comment
            /// This doc should be ignored
            #[prompt(rename = "overridden")]
            both_attrs: String,

            // Skip attribute
            #[prompt(skip)]
            skipped_field: String,
        }

        let instance = PriorityTestStruct {
            plain_field: "plain".to_string(),
            documented_field: "doc".to_string(),
            renamed_field: "renamed".to_string(),
            both_attrs: "both".to_string(),
            skipped_field: "skip".to_string(),
        };

        let prompt = instance.to_prompt();

        // Test field name fallback
        assert!(prompt.contains("plain_field: plain"));

        // Test doc comment as key
        assert!(prompt.contains("User's full name: doc"));
        assert!(!prompt.contains("documented_field: doc"));

        // Test rename attribute
        assert!(prompt.contains("custom_key: renamed"));
        assert!(!prompt.contains("renamed_field: renamed"));

        // Test rename overrides doc comment
        assert!(prompt.contains("overridden: both"));
        assert!(!prompt.contains("This doc should be ignored"));
        assert!(!prompt.contains("both_attrs: both"));

        // Test skip
        assert!(!prompt.contains("skipped_field"));
        assert!(!prompt.contains("skip"));
    }

    #[test]
    fn test_struct_multiple_attributes() {
        #[derive(ToPrompt)]
        struct ComplexStruct {
            #[prompt(rename = "id")]
            user_id: u32,

            #[prompt(skip)]
            #[allow(dead_code)]
            _internal: String,

            /// API endpoint URL
            #[prompt(rename = "endpoint")]
            api_url: String,
        }

        let instance = ComplexStruct {
            user_id: 42,
            _internal: "secret".to_string(),
            api_url: "https://api.example.com".to_string(),
        };

        let prompt = instance.to_prompt();

        assert!(prompt.contains("id: 42"));
        assert!(prompt.contains("endpoint: https://api.example.com"));
        assert!(!prompt.contains("user_id"));
        assert!(!prompt.contains("api_url"));
        assert!(!prompt.contains("_internal"));
        assert!(!prompt.contains("secret"));
        assert!(!prompt.contains("API endpoint URL"));
    }

    // TODO: Fix type inference issue when all fields are skipped
    // #[test]
    // fn test_struct_all_fields_skipped() {
    //     #[derive(ToPrompt)]
    //     struct AllSkippedStruct {
    //         #[prompt(skip)]
    //         field1: String,
    //         #[prompt(skip)]
    //         field2: u32,
    //     }

    //     let instance = AllSkippedStruct {
    //         field1: "test".to_string(),
    //         field2: 123,
    //     };

    //     let prompt = instance.to_prompt();
    //     assert_eq!(prompt, "");
    // }
}
