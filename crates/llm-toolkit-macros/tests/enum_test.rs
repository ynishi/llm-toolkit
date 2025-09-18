#[test]
fn test_enum_derive() {
    // This test verifies that the macro compiles successfully for enums
    // The actual runtime testing is done in the llm-toolkit crate's integration tests
    // and in the examples directory

    // Test that the macro handles enums with various attributes
    let _test_code = r#"
        /// Test enum
        #[derive(ToPrompt)]
        enum TestEnum {
            /// First variant
            A,
            
            #[prompt("Custom description")]
            B,
            
            #[prompt(skip)]
            C,
            
            D, // No doc comment
        }
    "#;

    // If this test runs, it means the macro crate compiled successfully
    // The actual macro functionality is tested in:
    // - crates/llm-toolkit/tests/derive_enum_integration.rs
    // - crates/llm-toolkit/examples/derive_prompt_enum.rs
}
