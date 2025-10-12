use llm_toolkit::ToPrompt;
use serde::Serialize;

// Simple nested struct that implements ToPrompt
#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "schema_only")]
/// Simple inner struct
struct Inner {
    value: String,
}

// Struct with template using mode syntax
#[derive(ToPrompt, Serialize)]
#[prompt(template = "Normal: {{ inner }}, Schema: {{ inner:schema_only }}")]
struct Outer {
    inner: Inner,
}

#[test]
fn test_template_with_field_mode() {
    let data = Outer {
        inner: Inner {
            value: "test".to_string(),
        },
    };

    let output = data.to_prompt();
    println!("Output: {}", output);

    // Should contain both normal and schema representations
    assert!(output.contains("Normal:"));
    assert!(output.contains("Schema:"));

    // The schema_only mode should show the schema (TypeScript format)
    assert!(output.contains("type Inner = {"));
}
