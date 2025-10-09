use llm_toolkit::ToPrompt;
use serde::{Deserialize, Serialize};

#[derive(ToPrompt, Serialize, Deserialize)]
#[prompt(mode = "full")]
struct Emblem {
    /// The name of the emblem
    name: String,
    /// A description of the emblem
    description: String,
}

#[derive(ToPrompt, Serialize, Deserialize)]
#[prompt(mode = "full")]
struct EmblemResponse {
    /// An obvious, straightforward emblem
    obvious_emblem: Emblem,
    /// A creative, unexpected emblem
    creative_emblem: Emblem,
}

#[test]
fn test_nested_object_schema_expansion() {
    let schema = EmblemResponse::prompt_schema();

    println!("Generated schema:\n{}", schema);

    // Schema should contain nested object structure, not just "emblem" string
    assert!(schema.contains("obvious_emblem"));
    assert!(schema.contains("creative_emblem"));

    // Should expand nested Emblem schema inline
    assert!(schema.contains("name"));
    assert!(schema.contains("description"));

    // Should NOT contain just the type name as a string value
    assert!(!schema.contains(r#""obvious_emblem": "emblem""#));
    assert!(!schema.contains(r#""creative_emblem": "emblem""#));

    // Should contain nested object braces (JSON style with quotes)
    assert!(schema.contains(r#""obvious_emblem": {"#));
    assert!(schema.contains(r#""creative_emblem": {"#));
}

#[test]
fn test_nested_object_with_primitives() {
    #[derive(ToPrompt, Serialize, Deserialize)]
    #[prompt(mode = "full")]
    struct Inner {
        /// Inner field
        value: String,
    }

    #[derive(ToPrompt, Serialize, Deserialize)]
    #[prompt(mode = "full")]
    struct Outer {
        /// A primitive field
        id: u64,
        /// A nested object
        inner: Inner,
        /// Another primitive
        name: String,
    }

    let schema = Outer::prompt_schema();

    println!("Schema with mixed types:\n{}", schema);

    // Primitives should be simple strings
    assert!(schema.contains(r#""id": "number""#));
    assert!(schema.contains(r#""name": "string""#));

    // Nested object should be expanded (JSON style)
    assert!(schema.contains(r#""inner": {"#));
    assert!(schema.contains(r#""value": "string""#));
}
