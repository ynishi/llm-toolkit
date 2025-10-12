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

    // Schema should use TypeScript type references (not inline expansion)
    assert!(schema.contains("obvious_emblem"));
    assert!(schema.contains("creative_emblem"));

    // Should reference Emblem type, not expand inline
    assert!(schema.contains("obvious_emblem: Emblem;"));
    assert!(schema.contains("creative_emblem: Emblem;"));

    // Nested type definition is NOT included in parent schema
    // Each type has its own schema via Type::prompt_schema()
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

    // Primitives use TypeScript format
    assert!(schema.contains("id: number;"));
    assert!(schema.contains("name: string;"));

    // Nested object should use type reference (TypeScript style)
    assert!(schema.contains("inner: Inner;"));
}
