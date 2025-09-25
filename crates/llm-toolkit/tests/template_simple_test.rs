use llm_toolkit::ToPrompt;
use serde::Serialize;

// Simple struct with template but no mode syntax (backward compatibility)
#[derive(ToPrompt, Serialize)]
#[prompt(template = "User {name} has role {role}.")]
struct SimpleUser {
    name: String,
    role: String,
}

#[test]
fn test_simple_template() {
    let user = SimpleUser {
        name: "Alice".to_string(),
        role: "Admin".to_string(),
    };

    let output = user.to_prompt();

    // Should use simple template rendering without mode processing
    assert_eq!(output, "User Alice has role Admin.");
}

// Test with primitive field types
#[derive(ToPrompt, Serialize)]
#[prompt(template = "Name: {name}, ID: {id}")]
struct PrimitiveFields {
    name: String,
    id: u32,
}

#[test]
fn test_template_with_primitive_fields() {
    let data = PrimitiveFields {
        name: "test value".to_string(),
        id: 42,
    };

    let output = data.to_prompt();

    // Should render primitive values directly
    assert!(output.contains("Name: test value"));
    assert!(output.contains("ID: 42"));
}
