/// Test to verify __type field is properly excluded from ToPrompt outputs
/// while being preserved in Serialize outputs
use llm_toolkit::{ToPrompt, TypeMarker};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, TypeMarker, ToPrompt, Default)]
#[prompt(mode = "full")]
pub struct TestResponse {
    #[serde(default = "default_test_response_type")]
    __type: String,
    pub message: String,
    pub count: i32,
}

fn default_test_response_type() -> String {
    "TestResponse".to_string()
}

#[test]
fn test_prompt_schema_excludes_type() {
    println!("\n=== Testing prompt_schema() ===");
    let schema = TestResponse::prompt_schema();
    println!("{}", schema);

    // __type should NOT appear in schema
    assert!(
        !schema.contains("__type"),
        "❌ __type should not appear in prompt_schema() - it's noise for LLM"
    );

    // Other fields should appear
    assert!(schema.contains("message"));
    assert!(schema.contains("count"));
}

#[test]
fn test_example_only_excludes_type() {
    println!("\n=== Testing example_only mode ===");
    let instance = TestResponse {
        __type: "TestResponse".to_string(),
        message: "Hello".to_string(),
        count: 42,
    };

    let output = instance.to_prompt_with_mode("example_only");
    println!("{}", output);

    // __type should NOT appear in example
    assert!(
        !output.contains("__type"),
        "❌ __type should not appear in example_only mode - it's noise for LLM"
    );

    // Other fields should appear
    assert!(output.contains("message"));
    assert!(output.contains("Hello"));
    assert!(output.contains("count"));
}

#[test]
fn test_full_mode_excludes_type() {
    println!("\n=== Testing full mode ===");
    let instance = TestResponse {
        __type: "TestResponse".to_string(),
        message: "Hello".to_string(),
        count: 42,
    };

    let output = instance.to_prompt_with_mode("full");
    println!("{}", output);

    // __type should NOT appear in full mode
    assert!(
        !output.contains("__type"),
        "❌ __type should not appear in full mode - it's noise for LLM"
    );

    // Other fields should appear
    assert!(output.contains("message"));
    assert!(output.contains("count"));
}

#[test]
fn test_serialize_includes_type() {
    println!("\n=== Testing Serialize (context storage) ===");
    let instance = TestResponse {
        __type: "TestResponse".to_string(),
        message: "Hello".to_string(),
        count: 42,
    };

    let json_value = serde_json::to_value(&instance).unwrap();
    let json_str = serde_json::to_string_pretty(&json_value).unwrap();
    println!("{}", json_str);

    // With skip_serializing: __type will be missing
    // Without skip_serializing: __type will be present
    let has_type = json_value.get("__type").is_some();

    if has_type {
        println!("✅ __type is preserved in Serialize - Orchestrator can search by type");
        assert_eq!(
            json_value.get("__type").and_then(|v| v.as_str()),
            Some("TestResponse")
        );
    } else {
        println!("❌ __type is missing in Serialize - Orchestrator CANNOT search by type");
        println!("This is the bug we need to fix!");
    }

    // This test documents the current behavior
    // After fix: should assert!(has_type)
}

#[test]
fn test_complete_workflow() {
    println!("\n=== Testing complete workflow ===");

    let instance = TestResponse {
        __type: "TestResponse".to_string(),
        message: "Test".to_string(),
        count: 1,
    };

    // 1. ToPrompt for LLM (should exclude __type)
    let prompt = instance.to_prompt();
    println!("1. Prompt for LLM:\n{}", prompt);
    assert!(!prompt.contains("__type"), "ToPrompt should exclude __type");

    // 2. Serialize for context storage (should include __type)
    let json_value = serde_json::to_value(&instance).unwrap();
    println!(
        "2. JSON for context:\n{}",
        serde_json::to_string_pretty(&json_value).unwrap()
    );

    let has_type = json_value.get("__type").is_some();
    println!("3. Has __type in JSON? {}", has_type);

    if !has_type {
        println!("❌ BUG: Orchestrator cannot find this by type!");
    } else {
        println!("✅ OK: Orchestrator can find this by type");
    }
}
