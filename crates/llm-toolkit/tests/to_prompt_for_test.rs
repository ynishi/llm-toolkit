#![cfg(feature = "derive")]

use llm_toolkit::prompt::{ToPrompt as ToPromptTrait, ToPromptFor as ToPromptForTrait};
use llm_toolkit::{ToPrompt, ToPromptFor};
use serde::Serialize;

// Simple target type
#[derive(Serialize)]
struct Target {
    id: String,
}

// Test basic field interpolation
#[derive(Serialize, ToPrompt, ToPromptFor, Default)]
#[prompt_for(target = "Target", template = "User {{ name }} has value {{ count }}")]
struct SimpleFields {
    name: String,
    count: u32,
}

// Test {self} placeholder
#[derive(Serialize, ToPrompt, ToPromptFor, Default)]
#[prompt_for(target = "Target", template = "Configuration:\n{{ self }}")]
struct WithSelf {
    enabled: bool,
    timeout: u32,
}

// Test {self:mode} placeholders
#[derive(Serialize, ToPrompt, ToPromptFor, Default)]
#[prompt_for(
    target = "Target",
    template = "## Schema\n{{ self:schema_only }}\n\n## Example\n{{ self:example_only }}"
)]
#[prompt(mode = "full")]
struct WithModes {
    field1: String,
    field2: i32,
    #[prompt(example = "test-value")]
    field3: String,
}

// Test mixed placeholders
#[derive(Serialize, ToPrompt, ToPromptFor, Default)]
#[prompt_for(
    target = "Target",
    template = "Config for {{ name }}:\nFull: {{ self }}\nSchema: {{ self:schema_only }}"
)]
#[prompt(mode = "full")]
struct MixedPlaceholders {
    name: String,
    value: i32,
}

#[test]
fn test_simple_field_interpolation() {
    let config = SimpleFields {
        name: "Alice".to_string(),
        count: 42,
    };
    let target = Target {
        id: "t1".to_string(),
    };

    let result = config.to_prompt_for_with_mode(&target, "full");
    assert_eq!(result, "User Alice has value 42");
}

#[test]
fn test_self_placeholder() {
    let config = WithSelf {
        enabled: true,
        timeout: 30,
    };
    let target = Target {
        id: "t2".to_string(),
    };

    let result = config.to_prompt_for_with_mode(&target, "full");
    assert!(result.contains("Configuration:"));
    assert!(result.contains("enabled:"));
    assert!(result.contains("timeout:"));
}

#[test]
fn test_mode_placeholders() {
    let config = WithModes {
        field1: "hello".to_string(),
        field2: 123,
        field3: "world".to_string(),
    };
    let target = Target {
        id: "t3".to_string(),
    };

    let result = config.to_prompt_for_with_mode(&target, "full");

    // Check schema section
    assert!(result.contains("## Schema"));
    assert!(result.contains("Schema for `WithModes`"));
    assert!(result.contains("\"field1\": \"string\""));
    assert!(result.contains("\"field2\": \"number\""));

    // Check example section
    assert!(result.contains("## Example"));
    assert!(result.contains("\"field1\": \"hello\""));
    assert!(result.contains("\"field2\": 123"));
    assert!(result.contains("\"field3\": \"test-value\"")); // Should use the example attribute
}

#[test]
fn test_mixed_placeholders() {
    let config = MixedPlaceholders {
        name: "TestConfig".to_string(),
        value: 999,
    };
    let target = Target {
        id: "t4".to_string(),
    };

    let result = config.to_prompt_for_with_mode(&target, "full");

    assert!(result.contains("Config for TestConfig:"));
    assert!(result.contains("Full:"));
    // Since MixedPlaceholders has mode support, {self} will use the mode-based output
    assert!(result.contains("Schema for `MixedPlaceholders`"));
    assert!(result.contains("\"name\": \"TestConfig\""));
    assert!(result.contains("\"value\": 999"));
    assert!(result.contains("Schema:"));
}

#[test]
fn test_mode_parameter_affects_self_placeholder() {
    // Test with a struct that HAS mode support
    let config = WithModes {
        field1: "test".to_string(),
        field2: 42,
        field3: "example".to_string(),
    };
    let target = Target {
        id: "t5".to_string(),
    };

    // For structs with mode support, the mode parameter should affect {self:mode} placeholders
    let result = config.to_prompt_for_with_mode(&target, "full");

    // Check that both schema_only and example_only modes are in the output
    assert!(result.contains("## Schema"));
    assert!(result.contains("Schema for `WithModes`"));
    assert!(result.contains("## Example"));
    assert!(result.contains("\"field1\": \"test\""));
}
