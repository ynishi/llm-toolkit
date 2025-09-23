use llm_toolkit::{ToPrompt, ToPromptSet};
use serde::Serialize;

// Test: Empty struct with multiple targets
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Empty1", template = "This is empty")]
#[prompt_for(name = "Empty2", template = "Also empty")]
struct EmptyStruct {}

#[test]
fn test_empty_struct_multiple_targets() {
    let empty = EmptyStruct {};

    let result1 = empty.to_prompt_for("Empty1").unwrap();
    assert_eq!(result1, "This is empty");

    let result2 = empty.to_prompt_for("Empty2").unwrap();
    assert_eq!(result2, "Also empty");
}

// Test: Field that appears only in one target (not in template)
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Minimal", template = "Just {{title}}")]
struct SelectiveFields {
    title: String,

    #[prompt_for(name = "Detailed")]
    extra_info: String,
}

#[test]
fn test_selective_field_inclusion() {
    let data = SelectiveFields {
        title: "Test Title".to_string(),
        extra_info: "Extra Information".to_string(),
    };

    // Minimal target should only show title (via template)
    let minimal = data.to_prompt_for("Minimal").unwrap();
    assert_eq!(minimal, "Just Test Title");
    assert!(!minimal.contains("Extra Information"));

    // Detailed target should include both fields
    let detailed = data.to_prompt_for("Detailed").unwrap();
    assert!(detailed.contains("title: Test Title"));
    assert!(detailed.contains("extra_info: Extra Information"));
}

// Test: Multiple renames for the same field
#[derive(ToPromptSet, Serialize, Debug)]
struct MultiRename {
    #[prompt_for(name = "Target1", rename = "first_name")]
    #[prompt_for(name = "Target2", rename = "second_name")]
    value: String,
}

#[test]
fn test_multiple_renames() {
    let data = MultiRename {
        value: "test_value".to_string(),
    };

    let target1 = data.to_prompt_for("Target1").unwrap();
    assert!(target1.contains("first_name: test_value"));
    assert!(!target1.contains("second_name"));
    assert!(!target1.contains("value:"));

    let target2 = data.to_prompt_for("Target2").unwrap();
    assert!(target2.contains("second_name: test_value"));
    assert!(!target2.contains("first_name"));
    assert!(!target2.contains("value:"));
}

// Test: Mixed template and non-template targets
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Formatted", template = "{{name}} ({{age}} years old)")]
struct MixedTargets {
    name: String,
    age: u32,

    #[prompt_for(name = "Raw")]
    city: String,
}

#[test]
fn test_mixed_template_and_raw() {
    let person = MixedTargets {
        name: "Alice".to_string(),
        age: 30,
        city: "Tokyo".to_string(),
    };

    // Formatted uses template
    let formatted = person.to_prompt_for("Formatted").unwrap();
    assert_eq!(formatted, "Alice (30 years old)");
    assert!(!formatted.contains("city"));

    // Raw uses key-value format
    let raw = person.to_prompt_for("Raw").unwrap();
    assert!(raw.contains("name: Alice"));
    assert!(raw.contains("age: 30"));
    assert!(raw.contains("city: Tokyo"));
}

// Test: Field with skip for specific target
#[derive(ToPromptSet, Serialize, Debug)]
struct ConditionalSkip {
    always_shown: String,

    #[prompt_for(name = "Public")]
    #[prompt_for(name = "Private", skip)]
    sensitive_data: String,
}

#[test]
fn test_conditional_skip() {
    let data = ConditionalSkip {
        always_shown: "Public info".to_string(),
        sensitive_data: "Secret".to_string(),
    };

    // Public target includes sensitive_data
    let public = data.to_prompt_for("Public").unwrap();
    assert!(public.contains("always_shown: Public info"));
    assert!(public.contains("sensitive_data: Secret"));

    // Private target skips sensitive_data (counterintuitive but testing the skip)
    let private = data.to_prompt_for("Private").unwrap();
    assert!(private.contains("always_shown: Public info"));
    assert!(!private.contains("sensitive_data"));
    assert!(!private.contains("Secret"));
}

// Test: All fields belong to specific targets (no default fields)
#[derive(ToPromptSet, Serialize, Debug)]
struct NoDefaultFields {
    #[prompt_for(name = "A")]
    field_a: String,

    #[prompt_for(name = "B")]
    field_b: String,

    #[prompt_for(name = "C")]
    field_c: String,
}

#[test]
fn test_no_default_fields() {
    let data = NoDefaultFields {
        field_a: "A".to_string(),
        field_b: "B".to_string(),
        field_c: "C".to_string(),
    };

    let target_a = data.to_prompt_for("A").unwrap();
    assert!(target_a.contains("field_a: A"));
    assert!(!target_a.contains("field_b"));
    assert!(!target_a.contains("field_c"));

    let target_b = data.to_prompt_for("B").unwrap();
    assert!(!target_b.contains("field_a"));
    assert!(target_b.contains("field_b: B"));
    assert!(!target_b.contains("field_c"));

    let target_c = data.to_prompt_for("C").unwrap();
    assert!(!target_c.contains("field_a"));
    assert!(!target_c.contains("field_b"));
    assert!(target_c.contains("field_c: C"));
}

// Test: Template with missing field (should handle gracefully)
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Incomplete",
    template = "Name: {{name}}, Missing: {{nonexistent}}"
)]
struct TemplateWithMissingField {
    name: String,
}

#[test]
fn test_template_with_missing_field() {
    let data = TemplateWithMissingField {
        name: "Test".to_string(),
    };

    // minijinja should handle missing fields gracefully
    let result = data.to_prompt_for("Incomplete");
    // This might render as "Name: Test, Missing: " or similar depending on minijinja's behavior
    assert!(result.is_ok());
    let text = result.unwrap();
    assert!(text.contains("Name: Test"));
}

// Test: Single target works (edge case - ToPromptSet can work with just one target)
#[derive(ToPromptSet, Serialize, Debug)]
struct SingleTargetStruct {
    #[prompt_for(name = "OnlyTarget")]
    field1: String,
    #[prompt_for(name = "OnlyTarget")]
    field2: i32,
}

#[test]
fn test_single_target_edge_case() {
    let data = SingleTargetStruct {
        field1: "test".to_string(),
        field2: 42,
    };

    // Even with a single target, ToPromptSet should work
    let result = data.to_prompt_for("OnlyTarget").unwrap();
    assert!(result.contains("field1: test"));
    assert!(result.contains("field2: 42"));

    // Non-existent target should error with available targets
    let err = data.to_prompt_for("NonExistent").unwrap_err();
    let err_msg = err.to_string();
    assert!(err_msg.contains("Available targets: [\"OnlyTarget\"]"));
}
