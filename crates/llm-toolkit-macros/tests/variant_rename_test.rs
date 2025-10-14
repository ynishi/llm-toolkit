//! Test ToPrompt macro with variant-level rename attributes
//!
//! Tests the priority system for variant renaming:
//! 1. #[prompt(rename = "...")] - highest priority
//! 2. #[serde(rename = "...")]  - per-variant serde
//! 3. #[serde(rename_all = "...")] - enum-level rule
//! 4. Default PascalCase

use llm_toolkit::prompt::ToPrompt;
use llm_toolkit_macros::ToPrompt;
use serde::{Deserialize, Serialize};

// Test 1: #[prompt(rename)] at variant level
#[allow(clippy::enum_variant_names)]
#[derive(Serialize, Deserialize, ToPrompt)]
enum PromptRenameEnum {
    #[prompt(rename = "custom_name_one")]
    VariantOne,
    #[prompt(rename = "custom_name_two")]
    VariantTwo,
    VariantThree, // No rename, should use default
}

// Test 2: #[serde(rename)] at variant level
#[allow(clippy::enum_variant_names)]
#[derive(Serialize, Deserialize, ToPrompt)]
enum SerdeRenameEnum {
    #[serde(rename = "serde_custom_one")]
    VariantOne,
    #[serde(rename = "serde_custom_two")]
    VariantTwo,
    VariantThree, // No rename, should use default
}

// Test 3: Priority - #[prompt(rename)] overrides #[serde(rename)]
#[derive(Serialize, Deserialize, ToPrompt)]
enum PriorityPromptOverSerdeEnum {
    #[prompt(rename = "prompt_wins")]
    #[serde(rename = "serde_loses")]
    VariantOne,
    #[serde(rename = "serde_only")]
    VariantTwo,
}

// Test 4: Priority - #[serde(rename)] overrides #[serde(rename_all)]
#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
enum PrioritySerdeRenameOverRenameAllEnum {
    #[serde(rename = "custom_override")]
    VariantOne, // Should use "custom_override", not "variant_one"
    VariantTwo, // Should use snake_case: "variant_two"
}

// Test 5: Priority - #[prompt(rename)] overrides #[serde(rename_all)]
#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
enum PriorityPromptOverRenameAllEnum {
    #[prompt(rename = "prompt_override")]
    VariantOne, // Should use "prompt_override", not "variant_one"
    VariantTwo, // Should use snake_case: "variant_two"
}

// Test 6: All three attributes present - #[prompt(rename)] wins
#[allow(clippy::enum_variant_names)]
#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
enum PriorityAllThreeEnum {
    #[prompt(rename = "prompt_highest_priority")]
    #[serde(rename = "serde_middle_priority")]
    VariantOne, // Should use "prompt_highest_priority"
    #[serde(rename = "serde_override")]
    VariantTwo, // Should use "serde_override"
    VariantThree, // Should use "variant_three" (snake_case from rename_all)
}

// Test 7: Mixed with descriptions
#[allow(clippy::enum_variant_names)]
#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
enum MixedWithDescriptionsEnum {
    #[prompt(rename = "custom_one")]
    #[prompt(description = "Custom renamed variant")]
    VariantOne,
    #[serde(rename = "serde_two")]
    #[prompt(description = "Serde renamed variant")]
    VariantTwo,
    #[prompt(description = "Snake case variant")]
    VariantThree,
}

#[test]
fn test_prompt_rename_variant() {
    let schema = PromptRenameEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Should contain custom prompt rename values
    assert!(
        schema.contains("\"custom_name_one\""),
        "Schema should contain prompt renamed variant: custom_name_one"
    );
    assert!(
        schema.contains("\"custom_name_two\""),
        "Schema should contain prompt renamed variant: custom_name_two"
    );

    // VariantThree has no rename, should use default PascalCase
    assert!(
        schema.contains("\"VariantThree\""),
        "Schema should contain default PascalCase variant: VariantThree"
    );

    // Should NOT contain original variant names for renamed ones
    assert!(
        !schema.contains("\"VariantOne\""),
        "Schema should not contain original VariantOne"
    );
    assert!(
        !schema.contains("\"VariantTwo\""),
        "Schema should not contain original VariantTwo"
    );
}

#[test]
fn test_serde_rename_variant() {
    let schema = SerdeRenameEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Should contain serde rename values
    assert!(
        schema.contains("\"serde_custom_one\""),
        "Schema should contain serde renamed variant: serde_custom_one"
    );
    assert!(
        schema.contains("\"serde_custom_two\""),
        "Schema should contain serde renamed variant: serde_custom_two"
    );

    // VariantThree has no rename, should use default
    assert!(
        schema.contains("\"VariantThree\""),
        "Schema should contain default variant: VariantThree"
    );
}

#[test]
fn test_serde_rename_matches_serialization() {
    // Test that serde serialization matches ToPrompt schema for variant-level rename
    let variant = SerdeRenameEnum::VariantOne;
    let json = serde_json::to_string(&variant).unwrap();
    assert_eq!(
        json, "\"serde_custom_one\"",
        "Serde should serialize to the renamed value"
    );

    // Verify deserialization works
    let result: Result<SerdeRenameEnum, _> = serde_json::from_str("\"serde_custom_one\"");
    assert!(result.is_ok(), "Should deserialize from renamed value");
}

#[test]
fn test_priority_prompt_over_serde() {
    let schema = PriorityPromptOverSerdeEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // #[prompt(rename)] should win over #[serde(rename)]
    assert!(
        schema.contains("\"prompt_wins\""),
        "Schema should contain prompt rename (highest priority)"
    );
    assert!(
        !schema.contains("\"serde_loses\""),
        "Schema should not contain serde rename when prompt rename exists"
    );

    // VariantTwo only has serde rename
    assert!(
        schema.contains("\"serde_only\""),
        "Schema should contain serde rename when no prompt rename"
    );
}

#[test]
fn test_priority_serde_rename_over_rename_all() {
    let schema = PrioritySerdeRenameOverRenameAllEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // #[serde(rename)] should override #[serde(rename_all)]
    assert!(
        schema.contains("\"custom_override\""),
        "Schema should contain serde rename override"
    );
    assert!(
        !schema.contains("\"variant_one\""),
        "Schema should not use rename_all when variant has serde rename"
    );

    // VariantTwo should use rename_all (snake_case)
    assert!(
        schema.contains("\"variant_two\""),
        "Schema should use rename_all for variants without serde rename"
    );
}

#[test]
fn test_priority_prompt_over_rename_all() {
    let schema = PriorityPromptOverRenameAllEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // #[prompt(rename)] should override #[serde(rename_all)]
    assert!(
        schema.contains("\"prompt_override\""),
        "Schema should contain prompt rename override"
    );
    assert!(
        !schema.contains("\"variant_one\""),
        "Schema should not use rename_all when variant has prompt rename"
    );

    // VariantTwo should use rename_all (snake_case)
    assert!(
        schema.contains("\"variant_two\""),
        "Schema should use rename_all for variants without prompt rename"
    );
}

#[test]
fn test_priority_all_three_attributes() {
    let schema = PriorityAllThreeEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Priority 1: #[prompt(rename)] wins
    assert!(
        schema.contains("\"prompt_highest_priority\""),
        "Schema should use prompt rename (highest priority)"
    );
    assert!(
        !schema.contains("\"serde_middle_priority\""),
        "Schema should not use serde rename when prompt rename exists"
    );
    assert!(
        !schema.contains("\"variant_one\""),
        "Schema should not use rename_all when prompt rename exists"
    );

    // Priority 2: #[serde(rename)] wins over rename_all
    assert!(
        schema.contains("\"serde_override\""),
        "Schema should use serde rename (middle priority)"
    );
    assert!(
        !schema.contains("\"variant_two\""),
        "Schema should not use rename_all when serde rename exists"
    );

    // Priority 3: rename_all applies when no specific renames
    assert!(
        schema.contains("\"variant_three\""),
        "Schema should use rename_all (lowest priority)"
    );
}

#[test]
fn test_mixed_with_descriptions() {
    let schema = MixedWithDescriptionsEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Renamed variants should still show descriptions
    assert!(
        schema.contains("\"custom_one\""),
        "Schema should contain prompt renamed variant"
    );
    assert!(
        schema.contains("Custom renamed variant"),
        "Schema should contain description for renamed variant"
    );

    assert!(
        schema.contains("\"serde_two\""),
        "Schema should contain serde renamed variant"
    );
    assert!(
        schema.contains("Serde renamed variant"),
        "Schema should contain description for serde renamed variant"
    );

    assert!(
        schema.contains("\"variant_three\""),
        "Schema should contain snake_case variant"
    );
    assert!(
        schema.contains("Snake case variant"),
        "Schema should contain description for snake_case variant"
    );
}

#[test]
fn test_to_prompt_respects_variant_rename() {
    // Test that to_prompt() implementation also respects variant renames
    let variant = PromptRenameEnum::VariantOne;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "custom_name_one",
        "to_prompt() should use renamed value"
    );

    let variant = PromptRenameEnum::VariantThree;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "VariantThree",
        "to_prompt() should use default value when no rename"
    );
}

#[test]
fn test_to_prompt_respects_serde_variant_rename() {
    let variant = SerdeRenameEnum::VariantOne;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "serde_custom_one",
        "to_prompt() should use serde renamed value"
    );
}

#[test]
fn test_to_prompt_priority_prompt_over_serde() {
    let variant = PriorityPromptOverSerdeEnum::VariantOne;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "prompt_wins",
        "to_prompt() should prioritize prompt rename over serde rename"
    );

    let variant = PriorityPromptOverSerdeEnum::VariantTwo;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "serde_only",
        "to_prompt() should use serde rename when no prompt rename"
    );
}

#[test]
fn test_to_prompt_priority_all_three() {
    let variant = PriorityAllThreeEnum::VariantOne;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "prompt_highest_priority",
        "to_prompt() should prioritize prompt rename"
    );

    let variant = PriorityAllThreeEnum::VariantTwo;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "serde_override",
        "to_prompt() should prioritize serde rename over rename_all"
    );

    let variant = PriorityAllThreeEnum::VariantThree;
    let prompt = variant.to_prompt();
    assert_eq!(
        prompt, "variant_three",
        "to_prompt() should use rename_all when no specific renames"
    );
}
