//! Test ToPrompt macro with #[serde(rename_all)] attribute

use llm_toolkit::prompt::ToPrompt;
use llm_toolkit_macros::ToPrompt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
enum VisualTreatment {
    DelicateLuminous,
    BoldGraphic,
    PainterlyTextured,
    CinematicCrisp,
    SoftAtmospheric,
    CleanUniversal,
}

#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "camelCase")]
enum EmotionalTone {
    NostalgicYouth,
    EpicGrandeur,
    TenderIntimate,
}

#[derive(Serialize, Deserialize, ToPrompt)]
enum NoRenameEnum {
    FooBar,
    BazQux,
}

#[test]
fn test_snake_case_rename() {
    let schema = VisualTreatment::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Should contain snake_case values
    assert!(
        schema.contains("\"delicate_luminous\""),
        "Schema should contain snake_case variant"
    );
    assert!(
        schema.contains("\"cinematic_crisp\""),
        "Schema should contain snake_case variant"
    );

    // Should NOT contain PascalCase values
    assert!(
        !schema.contains("\"DelicateLuminous\""),
        "Schema should not contain PascalCase variant"
    );
    assert!(
        !schema.contains("\"CinematicCrisp\""),
        "Schema should not contain PascalCase variant"
    );
}

#[test]
fn test_camel_case_rename() {
    let schema = EmotionalTone::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Should contain camelCase values
    assert!(
        schema.contains("\"nostalgicYouth\""),
        "Schema should contain camelCase variant"
    );
    assert!(
        schema.contains("\"epicGrandeur\""),
        "Schema should contain camelCase variant"
    );

    // Should NOT contain PascalCase values
    assert!(
        !schema.contains("\"NostalgicYouth\""),
        "Schema should not contain PascalCase variant"
    );
}

#[test]
fn test_no_rename() {
    let schema = NoRenameEnum::prompt_schema();
    println!("Generated schema:\n{}", schema);

    // Should contain original PascalCase values
    assert!(
        schema.contains("\"FooBar\""),
        "Schema should contain PascalCase variant"
    );
    assert!(
        schema.contains("\"BazQux\""),
        "Schema should contain PascalCase variant"
    );
}

#[test]
fn test_serde_serialization_match() {
    // Test that serde serialization matches ToPrompt schema

    // snake_case
    let visual = VisualTreatment::CinematicCrisp;
    let json = serde_json::to_string(&visual).unwrap();
    assert_eq!(
        json, "\"cinematic_crisp\"",
        "Serde should serialize to snake_case"
    );

    // camelCase
    let tone = EmotionalTone::NostalgicYouth;
    let json = serde_json::to_string(&tone).unwrap();
    assert_eq!(
        json, "\"nostalgicYouth\"",
        "Serde should serialize to camelCase"
    );

    // No rename (PascalCase)
    let no_rename = NoRenameEnum::FooBar;
    let json = serde_json::to_string(&no_rename).unwrap();
    assert_eq!(json, "\"FooBar\"", "Serde should serialize to PascalCase");
}

#[test]
fn test_deserialization_with_schema_values() {
    // Test that values from schema can be deserialized successfully

    // snake_case
    let json = "\"cinematic_crisp\"";
    let result: Result<VisualTreatment, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Should deserialize snake_case value from schema"
    );

    // camelCase
    let json = "\"nostalgicYouth\"";
    let result: Result<EmotionalTone, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Should deserialize camelCase value from schema"
    );

    // No rename
    let json = "\"FooBar\"";
    let result: Result<NoRenameEnum, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Should deserialize PascalCase value from schema"
    );
}
