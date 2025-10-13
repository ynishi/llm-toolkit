use llm_toolkit::ToPrompt;
use serde::{Deserialize, Serialize};

/// Character appearance details
#[derive(ToPrompt, Serialize, Deserialize)]
#[prompt(mode = "full")]
struct AppearanceResponse {
    /// Character's age (e.g., "30", "mid-20s", "appears ancient")
    age: String,
    /// Gender presentation
    gender: String,
    /// Height in centimeters
    height_cm: u32,
    /// Detailed body type description
    physique: String,
    /// Specific hair description
    hair_style: String,
    /// Detailed eye description
    eye_details: String,
    /// Complete outfit description
    clothing: String,
    /// Unique visual markers
    distinguishing_features: String,
}

/// Character personality traits
#[derive(ToPrompt, Serialize, Deserialize)]
#[prompt(mode = "full")]
struct PersonalityResponse {
    /// Core personality traits
    traits: Vec<String>,
    /// Character's motivations
    motivations: String,
}

/// Complete character profile response
#[derive(ToPrompt, Serialize, Deserialize)]
#[prompt(mode = "full")]
struct ProfileResponse {
    /// Character's full name
    name: String,
    /// 2-3 sentence summary
    summary: String,
    /// Appearance details
    appearance: AppearanceResponse,
    /// Personality details
    personality: PersonalityResponse,
    /// Cover portrait image generation brief
    cover_portrait_brief: String,
}

#[test]
fn test_bug_report_case_nested_types_expanded() {
    let schema = ProfileResponse::prompt_schema();

    println!("=== Bug Report Test Case ===");
    println!("{}", schema);
    println!("============================\n");

    // ✅ Should contain nested type definitions
    assert!(
        schema.contains("type AppearanceResponse"),
        "Schema should include AppearanceResponse definition"
    );
    assert!(
        schema.contains("type PersonalityResponse"),
        "Schema should include PersonalityResponse definition"
    );

    // ✅ AppearanceResponse should have all fields
    assert!(
        schema.contains("age: string"),
        "AppearanceResponse should have 'age' field"
    );
    assert!(
        schema.contains("gender: string"),
        "AppearanceResponse should have 'gender' field"
    );
    assert!(
        schema.contains("height_cm: number"),
        "AppearanceResponse should have 'height_cm' field"
    );

    // ✅ ProfileResponse should reference nested types
    assert!(
        schema.contains("appearance: AppearanceResponse"),
        "ProfileResponse should reference AppearanceResponse"
    );
    assert!(
        schema.contains("personality: PersonalityResponse"),
        "ProfileResponse should reference PersonalityResponse"
    );

    // ✅ Main type should be ProfileResponse
    assert!(
        schema.contains("type ProfileResponse"),
        "Schema should include ProfileResponse definition"
    );
}

#[test]
fn test_nested_type_definitions_come_first() {
    let schema = ProfileResponse::prompt_schema();

    // Find positions
    let appearance_pos = schema.find("type AppearanceResponse").unwrap();
    let personality_pos = schema.find("type PersonalityResponse").unwrap();
    let profile_pos = schema.find("type ProfileResponse").unwrap();

    // ✅ Nested types should come before main type
    assert!(
        appearance_pos < profile_pos,
        "AppearanceResponse definition should come before ProfileResponse"
    );
    assert!(
        personality_pos < profile_pos,
        "PersonalityResponse definition should come before ProfileResponse"
    );
}
