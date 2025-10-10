// Test to verify TypeMarker behavior with ToPrompt schema generation
use llm_toolkit::{ToPrompt, TypeMarker, type_marker};
use serde::{Deserialize, Serialize};

// Case 1: With manual __type field
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
#[prompt(mode = "full")]
pub struct ResponseWithManualType {
    #[serde(default = "default_response_type")]
    __type: String,
    pub message: String,
}

fn default_response_type() -> String {
    "ResponseWithManualType".to_string()
}

// Case 2: Using #[type_marker] attribute macro (recommended)
#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct ResponseWithTypeMacro {
    pub message: String,
}

// Case 3: Using #[prompt(type_marker)] with manual __type field
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
#[prompt(mode = "full", type_marker)]
pub struct ResponseWithPromptTypeMarker {
    #[serde(default = "default_response_prompt_type_marker")]
    __type: String,
    pub message: String,
}

fn default_response_prompt_type_marker() -> String {
    "ResponseWithPromptTypeMarker".to_string()
}

// Case 4: ERROR CASE - Using #[prompt(type_marker)] WITHOUT __type field
// Uncommenting this will cause a compile error with helpful message:
// #[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
// #[prompt(mode = "full", type_marker)]
// pub struct ResponseWithoutTypeField {
//     pub message: String,
// }

fn main() {
    println!("=== Case 1: With manual __type field ===");
    let schema1 = ResponseWithManualType::prompt_schema();
    println!("{}", schema1);
    println!("\nChecking for __type in schema:");
    if schema1.contains("__type") {
        println!("❌ UNEXPECTED: __type found in schema (should be auto-excluded)");
        std::process::exit(1);
    } else {
        println!("✅ __type correctly excluded from schema");
    }

    println!("\n=== Case 2: Using #[type_marker] attribute macro ===");
    let schema2 = ResponseWithTypeMacro::prompt_schema();
    println!("{}", schema2);
    println!("\nChecking for __type in schema:");
    if schema2.contains("__type") {
        println!("❌ UNEXPECTED: __type found in schema (should be auto-excluded)");
        std::process::exit(1);
    } else {
        println!("✅ __type correctly excluded from schema");
    }

    println!("\n=== Case 3: Using #[prompt(type_marker)] with manual field ===");
    let schema3 = ResponseWithPromptTypeMarker::prompt_schema();
    println!("{}", schema3);
    println!("\nChecking for __type in schema:");
    if schema3.contains("__type") {
        println!("❌ UNEXPECTED: __type found in schema (should be auto-excluded)");
        std::process::exit(1);
    } else {
        println!("✅ __type correctly excluded from schema");
    }

    println!("\n=== TypeMarker trait implementation ===");
    println!(
        "ResponseWithManualType::TYPE_NAME = {}",
        ResponseWithManualType::TYPE_NAME
    );
    println!(
        "ResponseWithTypeMacro::TYPE_NAME = {}",
        ResponseWithTypeMacro::TYPE_NAME
    );
    println!(
        "ResponseWithPromptTypeMarker::TYPE_NAME = {}",
        ResponseWithPromptTypeMarker::TYPE_NAME
    );

    println!("\n✅ All cases passed!");
}
