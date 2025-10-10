// Test to verify TypeMarker behavior with ToPrompt schema generation
use llm_toolkit::{ToPrompt, TypeMarker};
use serde::{Deserialize, Serialize};

// Case 1: With manual __type field (current expected usage)
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
#[prompt(mode = "full")]
pub struct ResponseWithType {
    #[serde(default = "default_response_type")]
    __type: String,
    pub message: String,
}

fn default_response_type() -> String {
    "ResponseWithType".to_string()
}

// Case 2: Without __type field but with type_marker attribute
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
#[prompt(mode = "full", type_marker)]
pub struct ResponseWithoutType {
    pub message: String,
}

fn main() {
    println!("=== Case 1: With manual __type field ===");
    let schema1 = ResponseWithType::prompt_schema();
    println!("{}", schema1);
    println!("\nChecking for __type in schema:");
    if schema1.contains("__type") {
        println!("✅ __type found in schema");
    } else {
        println!("❌ __type NOT found in schema");
    }

    println!("\n=== Case 2: With type_marker attribute ===");
    let schema2 = ResponseWithoutType::prompt_schema();
    println!("{}", schema2);
    println!("\nChecking for __type in schema:");
    if schema2.contains("__type") {
        println!("✅ __type found in schema (automatically added by type_marker attribute)");
    } else {
        println!("❌ __type NOT found in schema");
    }

    println!("\n=== TypeMarker trait implementation ===");
    println!(
        "ResponseWithType::TYPE_NAME = {}",
        ResponseWithType::TYPE_NAME
    );
    println!(
        "ResponseWithoutType::TYPE_NAME = {}",
        ResponseWithoutType::TYPE_NAME
    );
}
