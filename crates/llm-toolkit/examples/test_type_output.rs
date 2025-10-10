use llm_toolkit::{ToPrompt, TypeMarker};
use serde::{Deserialize, Serialize};

// With manual __type field
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

impl Default for ResponseWithType {
    fn default() -> Self {
        Self {
            __type: default_response_type(),
            message: "Example message".to_string(),
        }
    }
}

// Without __type field but with type_marker attribute
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
#[prompt(mode = "full", type_marker)]
pub struct ResponseWithoutType {
    pub message: String,
}

impl Default for ResponseWithoutType {
    fn default() -> Self {
        Self {
            message: "Example message".to_string(),
        }
    }
}

fn main() {
    println!("=== Case 1: schema_only ===");
    println!(
        "{}",
        ResponseWithType::default().to_prompt_with_mode("schema_only")
    );

    println!("\n=== Case 1: example_only ===");
    println!(
        "{}",
        ResponseWithType::default().to_prompt_with_mode("example_only")
    );

    println!("\n=== Case 2: schema_only ===");
    println!(
        "{}",
        ResponseWithoutType::default().to_prompt_with_mode("schema_only")
    );

    println!("\n=== Case 2: example_only ===");
    println!(
        "{}",
        ResponseWithoutType::default().to_prompt_with_mode("example_only")
    );
}
