//! Generate and save JSON Schema for Expertise type

use llm_toolkit::agent::expertise::Expertise;
use schemars::schema_for;

fn main() {
    println!("=== Expertise JSON Schema Generator ===\n");

    // Generate schema
    let schema = schema_for!(Expertise);
    let schema_value = serde_json::to_value(&schema).expect("Failed to serialize schema");

    // Print to stdout
    println!("--- JSON Schema ---\n");
    println!(
        "{}",
        serde_json::to_string_pretty(&schema_value).expect("Failed to serialize schema")
    );

    // Optionally save to file
    // Uncomment to save:
    /*
    let output_path = "expertise-schema.json";
    let json = serde_json::to_string_pretty(&schema_value)?;
    std::fs::write(output_path, json)?;
    println!("\n✅ Schema saved to: {}", output_path);
    */

    println!("\n--- Schema Stats ---");
    if let Some(obj) = schema_value.as_object() {
        println!("Top-level keys: {}", obj.keys().count());
        if let Some(defs) = obj.get("definitions").and_then(|v| v.as_object()) {
            println!("Type definitions: {}", defs.keys().count());
            println!("Defined types:");
            for key in defs.keys() {
                println!("  - {}", key);
            }
        }
    }
}
