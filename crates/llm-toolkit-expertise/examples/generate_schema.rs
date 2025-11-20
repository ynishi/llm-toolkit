//! Generate and save JSON Schema for Expertise type

use llm_toolkit_expertise::dump_expertise_schema;

fn main() {
    println!("=== Expertise JSON Schema Generator ===\n");

    // Generate schema
    let schema = dump_expertise_schema();

    // Print to stdout
    println!("--- JSON Schema ---\n");
    println!(
        "{}",
        serde_json::to_string_pretty(&schema).expect("Failed to serialize schema")
    );

    // Optionally save to file
    // Uncomment to save:
    /*
    let output_path = "expertise-schema.json";
    match save_expertise_schema(output_path) {
        Ok(_) => println!("\n✅ Schema saved to: {}", output_path),
        Err(e) => eprintln!("\n❌ Failed to save schema: {}", e),
    }
    */

    println!("\n--- Schema Stats ---");
    if let Some(obj) = schema.as_object() {
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
