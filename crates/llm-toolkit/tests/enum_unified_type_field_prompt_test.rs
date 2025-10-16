//! ToPrompt output test for Externally Tagged enum
//!
//! This test suite verifies that the `Action` enum generates appropriate
//! TypeScript-compatible prompt output for LLMs, ensuring the Externally Tagged
//! JSON format is correctly communicated.
//!
//! ## Critical Requirements
//!
//! The prompt output must guide LLMs to generate JSON in Externally Tagged format:
//! - Unit variants: `"Start"`
//! - Struct variants: `{ "Multiple": { "x": 1.0, "y": 2.0, "z": 3.0 } }`
//! - Tuple variants: `{ "Tuple": [10.0, 20.0] }`
//!
//! The prompt must NOT suggest the forbidden Internally Tagged format:
//! - ❌ FORBIDDEN: `{ "type": "Multiple", "x": 1.0 }`

use llm_toolkit::ToPrompt;
use serde::{Deserialize, Serialize};

/// Action enum representing various command types with Externally Tagged format
/// NOTE: No serde(tag) attribute = default Externally Tagged format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToPrompt)]
#[serde(deny_unknown_fields)]
pub enum Action {
    /// Start a process
    Start,

    /// End a process
    End,

    /// Send a message with single value
    Single { value: String },

    /// Set coordinates with multiple fields
    Multiple { x: f64, y: f64, z: f64 },

    /// Set 2D vector using tuple
    Tuple(f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Verify prompt_schema() generates exact expected output
    #[test]
    fn test_action_prompt_schema_format() {
        let schema = Action::prompt_schema();
        println!("\n=== Action::prompt_schema() output ===");
        println!("{}", schema);
        println!("=========================================\n");

        let expected = r#"/**
 * Action enum representing various command types with Externally Tagged format NOTE: No serde(tag) attribute = default Externally Tagged format
 */
type Action =
  | "Start"  // Start a process
  | "End"  // End a process
  | { "Single": { value: string } }  // Send a message with single value
  | { "Multiple": { x: number, y: number, z: number } }  // Set coordinates with multiple fields
  | { "Tuple": [number, number] }  // Set 2D vector using tuple;

Example values:
  "Start"
  { "Single": { value: "example" } }
  { "Tuple": [0, 0] }"#;

        assert_eq!(
            schema, expected,
            "Schema must match expected Externally Tagged format exactly"
        );
    }

    /// Test 2: Verify no Internally Tagged format is present (negative test)
    #[test]
    fn test_rejects_internally_tagged_format() {
        let schema = Action::prompt_schema();

        // Must NOT contain Internally Tagged patterns
        assert!(
            !schema.contains(r#"type: "Single""#),
            "Schema must NOT use Internally Tagged format (type: \"Single\")"
        );
        assert!(
            !schema.contains(r#"type: "Multiple""#),
            "Schema must NOT use Internally Tagged format (type: \"Multiple\")"
        );
        assert!(
            !schema.contains(r#"type: "Tuple""#),
            "Schema must NOT use Internally Tagged format (type: \"Tuple\")"
        );
    }

    /// Test 3: Verify Tuple variant with actual JSON round-trip
    ///
    /// Validates that the schema correctly represents how Tuple variants
    /// are actually serialized in Externally Tagged format.
    #[test]
    fn test_tuple_variant_schema_externally_tagged() {
        let schema = Action::prompt_schema();

        // Show actual JSON format
        let tuple_instance = Action::Tuple(10.0, 20.0);
        let json = serde_json::to_string_pretty(&tuple_instance).unwrap();
        println!("\n=== Tuple variant JSON (Serde output) ===");
        println!("{}", json);
        println!("==========================================\n");

        // Expected JSON: { "Tuple": [10.0, 20.0] }
        let expected_json = r##"{
  "Tuple": [
    10.0,
    20.0
  ]
}"##;

        assert_eq!(
            json, expected_json,
            "Tuple variant must serialize to Externally Tagged format"
        );

        // Schema must represent this format
        assert!(
            schema.contains(r#"{ "Tuple": [number, number] }"#),
            "Schema must show Tuple in Externally Tagged format: {{ \"Tuple\": [number, number] }}"
        );

        // Must NOT be bare array
        assert!(
            !schema.contains(r#"| [number, number]  // Set 2D vector"#),
            "Schema must NOT show Tuple as bare array (that would be Untagged format)"
        );
    }

    /// Test 4: Verify instance to_prompt() for Unit variants
    #[test]
    fn test_unit_variant_to_prompt() {
        let start = Action::Start;
        let prompt = start.to_prompt();
        assert_eq!(prompt, "Start: Start a process");

        let end = Action::End;
        let prompt = end.to_prompt();
        assert_eq!(prompt, "End: End a process");
    }

    /// Test 5: Verify instance to_prompt() for Struct variant (Single field)
    #[test]
    fn test_struct_single_variant_to_prompt() {
        let single = Action::Single {
            value: "Welcome".to_string(),
        };
        let prompt = single.to_prompt();
        assert_eq!(
            prompt,
            r#"Single: Send a message with single value { value: "Welcome" }"#
        );
    }

    /// Test 6: Verify instance to_prompt() for Struct variant (Multiple fields)
    #[test]
    fn test_struct_multiple_variant_to_prompt() {
        let multiple = Action::Multiple {
            x: 10.5,
            y: 20.0,
            z: 5.2,
        };
        let prompt = multiple.to_prompt();
        assert_eq!(
            prompt,
            "Multiple: Set coordinates with multiple fields { x: 10.5, y: 20.0, z: 5.2 }"
        );
    }

    /// Test 7: Verify instance to_prompt() for Tuple variant
    #[test]
    fn test_tuple_variant_to_prompt() {
        let tuple = Action::Tuple(10.0, 20.0);
        let prompt = tuple.to_prompt();
        assert_eq!(prompt, "Tuple: Set 2D vector using tuple(10.0, 20.0)");
    }

    /// Test 8: Verify JSON serialization examples are in Externally Tagged format (exact match)
    #[test]
    fn test_json_examples_are_externally_tagged() {
        // Unit variant
        let start = Action::Start;
        let json = serde_json::to_string_pretty(&start).unwrap();
        assert_eq!(json, r#""Start""#);

        // Struct variant - Single
        let single = Action::Single {
            value: "Welcome".to_string(),
        };
        let json = serde_json::to_string_pretty(&single).unwrap();
        let expected = r##"{
  "Single": {
    "value": "Welcome"
  }
}"##;
        assert_eq!(json, expected);

        // Struct variant - Multiple
        let multiple = Action::Multiple {
            x: 10.5,
            y: 20.0,
            z: 5.2,
        };
        let json = serde_json::to_string_pretty(&multiple).unwrap();
        let expected = r##"{
  "Multiple": {
    "x": 10.5,
    "y": 20.0,
    "z": 5.2
  }
}"##;
        assert_eq!(json, expected);

        // Tuple variant
        let tuple = Action::Tuple(10.0, 20.0);
        let json = serde_json::to_string_pretty(&tuple).unwrap();
        let expected = r##"{
  "Tuple": [
    10.0,
    20.0
  ]
}"##;
        assert_eq!(json, expected);
    }

    /// Test 9: Round-trip: Schema → JSON → Deserialize (exact match)
    /// Verify that following the schema produces valid JSON that deserializes correctly
    #[test]
    fn test_schema_guides_to_valid_json() {
        // Unit variant
        let json_start = r#""Start""#;
        let deserialized: Action = serde_json::from_str(json_start).unwrap();
        assert_eq!(deserialized, Action::Start);

        // Struct variant
        let json_multiple = r##"{
  "Multiple": {
    "x": 1.0,
    "y": 2.0,
    "z": 3.0
  }
}"##;
        let deserialized: Action = serde_json::from_str(json_multiple).unwrap();
        assert_eq!(
            deserialized,
            Action::Multiple {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );

        // Tuple variant
        let json_tuple = r##"{
  "Tuple": [5.0, 10.0]
}"##;
        let deserialized: Action = serde_json::from_str(json_tuple).unwrap();
        assert_eq!(deserialized, Action::Tuple(5.0, 10.0));
    }
}
