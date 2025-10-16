//! Externally Tagged enum JSON serialization/deserialization test suite
//!
//! This test suite ensures that LLM-generated JSON strictly adheres to Serde's
//! default "Externally Tagged" enum representation format.
//!
//! ## Critical Context
//!
//! Our system has **completely migrated** from the incorrect "Internally Tagged"
//! format to the correct "Externally Tagged" format.
//!
//! - ❌ OLD (FORBIDDEN): `{ "type": "Multiple", "x": 1.0 }`
//! - ✅ NEW (REQUIRED): `{ "Multiple": { "x": 1.0 } }`
//!
//! ## JSON Output Rules
//!
//! 1. **Unit variants** (`Start`, `End`):
//!    - Output as plain JSON string: `"Start"`
//!
//! 2. **Struct variants** (`Single`, `Multiple`):
//!    - Output as JSON object with variant name as the only key
//!    - Value is a nested object containing fields
//!    - Example: `{ "Multiple": { "x": 10.5, "y": 20.0, "z": 5.2 } }`
//!
//! 3. **Tuple variants** (`Tuple`):
//!    - Output as JSON object with variant name as the only key
//!    - Value is a JSON array containing tuple elements
//!    - Example: `{ "Tuple": [10.0, 20.0] }`

use serde::{Deserialize, Serialize};

/// Action enum representing various command types with Externally Tagged format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)] // Strict: reject any unexpected fields
pub enum Action {
    /// Unit variant: Start a process
    Start,

    /// Unit variant: End a process
    End,

    /// Struct variant with single field
    Single { value: String },

    /// Struct variant with multiple fields
    Multiple { x: f64, y: f64, z: f64 },

    /// Tuple variant with two f64 values
    Tuple(f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Unit variant "Start"
    /// Input instruction: "Start the process"
    /// Expected: "Start"
    #[test]
    fn test_unit_variant_start() {
        let json = r##""Start""##;

        let deserialized: Action =
            serde_json::from_str(json).expect("Failed to deserialize Start variant");

        let expected = Action::Start;

        assert_eq!(
            deserialized, expected,
            "Start variant must deserialize to Action::Start"
        );

        // Verify round-trip serialization
        let serialized =
            serde_json::to_string(&deserialized).expect("Failed to serialize Start variant");
        assert_eq!(serialized, json, "Round-trip serialization must match");
    }

    /// Test 2: Unit variant "End"
    /// Input instruction: "End the process"
    /// Expected: "End"
    #[test]
    fn test_unit_variant_end() {
        let json = r##""End""##;

        let deserialized: Action =
            serde_json::from_str(json).expect("Failed to deserialize End variant");

        let expected = Action::End;

        assert_eq!(
            deserialized, expected,
            "End variant must deserialize to Action::End"
        );

        // Verify round-trip serialization
        let serialized =
            serde_json::to_string(&deserialized).expect("Failed to serialize End variant");
        assert_eq!(serialized, json, "Round-trip serialization must match");
    }

    /// Test 3: Struct variant "Single" with one field
    /// Input instruction: "Send message 'Welcome' to user"
    /// Expected: { "Single": { "value": "Welcome" } }
    #[test]
    fn test_struct_variant_single() {
        let json = r##"{
  "Single": {
    "value": "Welcome"
  }
}"##;

        let deserialized: Action =
            serde_json::from_str(json).expect("Failed to deserialize Single variant");

        let expected = Action::Single {
            value: "Welcome".to_string(),
        };

        assert_eq!(
            deserialized, expected,
            "Single variant must deserialize with exact field values"
        );

        // Verify structure
        match deserialized {
            Action::Single { value } => {
                assert_eq!(value, "Welcome", "Single.value must match exactly");
            }
            _ => panic!("Expected Action::Single variant"),
        }
    }

    /// Test 4: Struct variant "Multiple" with multiple fields
    /// Input instruction: "Set coordinates to x=10.5, y=20.0, z=5.2"
    /// Expected: { "Multiple": { "x": 10.5, "y": 20.0, "z": 5.2 } }
    #[test]
    fn test_struct_variant_multiple() {
        let json = r##"{
  "Multiple": {
    "x": 10.5,
    "y": 20.0,
    "z": 5.2
  }
}"##;

        let deserialized: Action =
            serde_json::from_str(json).expect("Failed to deserialize Multiple variant");

        let expected = Action::Multiple {
            x: 10.5,
            y: 20.0,
            z: 5.2,
        };

        assert_eq!(
            deserialized, expected,
            "Multiple variant must deserialize with exact field values"
        );

        // Verify each field individually
        match deserialized {
            Action::Multiple { x, y, z } => {
                assert_eq!(x, 10.5, "Multiple.x must be 10.5");
                assert_eq!(y, 20.0, "Multiple.y must be 20.0");
                assert_eq!(z, 5.2, "Multiple.z must be 5.2");
            }
            _ => panic!("Expected Action::Multiple variant"),
        }
    }

    /// Test 5: Tuple variant "Tuple"
    /// Input instruction: "Set 2D vector to (10.0, 20.0)"
    /// Expected: { "Tuple": [10.0, 20.0] }
    #[test]
    fn test_tuple_variant() {
        let json = r##"{
  "Tuple": [
    10.0,
    20.0
  ]
}"##;

        let deserialized: Action =
            serde_json::from_str(json).expect("Failed to deserialize Tuple variant");

        let expected = Action::Tuple(10.0, 20.0);

        assert_eq!(
            deserialized, expected,
            "Tuple variant must deserialize with exact tuple values"
        );

        // Verify each tuple element individually
        match deserialized {
            Action::Tuple(first, second) => {
                assert_eq!(first, 10.0, "Tuple.0 must be 10.0");
                assert_eq!(second, 20.0, "Tuple.1 must be 20.0");
            }
            _ => panic!("Expected Action::Tuple variant"),
        }
    }

    /// Test 6: Negative test - Reject forbidden "Internally Tagged" format
    /// Input instruction: "Set coordinates to x=1.0, y=1.0, z=1.0"
    ///
    /// This test ensures that the OLD, FORBIDDEN format is properly rejected:
    /// ❌ { "type": "Multiple", "x": 1.0, "y": 1.0, "z": 1.0 }
    #[test]
    fn test_reject_internally_tagged_format() {
        // The OLD, FORBIDDEN "Internally Tagged" format
        let forbidden_json = r##"{
  "type": "Multiple",
  "x": 1.0,
  "y": 1.0,
  "z": 1.0
}"##;

        let result: Result<Action, _> = serde_json::from_str(forbidden_json);

        assert!(
            result.is_err(),
            "Internally Tagged format MUST be rejected. \
             This format is forbidden and should fail to deserialize."
        );

        // Verify the correct format works
        let correct_json = r##"{
  "Multiple": {
    "x": 1.0,
    "y": 1.0,
    "z": 1.0
  }
}"##;

        let result: Result<Action, _> = serde_json::from_str(correct_json);
        assert!(result.is_ok(), "Externally Tagged format MUST be accepted");

        let deserialized = result.unwrap();
        assert_eq!(
            deserialized,
            Action::Multiple {
                x: 1.0,
                y: 1.0,
                z: 1.0
            },
            "Correct format must deserialize to expected value"
        );
    }

    /// Additional strict test: Verify unknown fields are rejected
    #[test]
    fn test_reject_unknown_fields() {
        let json_with_extra_field = r##"{
  "Multiple": {
    "x": 1.0,
    "y": 2.0,
    "z": 3.0,
    "unknown": "field"
  }
}"##;

        let result: Result<Action, _> = serde_json::from_str(json_with_extra_field);

        assert!(
            result.is_err(),
            "Unknown fields must be rejected due to deny_unknown_fields"
        );
    }

    /// Additional strict test: Verify type safety for field types
    #[test]
    fn test_reject_wrong_field_types() {
        // String instead of f64
        let json_wrong_type = r##"{
  "Multiple": {
    "x": "not a number",
    "y": 2.0,
    "z": 3.0
  }
}"##;

        let result: Result<Action, _> = serde_json::from_str(json_wrong_type);

        assert!(result.is_err(), "Wrong field types must be rejected");
    }

    /// Additional strict test: Verify missing required fields are rejected
    #[test]
    fn test_reject_missing_fields() {
        let json_missing_field = r##"{
  "Multiple": {
    "x": 1.0,
    "y": 2.0
  }
}"##;

        let result: Result<Action, _> = serde_json::from_str(json_missing_field);

        assert!(result.is_err(), "Missing required fields must be rejected");
    }

    /// Comprehensive round-trip test for all variants
    #[test]
    fn test_all_variants_round_trip() {
        let test_cases = vec![
            Action::Start,
            Action::End,
            Action::Single {
                value: "test".to_string(),
            },
            Action::Multiple {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            Action::Tuple(4.0, 5.0),
        ];

        for original in test_cases {
            let json = serde_json::to_string(&original).expect("Serialization must succeed");

            let deserialized: Action =
                serde_json::from_str(&json).expect("Deserialization must succeed");

            assert_eq!(
                original, deserialized,
                "Round-trip serialization must preserve exact values"
            );
        }
    }
}
