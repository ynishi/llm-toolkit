#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;
    use serde::{Deserialize, Serialize};

    // Basic tuple variant test
    #[derive(ToPrompt, Serialize, Deserialize, Debug, PartialEq)]
    #[serde(untagged)]
    enum Coordinate {
        /// 2D coordinate
        Point2D(f64, f64),
        /// 3D coordinate
        Point3D(f64, f64, f64),
        /// Origin point
        Origin,
    }

    #[test]
    fn test_tuple_variant_schema() {
        let schema = Coordinate::prompt_schema();

        // Should be TypeScript tuple format
        assert!(schema.contains("type Coordinate ="));

        // Unit variant should remain simple
        assert!(schema.contains("| \"Origin\""));

        // Tuple variants should be arrays
        assert!(schema.contains("| [number, number]"));
        assert!(schema.contains("| [number, number, number]"));

        // Should have doc comments
        assert!(schema.contains("// 2D coordinate"));
        assert!(schema.contains("// 3D coordinate"));
        assert!(schema.contains("// Origin point"));
    }

    #[test]
    fn test_tuple_variant_instance_to_prompt() {
        // Unit variant
        let origin = Coordinate::Origin;
        let prompt = origin.to_prompt();
        assert_eq!(prompt, "Origin: Origin point");

        // Tuple variant
        let point2d = Coordinate::Point2D(10.5, 20.3);
        let prompt = point2d.to_prompt();

        // Should show variant name and values
        assert!(prompt.contains("Point2D"));
        assert!(prompt.contains("10.5"));
        assert!(prompt.contains("20.3"));
    }

    #[test]
    fn test_tuple_variant_serde_roundtrip() {
        let original = Coordinate::Point2D(1.0, 2.0);

        // Serialize
        let json = serde_json::to_string(&original).unwrap();

        // Should use untagged format (array)
        assert!(json.contains("[1.0,2.0]") || json.contains("[1.0, 2.0]"));

        // Deserialize
        let deserialized: Coordinate = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    // Mixed types in tuple
    #[derive(ToPrompt, Serialize, Deserialize, Debug, PartialEq)]
    #[serde(untagged)]
    enum Value {
        /// String-int pair
        Pair(String, i32),
        /// Single string
        Single(String),
        /// Triple with bool
        Triple(String, i32, bool),
    }

    #[test]
    fn test_mixed_type_tuple_schema() {
        let schema = Value::prompt_schema();

        // TypeScript format
        assert!(schema.contains("type Value ="));

        // Different tuple types
        assert!(schema.contains("| [string, number]"));
        assert!(schema.contains("| [string]"));
        assert!(schema.contains("| [string, number, boolean]"));

        // Doc comments
        assert!(schema.contains("// String-int pair"));
        assert!(schema.contains("// Single string"));
        assert!(schema.contains("// Triple with bool"));
    }

    #[test]
    fn test_tuple_variant_with_rename() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(untagged, rename_all = "snake_case")]
        enum Data {
            #[prompt(rename = "xy_coord")]
            XYCoordinate(f64, f64),
            ZCoordinate(f64),
        }

        let schema = Data::prompt_schema();

        // Should use prompt rename for XYCoordinate
        assert!(schema.contains("xy_coord"));

        // Type should still show tuple format
        assert!(schema.contains("[number, number]"));
        assert!(schema.contains("[number]"));
    }

    #[test]
    fn test_tuple_variant_with_skip() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(untagged)]
        enum Measurement {
            Distance(f64, f64),
            #[prompt(skip)]
            InternalDebug(String, i32),
        }

        let schema = Measurement::prompt_schema();

        // Distance should be present
        assert!(schema.contains("Distance"));
        assert!(schema.contains("[number, number]"));

        // InternalDebug should be skipped
        assert!(!schema.contains("InternalDebug"));
        assert!(!schema.contains("string, number"));
    }

    #[test]
    fn test_tuple_variant_single_element() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(untagged)]
        enum Wrapper {
            Value(i32),
            Text(String),
        }

        let schema = Wrapper::prompt_schema();

        // Single element tuples
        assert!(schema.contains("| [number]"));
        assert!(schema.contains("| [string]"));
    }

    #[test]
    fn test_tuple_variant_complex_types() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(untagged)]
        enum Complex {
            VecPair(Vec<String>, Vec<i32>),
            OptionalPair(Option<String>, i32),
        }

        let schema = Complex::prompt_schema();

        // Complex types in tuples
        assert!(schema.contains("string[]"));
        assert!(schema.contains("number[]"));
        assert!(schema.contains("string | null"));
    }

    // Mix of all variant types
    #[derive(ToPrompt, Serialize, Deserialize, Debug, PartialEq)]
    #[serde(untagged)]
    enum MixedEnum {
        /// Simple unit
        None,
        /// Tuple variant
        Pair(String, i32),
        // Note: Struct variants would need different serde tag (not shown here)
    }

    #[test]
    fn test_mixed_unit_and_tuple_schema() {
        let schema = MixedEnum::prompt_schema();

        // TypeScript format
        assert!(schema.contains("type MixedEnum ="));

        // Unit variant
        assert!(schema.contains("| \"None\""));

        // Tuple variant
        assert!(schema.contains("| [string, number]"));

        // Doc comments
        assert!(schema.contains("// Simple unit"));
        assert!(schema.contains("// Tuple variant"));
    }

    #[test]
    fn test_tuple_variant_instance_display() {
        let pair = MixedEnum::Pair("hello".to_string(), 42);
        let prompt = pair.to_prompt();

        // Should show values
        assert!(prompt.contains("Pair"));
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("42"));
    }

    #[test]
    fn test_tuple_variant_large_tuple() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(untagged)]
        enum LargeTuple {
            /// Five element tuple
            Quintuple(i32, i32, i32, i32, i32),
        }

        let schema = LargeTuple::prompt_schema();

        // Should handle large tuples
        assert!(schema.contains("[number, number, number, number, number]"));
        assert!(schema.contains("// Five element tuple"));
    }
}
