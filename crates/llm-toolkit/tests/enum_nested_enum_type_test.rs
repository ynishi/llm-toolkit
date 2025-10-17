//! Test for nested enum types in enum struct variants

#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum PanDirection {
        Left,
        Right,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum TiltDirection {
        Up,
        Down,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum DollyDirection {
        In,
        Out,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum ZoomDirection {
        In,
        Out,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum CameraMovement {
        Static,
        Pan { direction: PanDirection },
        Tilt { direction: TiltDirection },
        Dolly { direction: DollyDirection },
        Zoom { direction: ZoomDirection },
    }

    #[test]
    fn test_nested_enum_in_enum_variant() {
        let schema = CameraMovement::prompt_schema();
        println!(
            "\n=== CameraMovement schema ===\n{}\n=============================",
            schema
        );

        // Should reference nested enum types
        assert!(
            schema.contains("PanDirection"),
            "Schema should reference PanDirection type"
        );
        assert!(
            schema.contains("TiltDirection"),
            "Schema should reference TiltDirection type"
        );
        assert!(
            schema.contains("DollyDirection"),
            "Schema should reference DollyDirection type"
        );
        assert!(
            schema.contains("ZoomDirection"),
            "Schema should reference ZoomDirection type"
        );

        // Should include nested enum definitions
        assert!(
            schema.contains("\"Left\"") || schema.contains("Left"),
            "Schema should include PanDirection::Left"
        );
        assert!(
            schema.contains("\"Right\"") || schema.contains("Right"),
            "Schema should include PanDirection::Right"
        );
        assert!(
            schema.contains("\"Up\"") || schema.contains("Up"),
            "Schema should include TiltDirection::Up"
        );
        assert!(
            schema.contains("\"Down\"") || schema.contains("Down"),
            "Schema should include TiltDirection::Down"
        );
    }
}
