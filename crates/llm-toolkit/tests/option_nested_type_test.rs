//! Test for Option<T> with nested types expansion

#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    #[allow(clippy::enum_variant_names)]
    pub enum CameraAngle {
        FrontView,
        SideView,
        TopView,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum CameraDistance {
        Close,
        Medium,
        Far,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, ToPrompt)]
    pub enum CameraMovement {
        Static,
        Pan,
        Tilt,
        Zoom,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, ToPrompt)]
    #[prompt(mode = "full")]
    pub struct CameraWork {
        pub angle: CameraAngle,
        pub distance: CameraDistance,
        pub movement: Option<CameraMovement>,
    }

    #[test]
    fn test_option_nested_type_expansion() {
        let schema = CameraWork::prompt_schema();
        println!(
            "\n=== CameraWork schema ===\n{}\n=========================",
            schema
        );

        // Should reference nested types even when wrapped in Option
        assert!(
            schema.contains("CameraAngle"),
            "Schema should reference CameraAngle type"
        );
        assert!(
            schema.contains("CameraDistance"),
            "Schema should reference CameraDistance type"
        );
        assert!(
            schema.contains("CameraMovement"),
            "Schema should reference CameraMovement type even when Optional"
        );

        // Should include nested enum definitions
        assert!(
            schema.contains("FrontView") || schema.contains("\"FrontView\""),
            "Schema should include CameraAngle::FrontView"
        );
        assert!(
            schema.contains("Close") || schema.contains("\"Close\""),
            "Schema should include CameraDistance::Close"
        );
        assert!(
            schema.contains("Static") || schema.contains("\"Static\""),
            "Schema should include CameraMovement::Static"
        );

        // Optional field should be marked with | null
        assert!(
            schema.contains("| null") || schema.contains("?"),
            "Optional field should be marked as nullable"
        );
    }

    #[test]
    fn test_option_primitive_type() {
        #[derive(Debug, Clone, Serialize, Deserialize, ToPrompt)]
        #[prompt(mode = "full")]
        pub struct Config {
            pub name: String,
            pub count: Option<u32>,
        }

        let schema = Config::prompt_schema();
        println!("\n=== Config schema ===\n{}\n=====================", schema);

        // Primitive optional fields should work correctly
        assert!(
            schema.contains("count"),
            "Schema should contain count field"
        );
        assert!(
            schema.contains("| null") || schema.contains("number"),
            "Optional number field should be marked as nullable or number"
        );
    }
}
