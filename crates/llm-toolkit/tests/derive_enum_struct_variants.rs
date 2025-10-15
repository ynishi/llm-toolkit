#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;
    use serde::{Deserialize, Serialize};

    // Basic struct variant test
    #[derive(ToPrompt, Serialize, Deserialize, Debug, PartialEq)]
    #[serde(tag = "type")]
    enum AnalysisResult {
        /// Analysis approved
        Approved,
        /// Analysis needs revision
        NeedsRevision {
            reasons: Vec<String>,
            severity: String,
        },
        /// Analysis rejected
        Rejected { reason: String },
    }

    #[test]
    fn test_struct_variant_schema() {
        let schema = AnalysisResult::prompt_schema();

        // Should be TypeScript tagged union format
        assert!(schema.contains("type AnalysisResult ="));

        // Unit variant should remain simple
        assert!(schema.contains("| \"Approved\""));

        // Struct variants should be objects with type field
        assert!(schema.contains("| { type: \"NeedsRevision\""));
        assert!(schema.contains("reasons: string[]"));
        assert!(schema.contains("severity: string"));

        assert!(schema.contains("| { type: \"Rejected\""));
        assert!(schema.contains("reason: string"));

        // Should have doc comments
        assert!(schema.contains("// Analysis approved"));
        assert!(schema.contains("// Analysis needs revision"));
        assert!(schema.contains("// Analysis rejected"));
    }

    #[test]
    fn test_struct_variant_instance_to_prompt() {
        // Unit variant
        let approved = AnalysisResult::Approved;
        let prompt = approved.to_prompt();
        assert_eq!(prompt, "Approved: Analysis approved");

        // Struct variant
        let needs_revision = AnalysisResult::NeedsRevision {
            reasons: vec!["Missing data".to_string(), "Invalid format".to_string()],
            severity: "High".to_string(),
        };
        let prompt = needs_revision.to_prompt();

        // Should show variant name and fields
        assert!(prompt.contains("NeedsRevision"));
        assert!(prompt.contains("reasons"));
        assert!(prompt.contains("Missing data"));
        assert!(prompt.contains("severity"));
        assert!(prompt.contains("High"));
    }

    #[test]
    fn test_struct_variant_serde_roundtrip() {
        let original = AnalysisResult::NeedsRevision {
            reasons: vec!["Issue 1".to_string()],
            severity: "Medium".to_string(),
        };

        // Serialize
        let json = serde_json::to_string(&original).unwrap();

        // Should use serde(tag = "type") format
        assert!(json.contains("\"type\":\"NeedsRevision\""));
        assert!(json.contains("\"reasons\""));
        assert!(json.contains("\"severity\""));

        // Deserialize
        let deserialized: AnalysisResult = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    // Complex struct variant with nested types
    #[derive(ToPrompt, Serialize, Deserialize, Debug, PartialEq)]
    #[serde(tag = "type")]
    enum LightingTechnique {
        /// Chiaroscuro (dramatic high-contrast lighting)
        Chiaroscuro {
            contrast_level: String,
            light_source: String,
            shadow_direction: String,
        },
        /// Rembrandt lighting (triangle of light on cheek)
        Rembrandt {
            triangle_side: String,
            fill_ratio: f32,
        },
        /// Simple natural lighting
        Natural,
    }

    #[test]
    fn test_mixed_variants_schema() {
        let schema = LightingTechnique::prompt_schema();

        // TypeScript format
        assert!(schema.contains("type LightingTechnique ="));

        // Struct variants
        assert!(schema.contains("| { type: \"Chiaroscuro\""));
        assert!(schema.contains("contrast_level: string"));
        assert!(schema.contains("light_source: string"));
        assert!(schema.contains("shadow_direction: string"));

        assert!(schema.contains("| { type: \"Rembrandt\""));
        assert!(schema.contains("triangle_side: string"));
        assert!(schema.contains("fill_ratio: number"));

        // Unit variant
        assert!(schema.contains("| \"Natural\""));

        // Doc comments
        assert!(schema.contains("// Chiaroscuro (dramatic high-contrast lighting)"));
        assert!(schema.contains("// Rembrandt lighting (triangle of light on cheek)"));
        assert!(schema.contains("// Simple natural lighting"));
    }

    #[test]
    fn test_struct_variant_with_rename() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Action {
            #[prompt(rename = "send_msg")]
            SendMessage {
                to: String,
                content: String,
            },
            DeleteItem {
                id: u64,
            },
        }

        let schema = Action::prompt_schema();

        // Should use prompt rename for SendMessage
        assert!(schema.contains("| { type: \"send_msg\""));

        // Should use snake_case for DeleteItem (from rename_all)
        assert!(schema.contains("| { type: \"delete_item\""));

        // Field names should be present
        assert!(schema.contains("to: string"));
        assert!(schema.contains("content: string"));
        assert!(schema.contains("id: number"));
    }

    #[test]
    fn test_struct_variant_with_skip() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type")]
        enum Command {
            Execute {
                script: String,
            },
            #[prompt(skip)]
            InternalDebug {
                details: String,
            },
        }

        let schema = Command::prompt_schema();

        // Execute should be present
        assert!(schema.contains("| { type: \"Execute\""));
        assert!(schema.contains("script: string"));

        // InternalDebug should be skipped
        assert!(!schema.contains("InternalDebug"));
        assert!(!schema.contains("details"));
    }

    #[test]
    fn test_struct_variant_empty_fields() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type")]
        enum Event {
            Started,
            Progress { percent: u8 },
            Completed,
        }

        let schema = Event::prompt_schema();

        // All variants should be present
        assert!(schema.contains("| \"Started\""));
        assert!(schema.contains("| { type: \"Progress\""));
        assert!(schema.contains("percent: number"));
        assert!(schema.contains("| \"Completed\""));
    }

    #[test]
    fn test_struct_variant_with_vec() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type")]
        enum Response {
            Success { data: Vec<String> },
            Error { messages: Vec<String> },
        }

        let schema = Response::prompt_schema();

        // Should show array types
        assert!(schema.contains("data: string[]"));
        assert!(schema.contains("messages: string[]"));
    }

    #[test]
    fn test_struct_variant_with_option() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type")]
        enum UserAction {
            Login {
                username: String,
            },
            Update {
                username: String,
                email: Option<String>,
            },
        }

        let schema = UserAction::prompt_schema();

        // Option should be shown in TypeScript format
        assert!(schema.contains("email: string | null"));
    }

    #[test]
    fn test_struct_variant_number_types() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type")]
        enum Measurement {
            Temperature { celsius: f32 },
            Distance { meters: u32 },
            Count { items: i64 },
        }

        let schema = Measurement::prompt_schema();

        // All number types should map to TypeScript 'number'
        assert!(schema.contains("celsius: number"));
        assert!(schema.contains("meters: number"));
        assert!(schema.contains("items: number"));
    }

    #[test]
    fn test_struct_variant_bool_type() {
        #[derive(ToPrompt, Serialize, Deserialize)]
        #[serde(tag = "type")]
        enum Config {
            Setting { enabled: bool, name: String },
        }

        let schema = Config::prompt_schema();

        // bool should map to TypeScript 'boolean'
        assert!(schema.contains("enabled: boolean"));
        assert!(schema.contains("name: string"));
    }
}
