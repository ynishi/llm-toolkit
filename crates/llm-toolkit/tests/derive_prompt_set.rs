use llm_toolkit::{
    ToPromptSet,
    prompt::{PromptPart, ToPrompt},
};
use serde::Serialize;

#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Visual", template = "## {{title}}\n\n> {{description}}")]
struct Task {
    title: String,
    description: String,

    #[prompt_for(name = "Agent")]
    priority: u8,

    #[prompt_for(name = "Agent", rename = "internal_id")]
    id: u64,

    #[prompt_for(skip)]
    is_dirty: bool,
}

#[test]
fn test_visual_target_with_template() {
    let task = Task {
        title: "Implement feature".to_string(),
        description: "Add new functionality".to_string(),
        priority: 1,
        id: 42,
        is_dirty: false,
    };

    let result = task.to_prompt_for("Visual").unwrap();
    assert_eq!(result, "## Implement feature\n\n> Add new functionality");
}

#[test]
fn test_agent_target_with_key_value() {
    let task = Task {
        title: "Implement feature".to_string(),
        description: "Add new functionality".to_string(),
        priority: 1,
        id: 42,
        is_dirty: false,
    };

    let result = task.to_prompt_for("Agent").unwrap();
    assert!(result.contains("title: Implement feature"));
    assert!(result.contains("description: Add new functionality"));
    assert!(result.contains("priority: 1"));
    assert!(result.contains("internal_id: 42"));
    assert!(!result.contains("is_dirty"));
}

#[test]
fn test_unknown_target() {
    let task = Task {
        title: "Test".to_string(),
        description: "Test".to_string(),
        priority: 1,
        id: 1,
        is_dirty: false,
    };

    let result = task.to_prompt_for("Unknown");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Target 'Unknown' not found"));
    assert!(err_msg.contains("Available targets"));
    assert!(err_msg.contains("Visual"));
    assert!(err_msg.contains("Agent"));
}

// Test with format_with attribute
fn format_priority(priority: &u8) -> String {
    match priority {
        1 => "Low".to_string(),
        2 => "Medium".to_string(),
        3 => "High".to_string(),
        _ => "Unknown".to_string(),
    }
}

#[derive(ToPromptSet, Serialize, Debug)]
struct FormattedTask {
    title: String,

    #[prompt_for(name = "Human", format_with = "format_priority")]
    priority: u8,
}

#[test]
fn test_format_with() {
    let task = FormattedTask {
        title: "Important task".to_string(),
        priority: 3,
    };

    let result = task.to_prompt_for("Human").unwrap();
    assert!(result.contains("title: Important task"));
    assert!(result.contains("priority: High"));
}

// Test with multimodal content
#[derive(Debug, Clone)]
struct ImageData {
    media_type: String,
    data: Vec<u8>,
}

impl ToPrompt for ImageData {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Image {
            media_type: self.media_type.clone(),
            data: self.data.clone(),
        }]
    }

    fn to_prompt(&self) -> String {
        "[Image]".to_string()
    }
}

impl Serialize for ImageData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("[Image]")
    }
}

#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Multimodal",
    template = "Analyzing image with caption: {{caption}}"
)]
struct MultimodalTask {
    caption: String,

    #[prompt_for(name = "Multimodal", image)]
    image: ImageData,
}

#[test]
fn test_multimodal_prompt() {
    let task = MultimodalTask {
        caption: "Test image".to_string(),
        image: ImageData {
            media_type: "image/png".to_string(),
            data: vec![0, 1, 2, 3],
        },
    };

    let parts = task.to_prompt_parts_for("Multimodal").unwrap();

    // Should have both image and text parts
    assert_eq!(parts.len(), 2);

    // Check image part
    match &parts[0] {
        PromptPart::Image { media_type, data } => {
            assert_eq!(media_type, "image/png");
            assert_eq!(data, &[0, 1, 2, 3]);
        }
        _ => panic!("Expected image part first"),
    }

    // Check text part
    match &parts[1] {
        PromptPart::Text(text) => {
            assert_eq!(text, "Analyzing image with caption: Test image");
        }
        _ => panic!("Expected text part second"),
    }
}

// Test multiple targets with different configurations
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Brief", template = "{{title}}")]
#[prompt_for(
    name = "Detailed",
    template = "Title: {{title}}\nDescription: {{description}}\nStatus: {{status}}"
)]
struct MultiTargetTask {
    title: String,
    description: String,

    #[prompt_for(name = "Detailed")]
    status: String,

    #[prompt_for(name = "Debug")]
    debug_info: String,
}

#[test]
fn test_multiple_targets() {
    let task = MultiTargetTask {
        title: "Task Title".to_string(),
        description: "Task Description".to_string(),
        status: "Active".to_string(),
        debug_info: "Debug data".to_string(),
    };

    // Test Brief target
    let brief = task.to_prompt_for("Brief").unwrap();
    assert_eq!(brief, "Task Title");

    // Test Detailed target
    let detailed = task.to_prompt_for("Detailed").unwrap();
    assert_eq!(
        detailed,
        "Title: Task Title\nDescription: Task Description\nStatus: Active"
    );

    // Test Debug target (key-value format)
    let debug = task.to_prompt_for("Debug").unwrap();
    assert!(debug.contains("title: Task Title"));
    assert!(debug.contains("description: Task Description"));
    assert!(debug.contains("debug_info: Debug data"));
    assert!(!debug.contains("status")); // Status is only for Detailed target
}
