//! Test cases for multimodal prompt functionality

use llm_toolkit::{ImageData, PromptPart, ToPrompt};

#[cfg(feature = "derive")]
use serde::Serialize;

#[test]
fn test_prompt_part_text() {
    let part = PromptPart::Text("Hello, world!".to_string());
    match part {
        PromptPart::Text(text) => assert_eq!(text, "Hello, world!"),
        _ => panic!("Expected Text variant"),
    }
}

#[test]
fn test_prompt_part_image() {
    let data = vec![0xFF, 0xD8, 0xFF, 0xE0];
    let part = PromptPart::Image {
        media_type: "image/jpeg".to_string(),
        data: data.clone(),
    };

    match part {
        PromptPart::Image {
            media_type,
            data: img_data,
        } => {
            assert_eq!(media_type, "image/jpeg");
            assert_eq!(img_data, data);
        }
        _ => panic!("Expected Image variant"),
    }
}

#[test]
fn test_to_prompt_parts_basic_types() {
    let s = "test string";
    let parts = s.to_prompt_parts();
    assert_eq!(parts.len(), 1);
    match &parts[0] {
        PromptPart::Text(text) => assert_eq!(text, "test string"),
        _ => panic!("Expected Text variant"),
    }

    let n = 42;
    let parts = n.to_prompt_parts();
    assert_eq!(parts.len(), 1);
    match &parts[0] {
        PromptPart::Text(text) => assert_eq!(text, "42"),
        _ => panic!("Expected Text variant"),
    }
}

#[test]
fn test_to_prompt_backward_compatibility() {
    // Test that to_prompt() still works with the default implementation
    let s = "test";
    assert_eq!(s.to_prompt(), "test");

    let n = 123;
    assert_eq!(n.to_prompt(), "123");
}

#[test]
fn test_image_data_to_prompt_parts() {
    let img_data = vec![1, 2, 3, 4];
    let img = ImageData::new("image/png", img_data.clone());
    let parts = img.to_prompt_parts();

    assert_eq!(parts.len(), 1);
    match &parts[0] {
        PromptPart::Image { media_type, data } => {
            assert_eq!(media_type, "image/png");
            assert_eq!(data, &img_data);
        }
        _ => panic!("Expected Image variant"),
    }
}

#[cfg(feature = "derive")]
#[test]
fn test_derive_with_image_field() {
    use llm_toolkit::ToPrompt;

    #[derive(ToPrompt, Serialize)]
    #[prompt(template = "この画像について、次の指示に従ってください: {{instruction}}")]
    struct ImageAnalysisPrompt {
        #[prompt(image)]
        image: ImageData,
        instruction: String,
        #[prompt(skip)]
        internal_metadata: String,
    }

    let prompt = ImageAnalysisPrompt {
        image: ImageData::new("image/jpeg", vec![0xFF, 0xD8]),
        instruction: "猫の種類を教えてください".to_string(),
        internal_metadata: "user_id: 42".to_string(),
    };

    let parts = prompt.to_prompt_parts();

    // Should have 2 parts: image and text
    assert_eq!(parts.len(), 2);

    // First part should be the image
    match &parts[0] {
        PromptPart::Image { media_type, data } => {
            assert_eq!(media_type, "image/jpeg");
            assert_eq!(data, &vec![0xFF, 0xD8]);
        }
        _ => panic!("Expected Image variant at index 0"),
    }

    // Second part should be the rendered template text
    match &parts[1] {
        PromptPart::Text(text) => {
            assert_eq!(
                text,
                "この画像について、次の指示に従ってください: 猫の種類を教えてください"
            );
        }
        _ => panic!("Expected Text variant at index 1"),
    }

    // Test backward compatibility with to_prompt()
    let text_only = prompt.to_prompt();
    assert_eq!(
        text_only,
        "この画像について、次の指示に従ってください: 猫の種類を教えてください"
    );
}

#[cfg(feature = "derive")]
#[test]
fn test_derive_without_template() {
    use llm_toolkit::ToPrompt;

    #[derive(ToPrompt, Serialize)]
    struct MixedPrompt {
        #[prompt(image)]
        screenshot: ImageData,
        description: String,
        #[prompt(rename = "user_query")]
        query: String,
        #[prompt(skip)]
        debug_info: String,
    }

    let prompt = MixedPrompt {
        screenshot: ImageData::new("image/png", vec![0x89, 0x50, 0x4E, 0x47]),
        description: "Screenshot of error dialog".to_string(),
        query: "How to fix this error?".to_string(),
        debug_info: "Internal debug data".to_string(),
    };

    let parts = prompt.to_prompt_parts();

    // Should have 2 parts: image and text fields
    assert_eq!(parts.len(), 2);

    // First part should be the image
    match &parts[0] {
        PromptPart::Image { media_type, data } => {
            assert_eq!(media_type, "image/png");
            assert_eq!(data, &vec![0x89, 0x50, 0x4E, 0x47]);
        }
        _ => panic!("Expected Image variant"),
    }

    // Second part should be the text fields
    match &parts[1] {
        PromptPart::Text(text) => {
            assert!(text.contains("description: Screenshot of error dialog"));
            assert!(text.contains("user_query: How to fix this error?"));
            assert!(!text.contains("debug_info")); // Should be skipped
        }
        _ => panic!("Expected Text variant"),
    }
}

#[cfg(feature = "derive")]
#[test]
fn test_multiple_images() {
    use llm_toolkit::ToPrompt;

    #[derive(ToPrompt, Serialize)]
    struct ComparisonPrompt {
        #[prompt(image)]
        before_image: ImageData,
        #[prompt(image)]
        after_image: ImageData,
        task: String,
    }

    let prompt = ComparisonPrompt {
        before_image: ImageData::new("image/png", vec![1, 2]),
        after_image: ImageData::new("image/png", vec![3, 4]),
        task: "Compare these two images".to_string(),
    };

    let parts = prompt.to_prompt_parts();

    // Should have 3 parts: 2 images and 1 text
    assert_eq!(parts.len(), 3);

    // First two parts should be images
    match &parts[0] {
        PromptPart::Image { data, .. } => assert_eq!(data, &vec![1, 2]),
        _ => panic!("Expected first Image variant"),
    }

    match &parts[1] {
        PromptPart::Image { data, .. } => assert_eq!(data, &vec![3, 4]),
        _ => panic!("Expected second Image variant"),
    }

    // Last part should be text
    match &parts[2] {
        PromptPart::Text(text) => {
            assert!(text.contains("task: Compare these two images"));
        }
        _ => panic!("Expected Text variant"),
    }
}
