//! Example demonstrating multimodal prompt generation with images

use llm_toolkit::{ImageData, PromptPart, ToPrompt};
use serde::Serialize;

#[derive(ToPrompt, Serialize)]
#[prompt(template = "Analyze this image and answer: {{question}}")]
struct ImageAnalysisPrompt {
    #[prompt(image)]
    image: ImageData,
    question: String,
}

#[derive(ToPrompt, Serialize)]
struct ComparisonPrompt {
    #[prompt(image)]
    before_image: ImageData,
    #[prompt(image)]
    after_image: ImageData,
    /// What should be compared
    comparison_task: String,
    #[prompt(skip)]
    internal_id: u32,
}

fn main() {
    // Example 1: Simple image analysis with template
    println!("=== Example 1: Image Analysis with Template ===\n");

    let analysis_prompt = ImageAnalysisPrompt {
        image: ImageData::new("image/jpeg", vec![0xFF, 0xD8, 0xFF, 0xE0]),
        question: "What objects can you identify in this image?".to_string(),
    };

    let parts = analysis_prompt.to_prompt_parts();
    println!("Generated {} prompt parts:", parts.len());

    for (i, part) in parts.iter().enumerate() {
        match part {
            PromptPart::Text(text) => {
                println!("Part {}: Text - {}", i + 1, text);
            }
            PromptPart::Image { media_type, data } => {
                println!(
                    "Part {}: Image - Type: {}, Size: {} bytes",
                    i + 1,
                    media_type,
                    data.len()
                );
            }
        }
    }

    println!("\nText-only prompt (backward compatible):");
    println!("{}", analysis_prompt.to_prompt());

    // Example 2: Multiple images without template
    println!("\n=== Example 2: Multiple Images Comparison ===\n");

    let comparison_prompt = ComparisonPrompt {
        before_image: ImageData::new("image/png", vec![0x89, 0x50, 0x4E, 0x47]),
        after_image: ImageData::new("image/png", vec![0x89, 0x50, 0x4E, 0x48]),
        comparison_task: "Identify the differences between these two screenshots".to_string(),
        internal_id: 12345, // This will be skipped
    };

    let parts = comparison_prompt.to_prompt_parts();
    println!("Generated {} prompt parts:", parts.len());

    for (i, part) in parts.iter().enumerate() {
        match part {
            PromptPart::Text(text) => {
                println!("Part {}: Text - {}", i + 1, text);
            }
            PromptPart::Image { media_type, data } => {
                println!(
                    "Part {}: Image - Type: {}, Size: {} bytes",
                    i + 1,
                    media_type,
                    data.len()
                );
            }
        }
    }

    // Example 3: Using ImageData helpers
    println!("\n=== Example 3: ImageData Helpers ===\n");

    // From base64
    let base64_image =
        ImageData::from_base64("SGVsbG8gV29ybGQ=", "image/test").expect("Failed to decode base64");
    println!(
        "Created image from base64, size: {} bytes",
        base64_image.data.len()
    );

    // Convert to base64
    let encoded = base64_image.to_base64();
    println!("Encoded back to base64: {}", encoded);

    // From data URL
    let data_url = "data:image/png;base64,iVBORw0KGgo=";
    let url_image = ImageData::try_from(data_url).expect("Failed to parse data URL");
    println!("Parsed image from data URL, type: {}", url_image.media_type);
}
