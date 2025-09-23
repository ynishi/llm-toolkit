use llm_toolkit::{ToPrompt, ToPromptSet, prompt::PromptPart};
use serde::Serialize;

// Example 1: Basic multi-target prompt generation
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Visual",
    template = "# {{title}}\n\n{{description}}\n\n**Priority**: {{priority}}"
)]
#[prompt_for(name = "Markdown", template = "- [ ] {{title}}: {{description}}")]
struct TodoItem {
    title: String,
    description: String,
    priority: String,

    // This field is only included in the API target
    #[prompt_for(name = "API")]
    id: u64,

    // This field is renamed in the API target
    #[prompt_for(name = "API", rename = "created_at")]
    timestamp: u64,

    // This field is skipped in all targets
    #[prompt_for(skip)]
    internal_state: String,
}

// Example 2: Using format_with for custom formatting
fn format_status(status: &Status) -> String {
    match status {
        Status::Draft => "📝 Draft".to_string(),
        Status::InProgress => "🚧 In Progress".to_string(),
        Status::Completed => "✅ Completed".to_string(),
    }
}

#[derive(Debug, Clone, Serialize)]
enum Status {
    Draft,
    InProgress,
    Completed,
}

impl ToPrompt for Status {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(format!("{:?}", self))]
    }

    fn to_prompt(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Human", template = "Document: {{title}}\nStatus: {{status}}")]
struct Document {
    title: String,

    #[prompt_for(name = "Human", format_with = "format_status")]
    status: Status,

    #[prompt_for(name = "Machine")]
    id: String,
}

// Example 3: Multimodal prompts with images
#[derive(Debug, Clone)]
struct ImageAttachment {
    media_type: String,
    data: Vec<u8>,
    caption: String,
}

impl ToPrompt for ImageAttachment {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![
            PromptPart::Image {
                media_type: self.media_type.clone(),
                data: self.data.clone(),
            },
            PromptPart::Text(format!("Caption: {}", self.caption)),
        ]
    }

    fn to_prompt(&self) -> String {
        format!("[Image: {}]", self.caption)
    }
}

impl Serialize for ImageAttachment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.caption)
    }
}

#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Analysis",
    template = "Analyze this content:\nTitle: {{title}}\nDescription: {{description}}"
)]
struct ContentAnalysis {
    title: String,
    description: String,

    #[prompt_for(name = "Analysis", image)]
    screenshot: ImageAttachment,

    #[prompt_for(name = "Summary")]
    key_points: StringList,
}

// Custom wrapper for Vec<String> to implement ToPrompt
#[derive(Debug)]
struct StringList(Vec<String>);

impl ToPrompt for StringList {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Text(self.0.join(", "))]
    }

    fn to_prompt(&self) -> String {
        self.0.join(", ")
    }
}

impl Serialize for StringList {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Todo item with multiple targets
    println!("=== Todo Item Example ===\n");

    let todo = TodoItem {
        title: "Implement ToPromptSet".to_string(),
        description: "Add support for multiple prompt targets".to_string(),
        priority: "High".to_string(),
        id: 12345,
        timestamp: 1700000000,
        internal_state: "active".to_string(),
    };

    println!("Visual Target:");
    println!("{}\n", todo.to_prompt_for("Visual")?);

    println!("Markdown Target:");
    println!("{}\n", todo.to_prompt_for("Markdown")?);

    println!("API Target:");
    println!("{}\n", todo.to_prompt_for("API")?);

    // Example 2: Document with custom formatting
    println!("=== Document Example ===\n");

    let doc = Document {
        title: "Project Proposal".to_string(),
        status: Status::InProgress,
        id: "doc-123".to_string(),
    };

    println!("Human Target:");
    println!("{}\n", doc.to_prompt_for("Human")?);

    println!("Machine Target:");
    println!("{}\n", doc.to_prompt_for("Machine")?);

    // Show all status variants with custom formatting
    println!("All Status Formats:");
    for status in [Status::Draft, Status::InProgress, Status::Completed] {
        println!("  {:?} -> {}", status, format_status(&status));
    }
    println!();

    // Example 3: Multimodal content
    println!("=== Multimodal Content Example ===\n");

    let analysis = ContentAnalysis {
        title: "UI Screenshot Analysis".to_string(),
        description: "Analyzing the new dashboard design".to_string(),
        screenshot: ImageAttachment {
            media_type: "image/png".to_string(),
            data: vec![137, 80, 78, 71], // PNG header bytes
            caption: "Dashboard screenshot".to_string(),
        },
        key_points: StringList(vec![
            "Clean layout".to_string(),
            "Good color scheme".to_string(),
            "Responsive design".to_string(),
        ]),
    };

    println!("Analysis Target (parts):");
    let parts = analysis.to_prompt_parts_for("Analysis")?;
    for (i, part) in parts.iter().enumerate() {
        match part {
            PromptPart::Text(text) => println!("  Part {}: Text - {}", i + 1, text),
            PromptPart::Image { media_type, data } => {
                println!(
                    "  Part {}: Image - {} ({} bytes)",
                    i + 1,
                    media_type,
                    data.len()
                )
            }
        }
    }
    println!();

    println!("Summary Target:");
    println!("{}\n", analysis.to_prompt_for("Summary")?);

    // Demonstrate error handling
    println!("=== Error Handling ===\n");

    match todo.to_prompt_for("NonExistent") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error: {}", e),
    }

    Ok(())
}
