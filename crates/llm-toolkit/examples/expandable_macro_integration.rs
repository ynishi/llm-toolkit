//! Example demonstrating define_intent! macro with #[expand] attributes.
//!
//! This shows how to use #[expand] attributes on enum variants to automatically
//! generate Expandable and Selectable trait implementations.
//!
//! Run with: cargo run --example expandable_macro_integration --features agent,derive

use llm_toolkit::define_intent;
use llm_toolkit::intent::expandable::{Expandable, Selectable, SelectionRegistry};

// Define actions with automatic Expandable/Selectable implementations
#[define_intent]
#[intent(
    prompt = r#"
Select an appropriate action based on the user's request.

Available actions:
{{ actions_doc }}

User request: {{ user_request }}

Respond with: <action>ACTION_NAME</action>
"#,
    extractor_tag = "action"
)]
/// Available actions for the assistant
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantAction {
    /// Search for information
    #[expand(template = "Search the web for: {{ query }}\nReturn top 3 results.")]
    WebSearch { query: String },

    /// Read a file
    #[expand(template = "Read the file at path: {{ path }}\nReturn the contents.")]
    FileRead { path: String },

    /// Write to a file
    #[expand(
        template = "Write to file: {{ path }}\n\nContent:\n{{ content }}\n\nConfirm when done."
    )]
    FileWrite { path: String, content: String },

    /// Perform calculation
    #[expand(template = "Calculate: {{ expression }}\nReturn the numeric result.")]
    Calculate { expression: String },

    /// Send a message (no expand - uses default)
    SendMessage {
        _recipient: String,
        _message: String,
    },
}

// Note: FromStr is still required for intent extraction
impl std::str::FromStr for AssistantAction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "WebSearch" => Ok(AssistantAction::WebSearch {
                query: String::new(),
            }),
            "FileRead" => Ok(AssistantAction::FileRead {
                path: String::new(),
            }),
            "FileWrite" => Ok(AssistantAction::FileWrite {
                path: String::new(),
                content: String::new(),
            }),
            "Calculate" => Ok(AssistantAction::Calculate {
                expression: String::new(),
            }),
            "SendMessage" => Ok(AssistantAction::SendMessage {
                _recipient: String::new(),
                _message: String::new(),
            }),
            _ => Err(format!("Unknown action: {}", s)),
        }
    }
}

fn main() {
    println!("=== Expandable Macro Integration Example ===\n");

    // The Expandable and Selectable traits are automatically implemented!

    // Create a registry
    let mut registry = SelectionRegistry::new();

    // Register actions - note that we need to provide actual values
    registry.register(AssistantAction::WebSearch {
        query: "Rust programming".to_string(),
    });

    registry.register(AssistantAction::FileRead {
        path: "/tmp/example.txt".to_string(),
    });

    registry.register(AssistantAction::FileWrite {
        path: "/tmp/output.txt".to_string(),
        content: "Hello, World!".to_string(),
    });

    registry.register(AssistantAction::Calculate {
        expression: "2 + 2".to_string(),
    });

    registry.register(AssistantAction::SendMessage {
        _recipient: "user".to_string(),
        _message: "Task complete".to_string(),
    });

    println!("Registered {} actions\n", registry.len());

    // Test Selectable implementation (auto-generated)
    println!("=== Testing Selectable Trait ===\n");

    let search_action = AssistantAction::WebSearch {
        query: "OpenAI GPT".to_string(),
    };

    println!("Action: {:?}", search_action);
    println!("Selection ID: {}", search_action.selection_id());
    println!("Description: {}\n", search_action.description());

    // Test Expandable implementation (auto-generated from #[expand] templates)
    println!("=== Testing Expandable Trait ===\n");

    let actions = vec![
        AssistantAction::WebSearch {
            query: "Rust async".to_string(),
        },
        AssistantAction::FileRead {
            path: "/etc/hosts".to_string(),
        },
        AssistantAction::Calculate {
            expression: "10 * 5 + 3".to_string(),
        },
        AssistantAction::SendMessage {
            _recipient: "admin".to_string(),
            _message: "Process complete".to_string(),
        },
    ];

    for action in &actions {
        println!("Action: {}", action.selection_id());
        let expanded = action.expand();
        println!("Expanded:\n{}\n", expanded.to_text());
        println!("---\n");
    }

    // Generate prompt section for LLM
    println!("=== Registry Prompt Section ===\n");
    let prompt_section = registry.to_prompt_section();
    println!("{}", prompt_section);

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("1. #[expand(template = \"...\")] automatically implements Expandable");
    println!("2. Selectable is automatically implemented using variant names");
    println!("3. Template can use field names as {{ field_name }}");
    println!("4. Variants without #[expand] use default expansion");
}
