use llm_toolkit::ToPrompt;
use serde::Serialize;

// Test struct that implements ToPrompt with different modes
#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "schema_only")]
/// A style guide for generated content
struct StyleGuide {
    /// Color palette to use
    color_palette: String,
    /// Camera angle preference
    camera_angle: String,
}

// Test struct that implements ToPrompt with different modes
#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "schema_only")]
/// Main concept for image generation
struct Concept {
    /// Main prompt text
    #[prompt(example = "a futuristic city")]
    prompt: String,
    /// Visual style
    style: String,
}

// Struct with template using mode syntax
#[derive(ToPrompt, Serialize)]
#[prompt(template = r#"
SYSTEM PROMPT:
{system_prompt}

---
USER CONCEPT:
{user_concept}

---
STYLE GUIDE (Schema Only):
{style_guide:schema_only}

---
ALTERNATIVE CONCEPT:
{alternative:example_only}
"#)]
struct ImageGenerationPrompt {
    system_prompt: String,
    user_concept: Concept,
    style_guide: StyleGuide,
    alternative: Concept,
}

#[test]
fn test_template_with_mode_syntax() {
    let prompt = ImageGenerationPrompt {
        system_prompt: "Generate a high-quality image".to_string(),
        user_concept: Concept {
            prompt: "a bustling marketplace".to_string(),
            style: "photorealistic".to_string(),
        },
        style_guide: StyleGuide {
            color_palette: "warm tones".to_string(),
            camera_angle: "birds eye view".to_string(),
        },
        alternative: Concept {
            prompt: "a serene garden".to_string(),
            style: "watercolor".to_string(),
        },
    };

    let output = prompt.to_prompt();

    // Check that the template was rendered correctly
    assert!(output.contains("SYSTEM PROMPT:"));
    assert!(output.contains("Generate a high-quality image"));

    // Check that user_concept uses default (full) mode
    assert!(output.contains("USER CONCEPT:"));
    // Full mode should contain schema information
    assert!(output.contains("Schema for `Concept`"));

    // Check that style_guide uses schema_only mode
    assert!(output.contains("STYLE GUIDE (Schema Only):"));
    assert!(output.contains("Schema for `StyleGuide`"));
    // Schema only mode should NOT contain actual values
    assert!(!output.contains("warm tones"));
    assert!(!output.contains("birds eye view"));

    // Check that alternative uses example_only mode
    assert!(output.contains("ALTERNATIVE CONCEPT:"));
    // Example only mode should contain actual values as JSON
    assert!(output.contains("\"prompt\""));
    assert!(output.contains("\"serene garden\"") || output.contains("\"a futuristic city\"")); // May use example attribute
    assert!(output.contains("\"style\""));
}

// Simple struct with template but no mode syntax (backward compatibility)
#[derive(ToPrompt, Serialize)]
#[prompt(template = "User {name} has role {role}.")]
struct SimpleUser {
    name: String,
    role: String,
}

#[test]
fn test_template_without_mode_syntax_backward_compat() {
    let user = SimpleUser {
        name: "Alice".to_string(),
        role: "Admin".to_string(),
    };

    let output = user.to_prompt();

    // Should use simple template rendering without mode processing
    assert_eq!(output, "User Alice has role Admin.");
}

// Complex nested example
#[derive(ToPrompt, Serialize, Default)]
#[prompt(mode = "schema_only")]
struct TaskConfig {
    /// Task priority
    priority: u32,
    /// Task deadline
    deadline: String,
}

#[derive(ToPrompt, Serialize)]
#[prompt(template = r#"Task Management System

Configuration:
{config:full}

Quick Reference (Schema):
{config:schema_only}

Example Configuration:
{config:example_only}
"#)]
struct TaskManagement {
    config: TaskConfig,
}

#[test]
fn test_same_field_with_different_modes() {
    let task_mgmt = TaskManagement {
        config: TaskConfig {
            priority: 1,
            deadline: "2024-12-31".to_string(),
        },
    };

    let output = task_mgmt.to_prompt();

    // Check all three modes are rendered differently
    assert!(output.contains("Configuration:"));
    assert!(output.contains("Quick Reference (Schema):"));
    assert!(output.contains("Example Configuration:"));

    // All sections should reference TaskConfig in their own way
    let config_mentions = output.matches("TaskConfig").count();
    assert!(config_mentions >= 2); // At least in full and schema_only modes
}

// Test with primitive field types
#[derive(ToPrompt, Serialize)]
#[prompt(template = "Name: {name}, ID: {id}")]
struct PrimitiveFields {
    name: String,
    id: u32,
}

#[test]
fn test_template_with_primitive_fields() {
    let data = PrimitiveFields {
        name: "test value".to_string(),
        id: 42,
    };

    let output = data.to_prompt();

    // Should render primitive values directly
    assert!(output.contains("Name: test value"));
    assert!(output.contains("ID: 42"));
}
