use llm_toolkit::ToPrompt;
use serde::Serialize;
use serde_json::Value;

#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "schema_only")]
/// A concept for image generation.
struct Concept {
    /// The main idea for the art to be generated.
    #[prompt(example = "a cinematic, dynamic shot of a futuristic city at night")]
    prompt: String,

    /// Elements to exclude from the generation.
    #[serde(default)]
    negative_prompt: Option<String>,

    /// The style of the generation.
    #[prompt(example = "anime")]
    style: String,
}

#[test]
fn test_struct_schema_only_mode() {
    let concept = Concept {
        prompt: "test prompt".to_string(),
        negative_prompt: None,
        style: "test style".to_string(),
    };

    // Test schema_only mode
    let schema = concept.to_prompt_parts_with_mode("schema_only");
    assert_eq!(schema.len(), 1);

    if let llm_toolkit::prompt::PromptPart::Text(text) = &schema[0] {
        // Check that the schema contains struct name (TypeScript format)
        assert!(text.contains("type Concept = {"));
        // Check that it contains the doc comment (JSDoc format)
        assert!(text.contains("A concept for image generation"));
        // Check field names and types (TypeScript format)
        assert!(text.contains("prompt: string;"));
        assert!(text.contains("negative_prompt: string | null;"));
        assert!(text.contains("style: string;"));
        // Check field doc comments
        assert!(text.contains("The main idea for the art to be generated"));
        assert!(text.contains("Elements to exclude from the generation"));
        assert!(text.contains("The style of the generation"));
    } else {
        panic!("Expected Text variant");
    }
}

#[test]
fn test_struct_to_prompt_with_mode() {
    let concept = Concept {
        prompt: "test prompt".to_string(),
        negative_prompt: Some("test negative".to_string()),
        style: "test style".to_string(),
    };

    // Test to_prompt_with_mode (TypeScript format)
    let schema_str = concept.to_prompt_with_mode("schema_only");
    assert!(schema_str.contains("type Concept = {"));
    assert!(schema_str.contains("prompt: string;"));
}

#[derive(ToPrompt, Serialize)]
#[prompt(mode = "schema_only")]
/// User profile data structure
struct UserProfile {
    /// User's unique identifier
    id: u64,
    /// User's display name
    name: String,
    /// User's email address
    email: Option<String>,
    /// Active status of the user
    is_active: bool,
}

#[test]
fn test_struct_without_example_attribute() {
    let profile = UserProfile {
        id: 123,
        name: "Alice".to_string(),
        email: Some("alice@example.com".to_string()),
        is_active: true,
    };

    // Should generate schema when using schema_only mode (TypeScript format)
    let schema = profile.to_prompt_with_mode("schema_only");
    assert!(schema.contains("type UserProfile = {"));
    assert!(schema.contains("id: number;"));
    assert!(schema.contains("name: string;"));
    assert!(schema.contains("email: string | null;"));
    assert!(schema.contains("is_active: boolean;"));
}

#[test]
fn test_example_only_mode_with_examples() {
    let concept = Concept::default();

    // Test example_only mode - should use example attributes
    let example_str = concept.to_prompt_with_mode("example_only");

    // Parse the JSON to verify structure
    let json: Value = serde_json::from_str(&example_str).expect("Invalid JSON");

    // Should use example values from attributes
    assert_eq!(
        json["prompt"].as_str(),
        Some("a cinematic, dynamic shot of a futuristic city at night")
    );
    assert_eq!(json["style"].as_str(), Some("anime"));

    // Fields without example should use Default values
    assert_eq!(json["negative_prompt"], Value::Null);
}

#[test]
fn test_full_mode_combines_schema_and_example() {
    let concept = Concept::default();

    // Test full mode - should combine schema and example (TypeScript format)
    let full_output = concept.to_prompt_with_mode("full");

    // Should contain both schema and example sections
    assert!(full_output.contains("type Concept = {"));
    assert!(full_output.contains("### Example"));
    assert!(full_output.contains("Here is an example of a valid `Concept` object"));

    // Should contain schema information (TypeScript format)
    assert!(full_output.contains("prompt: string;"));
    assert!(full_output.contains("negative_prompt: string | null;"));

    // Should contain example JSON
    assert!(full_output.contains("a cinematic, dynamic shot of a futuristic city at night"));
    assert!(full_output.contains("anime"));
}

#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "schema_only")]
/// Configuration for a service
struct ServiceConfig {
    /// Service name
    #[prompt(example = "my-service")]
    name: String,

    /// Port number
    port: u16,

    /// Enable debug mode
    debug: bool,
}

#[test]
fn test_default_fallback_for_fields_without_example() {
    let config = ServiceConfig::default();

    // Test example_only mode with Default fallback
    let example_str = config.to_prompt_with_mode("example_only");

    // Parse JSON to verify
    let json: Value = serde_json::from_str(&example_str).expect("Invalid JSON");

    // Field with example attribute
    assert_eq!(json["name"].as_str(), Some("my-service"));

    // Fields without example should use Default values
    assert_eq!(json["port"].as_u64(), Some(0)); // u16 default is 0
    assert_eq!(json["debug"].as_bool(), Some(false)); // bool default is false
}

#[derive(ToPrompt, Serialize)]
#[prompt(mode = "schema_only")]
/// Task without Default implementation
struct TaskWithoutDefault {
    /// Task ID
    #[prompt(example = "task-123")]
    task_id: String,

    /// Task priority
    priority: u32,
}

#[test]
fn test_struct_without_default_uses_actual_values() {
    let task = TaskWithoutDefault {
        task_id: "actual-task-456".to_string(),
        priority: 5,
    };

    // Test example_only mode without Default implementation
    let example_str = task.to_prompt_with_mode("example_only");

    // Parse JSON to verify
    let json: Value = serde_json::from_str(&example_str).expect("Invalid JSON");

    // Field with example attribute should still use the example
    assert_eq!(json["task_id"].as_str(), Some("task-123"));

    // Field without example and no Default should use actual value
    assert_eq!(json["priority"].as_u64(), Some(5));
}
