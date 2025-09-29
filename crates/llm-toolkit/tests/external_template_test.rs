use llm_toolkit::{PromptPart, ToPrompt};
use serde::Serialize;

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(template_file = "tests/templates/user_profile.jinja")]
struct UserProfile {
    name: String,
    age: u32,
    role: String,
}

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(template_file = "tests/templates/validated.jinja", validate = true)]
struct ValidatedStruct {
    title: String,
    description: String,
    count: u32,
}

#[test]
fn test_external_template_file() {
    let user = UserProfile {
        name: "Alice".to_string(),
        age: 30,
        role: "Developer".to_string(),
    };

    let prompt = user.to_prompt();
    assert!(prompt.contains("Alice"));
    assert!(prompt.contains("30"));
    assert!(prompt.contains("Developer"));
}

#[test]
fn test_validated_external_template() {
    let item = ValidatedStruct {
        title: "Test Item".to_string(),
        description: "This is a test item".to_string(),
        count: 42,
    };

    let prompt = item.to_prompt();
    assert!(prompt.contains("Test Item"));
    assert!(prompt.contains("This is a test item"));
    assert!(prompt.contains("42"));
}

#[test]
fn test_to_prompt_parts_with_external_template() {
    let user = UserProfile {
        name: "Bob".to_string(),
        age: 25,
        role: "Designer".to_string(),
    };

    let parts = user.to_prompt_parts();
    assert!(!parts.is_empty());

    // Check that the first part is text
    if let PromptPart::Text(text) = &parts[0] {
        assert!(text.contains("Bob"));
        assert!(text.contains("25"));
        assert!(text.contains("Designer"));
    } else {
        panic!("Expected text part");
    }
}
