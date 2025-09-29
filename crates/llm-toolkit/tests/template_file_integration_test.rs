use llm_toolkit::ToPrompt;
use serde::Serialize;

#[test]
fn test_template_file_loading() {
    #[derive(ToPrompt, Serialize)]
    #[prompt(template_file = "tests/templates/integration_test.jinja")]
    struct User {
        name: String,
        age: u32,
    }

    let user = User {
        name: "Alice".to_string(),
        age: 30,
    };

    let prompt = user.to_prompt();
    assert!(prompt.contains("Alice"));
    assert!(prompt.contains("30"));
}

#[test]
fn test_template_file_with_validation() {
    #[derive(ToPrompt, Serialize)]
    #[prompt(template_file = "tests/templates/simple_valid.jinja", validate = true)]
    struct ValidatedUser {
        name: String,
    }

    let user = ValidatedUser {
        name: "Bob".to_string(),
    };

    let prompt = user.to_prompt();
    assert!(prompt.contains("Bob"));
}
