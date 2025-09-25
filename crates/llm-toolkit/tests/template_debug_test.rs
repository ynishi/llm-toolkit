use llm_toolkit::ToPrompt;
use serde::Serialize;

// Simple struct with template but no mode syntax (backward compatibility)
#[derive(ToPrompt, Serialize, Debug)]
#[prompt(template = "Test")]
struct Minimal {
    value: String,
}

#[test]
fn test_minimal_template() {
    let data = Minimal {
        value: "test".to_string(),
    };

    let output = data.to_prompt();
    println!("Output: {:?}", output);

    // Should output "Test"
    assert_eq!(output, "Test");
}

#[derive(ToPrompt, Serialize, Debug)]
#[prompt(template = "Value: {value}")]
struct WithPlaceholder {
    value: String,
}

#[test]
fn test_with_placeholder() {
    let data = WithPlaceholder {
        value: "test".to_string(),
    };

    let output = data.to_prompt();
    println!("Output with placeholder: {:?}", output);

    // Should output "Value: test"
    assert_eq!(output, "Value: test");
}
