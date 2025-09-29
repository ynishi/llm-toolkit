use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(template_file = "tests/templates/simple_mode.jinja")]
struct SimpleReport {
    name: String,
    age: u32,
}

#[test]
fn test_simple_external_template() {
    let report = SimpleReport {
        name: "Alice".to_string(),
        age: 25,
    };

    let prompt = report.to_prompt();
    println!("Generated prompt:\n{}", prompt);
    assert!(prompt.contains("Report for Alice"));
    assert!(prompt.contains("Age: 25"));
    assert!(prompt.contains("Status: Active"));
}

// Test file not found error
#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct NonExistent {
    value: String,
}

// This would fail at compile time with:
// #[derive(ToPrompt)]
// #[prompt(template_file = "nonexistent.jinja")]
// struct NonExistent { ... }
