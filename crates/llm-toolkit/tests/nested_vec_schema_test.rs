use llm_toolkit::ToPrompt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct EvaluationResult {
    /// The rule being checked
    pub rule: String,
    /// Whether this specific rule passed
    pub passed: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct ProducerOutput {
    /// Whether the evaluation passed all checks
    pub evaluation_passed: bool,
    /// List of evaluation results for each rule
    pub results: Vec<EvaluationResult>,
}

#[test]
fn test_nested_vec_schema_expansion() {
    let schema = ProducerOutput::prompt_schema();

    println!("Generated schema:\n{}", schema);

    // Check that schema contains the header
    assert!(schema.contains("### Schema for `ProducerOutput`"));

    // Check that evaluation_passed field is present
    assert!(schema.contains("\"evaluation_passed\": \"boolean\""));

    // Check that results field is present as an array
    assert!(schema.contains("\"results\": ["));

    // Check that nested EvaluationResult schema is expanded inline
    assert!(schema.contains("\"rule\": \"string\""));
    assert!(schema.contains("\"passed\": \"boolean\""));

    // The nested schema should be indented
    assert!(schema.contains("    \"rule\": \"string\""));
}

#[test]
fn test_evaluation_result_schema() {
    let schema = EvaluationResult::prompt_schema();

    println!("EvaluationResult schema:\n{}", schema);

    assert!(schema.contains("### Schema for `EvaluationResult`"));
    assert!(schema.contains("\"rule\": \"string\""));
    assert!(schema.contains("\"passed\": \"boolean\""));
}

#[test]
fn test_nested_schema_with_comments() {
    let schema = ProducerOutput::prompt_schema();

    // Check that field comments are preserved
    assert!(schema.contains("// Whether the evaluation passed all checks"));
    assert!(schema.contains("// List of evaluation results for each rule"));

    // Check that nested field comments are included
    assert!(schema.contains("// The rule being checked"));
    assert!(schema.contains("// Whether this specific rule passed"));
}
