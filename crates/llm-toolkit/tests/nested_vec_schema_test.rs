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

    // Check TypeScript-style type definition
    assert!(schema.contains("type ProducerOutput = {"));

    // Check that evaluation_passed field is present (TypeScript format)
    assert!(schema.contains("evaluation_passed: boolean;"));

    // Check that results field uses TypeScript array syntax
    assert!(schema.contains("results: EvaluationResult[];"));

    // Note: Nested types are referenced by name only, not expanded inline
    assert!(!schema.contains("rule: string"));
}

#[test]
fn test_evaluation_result_schema() {
    let schema = EvaluationResult::prompt_schema();

    println!("EvaluationResult schema:\n{}", schema);

    assert!(schema.contains("type EvaluationResult = {"));
    assert!(schema.contains("rule: string;"));
    assert!(schema.contains("passed: boolean;"));
}

#[test]
fn test_nested_schema_with_comments() {
    let schema = ProducerOutput::prompt_schema();

    // Check that field comments are preserved (TypeScript format)
    assert!(schema.contains("// Whether the evaluation passed all checks"));
    assert!(schema.contains("// List of evaluation results for each rule"));

    // Note: Nested type comments are NOT included in parent's schema
    // (they're in EvaluationResult's own schema)
    assert!(!schema.contains("// The rule being checked"));
    assert!(!schema.contains("// Whether this specific rule passed"));
}
