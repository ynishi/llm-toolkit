use llm_toolkit::{BlueprintWorkflow, Orchestrator, TypeMarker};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Debug, Clone, TypeMarker, PartialEq)]
struct HighConceptResponse {
    #[serde(default = "default_high_concept_type")]
    __type: String,
    pub reasoning: String,
    pub high_concept: String,
}

fn default_high_concept_type() -> String {
    "HighConceptResponse".to_string()
}

#[derive(Serialize, Deserialize, Debug, Clone, TypeMarker, PartialEq)]
struct ProfileResponse {
    #[serde(default = "default_profile_type")]
    __type: String,
    pub name: String,
    pub age: i32,
}

fn default_profile_type() -> String {
    "ProfileResponse".to_string()
}

#[test]
fn test_get_typed_output_success() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = Orchestrator::new(blueprint);

    // Manually insert a response into the context
    let high_concept = HighConceptResponse {
        __type: "HighConceptResponse".to_string(),
        reasoning: "Deep analysis...".to_string(),
        high_concept: "A brilliant concept".to_string(),
    };

    // Use the context() method to get mutable access
    let context = orchestrator.context_mut();
    context.insert("step_1".to_string(), json!(high_concept));

    // Retrieve using type marker
    let retrieved: HighConceptResponse = orchestrator.get_typed_output().unwrap();

    assert_eq!(retrieved.__type, "HighConceptResponse");
    assert_eq!(retrieved.reasoning, "Deep analysis...");
    assert_eq!(retrieved.high_concept, "A brilliant concept");
}

#[test]
fn test_get_typed_output_multiple_types() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = Orchestrator::new(blueprint);

    // Insert multiple different types
    let high_concept = HighConceptResponse {
        __type: "HighConceptResponse".to_string(),
        reasoning: "Deep analysis...".to_string(),
        high_concept: "A brilliant concept".to_string(),
    };

    let profile = ProfileResponse {
        __type: "ProfileResponse".to_string(),
        name: "Alice".to_string(),
        age: 30,
    };

    let context = orchestrator.context_mut();
    context.insert("step_1".to_string(), json!(high_concept));
    context.insert("step_2".to_string(), json!(profile));

    // Retrieve each type
    let retrieved_concept: HighConceptResponse = orchestrator.get_typed_output().unwrap();
    let retrieved_profile: ProfileResponse = orchestrator.get_typed_output().unwrap();

    assert_eq!(retrieved_concept.__type, "HighConceptResponse");
    assert_eq!(retrieved_profile.__type, "ProfileResponse");
    assert_eq!(retrieved_profile.name, "Alice");
    assert_eq!(retrieved_profile.age, 30);
}

#[test]
fn test_get_typed_output_not_found() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let orchestrator = Orchestrator::new(blueprint);

    // Try to retrieve a type that doesn't exist in context
    let result: Result<HighConceptResponse, _> = orchestrator.get_typed_output();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("No output found with __type")
    );
}

#[test]
fn test_get_typed_output_wrong_format() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = Orchestrator::new(blueprint);

    // Insert data with correct __type but wrong structure
    let wrong_data = json!({
        "__type": "HighConceptResponse",
        "wrong_field": "value"
    });

    let context = orchestrator.context_mut();
    context.insert("step_1".to_string(), wrong_data);

    // Try to retrieve - should fail deserialization
    let result: Result<HighConceptResponse, _> = orchestrator.get_typed_output();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to deserialize")
    );
}

#[test]
fn test_get_typed_output_ignores_step_id() {
    let blueprint = BlueprintWorkflow::new("Test workflow".to_string());
    let mut orchestrator = Orchestrator::new(blueprint);

    // Insert with non-deterministic step ID
    let high_concept = HighConceptResponse {
        __type: "HighConceptResponse".to_string(),
        reasoning: "Deep analysis...".to_string(),
        high_concept: "A brilliant concept".to_string(),
    };

    let context = orchestrator.context_mut();
    context.insert("world_generation_analysis".to_string(), json!(high_concept));

    // Should still retrieve by type, regardless of step ID
    let retrieved: HighConceptResponse = orchestrator.get_typed_output().unwrap();

    assert_eq!(retrieved.__type, "HighConceptResponse");
    assert_eq!(retrieved.reasoning, "Deep analysis...");
}
