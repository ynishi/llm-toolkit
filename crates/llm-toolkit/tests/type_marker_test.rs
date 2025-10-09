use llm_toolkit::TypeMarker;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, TypeMarker)]
struct TestResponse {
    #[serde(default = "default_test_type")]
    __type: String,
    pub value: String,
}

fn default_test_type() -> String {
    "TestResponse".to_string()
}

#[derive(Serialize, Deserialize, Debug, Clone, TypeMarker)]
struct AnotherResponse {
    #[serde(default = "default_another_type")]
    __type: String,
    pub count: i32,
}

fn default_another_type() -> String {
    "AnotherResponse".to_string()
}

#[test]
fn test_type_marker_constant() {
    assert_eq!(TestResponse::TYPE_NAME, "TestResponse");
    assert_eq!(AnotherResponse::TYPE_NAME, "AnotherResponse");
}

#[test]
fn test_type_marker_method() {
    assert_eq!(TestResponse::type_marker(), "TestResponse");
    assert_eq!(AnotherResponse::type_marker(), "AnotherResponse");
}

#[test]
fn test_different_types_have_different_markers() {
    assert_ne!(TestResponse::TYPE_NAME, AnotherResponse::TYPE_NAME);
}

#[test]
fn test_serialization_with_type_marker() {
    let response = TestResponse {
        __type: default_test_type(),
        value: "test value".to_string(),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains(r#""__type":"TestResponse""#));
    assert!(json.contains(r#""value":"test value""#));
}

#[test]
fn test_deserialization_with_type_marker() {
    let json = r#"{"__type":"TestResponse","value":"test value"}"#;
    let response: TestResponse = serde_json::from_str(json).unwrap();

    assert_eq!(response.__type, "TestResponse");
    assert_eq!(response.value, "test value");
}

#[test]
fn test_default_type_field_on_missing() {
    // When __type is missing in JSON, serde(default) should fill it
    let json = r#"{"value":"test value"}"#;
    let response: TestResponse = serde_json::from_str(json).unwrap();

    assert_eq!(response.__type, "TestResponse");
    assert_eq!(response.value, "test value");
}
