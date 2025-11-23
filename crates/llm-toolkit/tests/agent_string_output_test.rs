#![cfg(feature = "agent")]

use llm_toolkit::agent::{Agent, AgentError, Payload};

#[derive(Clone, Default)]
struct MockPlainAgent {
    response: String,
}

impl MockPlainAgent {
    fn with_response(response: &str) -> Self {
        Self {
            response: response.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl Agent for MockPlainAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "mock";
        &EXPERTISE
    }

    async fn execute(&self, _: Payload) -> Result<String, AgentError> {
        Ok(self.response.clone())
    }
}

// Attribute macro agent that uses our mock backend
#[llm_toolkit_macros::agent(
    expertise = "Plain text echo",
    output = "String",
    default_inner = "MockPlainAgent"
)]
struct PlainTextAgent;

#[tokio::test]
async fn returns_plain_text_when_no_json_present() {
    let inner = MockPlainAgent::with_response("hello world");
    let agent = PlainTextAgent::new(inner);

    let result = agent
        .execute("ignored input".to_string().into())
        .await
        .expect("agent should return plain text");

    assert_eq!(result, "hello world");
}

#[tokio::test]
async fn strips_quotes_when_response_is_json_string() {
    let inner = MockPlainAgent::with_response("\"hello json\"");
    let agent = PlainTextAgent::new(inner);

    let result = agent
        .execute("ignored input".to_string().into())
        .await
        .expect("agent should unwrap JSON string");

    assert_eq!(result, "hello json");
}

#[tokio::test]
async fn extracts_from_markdown_code_block() {
    let response = r#"
Here is the answer:
```json
"result inside block"
```
"#;

    let inner = MockPlainAgent::with_response(response);
    let agent = PlainTextAgent::new(inner);

    let result = agent
        .execute("ignored input".to_string().into())
        .await
        .expect("agent should extract string from code block");

    assert_eq!(result, "result inside block");
}

#[tokio::test]
async fn falls_back_to_extracted_json_when_not_a_string() {
    let json_object = r#"{"value":"structured"}"#;
    let inner = MockPlainAgent::with_response(json_object);
    let agent = PlainTextAgent::new(inner);

    let result = agent
        .execute("ignored input".to_string().into())
        .await
        .expect("agent should fallback to extracted JSON when not a string");

    assert_eq!(result, json_object);
}
