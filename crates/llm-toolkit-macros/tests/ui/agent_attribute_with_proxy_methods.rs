// Test for agent attribute macro with proxy_methods parameter
// Tests that proxy_methods correctly generates builder method proxies
extern crate tracing;

use llm_toolkit::agent::{Agent, AgentError, Payload};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Define a custom agent (mock implementation for testing)
#[derive(Default, Clone)]
struct CustomAgent {
    model: String,
    cwd: Option<PathBuf>,
    env_vars: Vec<(String, String)>,
}

impl CustomAgent {
    fn new() -> Self {
        Self {
            model: "default".to_string(),
            cwd: None,
            env_vars: Vec::new(),
        }
    }

    fn with_model_str(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    fn with_cwd(mut self, path: impl Into<PathBuf>) -> Self {
        self.cwd = Some(path.into());
        self
    }

    fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.push((key.into(), value.into()));
        self
    }
}

#[async_trait::async_trait]
impl Agent for CustomAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Custom agent with builder methods";
        &EXPERTISE
    }

    async fn execute(&self, intent: Payload) -> Result<String, AgentError> {
        let text_intent = intent.to_text();
        let cwd_info = self
            .cwd
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "no cwd".to_string());
        Ok(format!(
            "{{\"result\": \"Processed in {}: {}\"}}",
            cwd_info, text_intent
        ))
    }
}

// Define output type
#[derive(Serialize, Deserialize, Debug, llm_toolkit_macros::ToPrompt)]
struct TestOutput {
    result: String,
}

// Agent with proxy_methods
#[llm_toolkit_macros::agent(
    expertise = "Test agent with proxy methods",
    output = "TestOutput",
    default_inner = "CustomAgent",
    proxy_methods = ["with_cwd", "with_env", "with_model_str"]
)]
struct TestAgentWithProxy;

// Agent without proxy_methods for comparison
#[llm_toolkit_macros::agent(
    expertise = "Test agent without proxy methods",
    output = "TestOutput",
    default_inner = "CustomAgent"
)]
struct TestAgentNoProxy;

fn main() {
    // Test 1: Agent with proxy_methods should have builder methods
    let agent_with_proxy = TestAgentWithProxy::default()
        .with_cwd("/test/path")
        .with_env("KEY", "VALUE")
        .with_model_str("test-model");

    // Test 2: Agent without proxy_methods uses default
    let agent_no_proxy = TestAgentNoProxy::default();

    // Test 3: Custom injection still works
    let custom = CustomAgent::new()
        .with_cwd("/custom/path")
        .with_model_str("custom-model");
    let _agent_custom = TestAgentWithProxy::new(custom);

    // Verify expertise includes JSON schema instructions
    assert!(agent_with_proxy
        .expertise()
        .starts_with("Test agent with proxy methods"));
    assert!(agent_with_proxy.expertise().contains("IMPORTANT"));
    assert!(agent_no_proxy
        .expertise()
        .starts_with("Test agent without proxy methods"));
    assert!(agent_no_proxy.expertise().contains("IMPORTANT"));
}
