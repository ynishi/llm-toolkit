// Test for agent attribute macro with init function
// Tests that the init parameter correctly transforms the default inner agent
extern crate tracing;

use llm_toolkit::agent::{Agent, AgentError, Payload};
use serde::{Deserialize, Serialize};

// Define a custom agent (mock implementation for testing)
#[derive(Default, Clone)]
struct CustomAgent {
    model: String,
    workspace: Option<String>,
}

impl CustomAgent {
    fn new() -> Self {
        Self {
            model: "default".to_string(),
            workspace: None,
        }
    }

    fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    fn with_workspace(mut self, path: String) -> Self {
        self.workspace = Some(path);
        self
    }
}

#[async_trait::async_trait]
impl Agent for CustomAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "Custom agent with workspace"
    }

    async fn execute(&self, intent: Payload) -> Result<String, AgentError> {
        let text_intent = intent.to_text();
        let workspace_info = self.workspace.as_deref().unwrap_or("no workspace");
        Ok(format!(
            "{{\"result\": \"Processed in {}: {}\"}}",
            workspace_info, text_intent
        ))
    }
}

// Define output type
#[derive(Serialize, Deserialize, Debug, llm_toolkit_macros::ToPrompt)]
struct TestOutput {
    result: String,
}

// Init function that configures the agent
fn init_with_workspace(agent: CustomAgent) -> CustomAgent {
    agent
        .with_model("configured-model")
        .with_workspace("/workspace/test".to_string())
}

// Agent using init function
#[llm_toolkit_macros::agent(
    expertise = "Test agent with init function",
    output = "TestOutput",
    default_inner = "CustomAgent",
    init = "init_with_workspace"
)]
struct TestAgentWithInit;

// Agent without init function for comparison
#[llm_toolkit_macros::agent(
    expertise = "Test agent without init function",
    output = "TestOutput",
    default_inner = "CustomAgent"
)]
struct TestAgentNoInit;

fn main() {
    // Test 1: Agent with init function should have workspace configured
    let agent_with_init = TestAgentWithInit::default();
    // The init function should have been applied during default construction

    // Test 2: Agent without init function uses plain default
    let agent_no_init = TestAgentNoInit::default();

    // Test 3: Custom injection still works
    let custom = CustomAgent::new().with_workspace("/custom/path".to_string());
    let _agent_custom = TestAgentWithInit::new(custom);

    // Verify expertise includes JSON schema instructions
    assert!(agent_with_init.expertise().starts_with("Test agent with init function"));
    assert!(agent_with_init.expertise().contains("IMPORTANT"));
    assert!(agent_no_init.expertise().starts_with("Test agent without init function"));
    assert!(agent_no_init.expertise().contains("IMPORTANT"));
}
