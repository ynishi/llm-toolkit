### Agent API: Capability and Intent Separation


The Agent API follows the principle of **capability and intent separation**:
- **Capability**: An agent declares what it can do via two associated types:
  - `type Output`: The structured output type this agent produces
  - `type Expertise`: The expertise definition (typically `&'static str` or `Expertise` from `llm-toolkit-expertise`)
  - `fn expertise(&self) -> &Self::Expertise`: Returns agent's expertise/capabilities
- **Intent**: The orchestrator provides what needs to be done as a `Payload` (multi-modal content)

This separation enables maximum reusability and flexibility.

**Basic Agent Implementation:**

```rust
use llm_toolkit::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

struct MyAgent;

#[async_trait]
impl Agent for MyAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Analyzes data and provides insights";
        &EXPERTISE
    }

    async fn execute(&self, payload: Payload) -> Result<String, AgentError> {
        // Implementation
        Ok("analysis result".to_string())
    }
}
```

