#### Customizing Internal Agents with `with_internal_agents`


By default, `Orchestrator::new()` uses `ClaudeCodeAgent` and `ClaudeCodeJsonAgent` as internal agents for strategy generation and redesign decisions. You can inject custom internal agents for testing, different LLM backends, or specialized behavior.

**Why customize internal agents?**
- **Testing**: Use mock agents to test orchestration logic without external API calls
- **Different LLM providers**: Use Gemini, Ollama, or custom backends for strategy generation
- **Cost optimization**: Use cheaper models for internal decision-making
- **Offline execution**: Run workflows completely offline with mock agents

**Usage:**

```rust
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator};
use llm_toolkit::agent::{Agent, AgentError, Payload};

// Define custom internal agents (e.g., mock agents for testing)
struct MockStrategyAgent;

#[async_trait::async_trait]
impl Agent for MockStrategyAgent {
    type Output = StrategyMap;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock strategy generator for testing";
        &EXPERTISE
    }

    async fn execute(&self, intent: Payload) -> Result<StrategyMap, AgentError> {
        // Return a predefined strategy for testing
        let mut strategy = StrategyMap::new("Mock workflow".to_string());
        strategy.add_step(/* ... */);
        Ok(strategy)
    }
}

struct MockDecisionAgent;

#[async_trait::async_trait]
impl Agent for MockDecisionAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock decision maker for testing";
        &EXPERTISE
    }

    async fn execute(&self, intent: Payload) -> Result<String, AgentError> {
        Ok("RETRY".to_string())  // Simple retry strategy
    }
}

// Create orchestrator with custom internal agents
let orchestrator = Orchestrator::with_internal_agents(
    blueprint,
    Box::new(MockDecisionAgent),      // For intent generation & redesign decisions
    Box::new(MockStrategyAgent),      // For StrategyMap generation
);

// The orchestrator now uses your custom agents for all internal operations
let result = orchestrator.execute(task).await;
```

**Default Internal Agents:**

When using `Orchestrator::new()`, the following internal agents are used:
- **Strategy Generation**: `ClaudeCodeJsonAgent` wrapped in `RetryAgent` (max 3 retries)
- **Intent & Redesign**: `ClaudeCodeAgent` wrapped in `RetryAgent` (max 3 retries)

Both agents are automatically wrapped with `RetryAgent` to ensure robustness in critical orchestration decisions.

**IMPORTANT for `with_internal_agents()`:**

When providing custom internal agents, **you should wrap them with `RetryAgent`** for production use:

```rust
use llm_toolkit::agent::impls::{RetryAgent, gemini::GeminiAgent};

let orchestrator = Orchestrator::with_internal_agents(
    blueprint,
    Box::new(RetryAgent::new(GeminiAgent::new(), 3)),  // Recommended
    Box::new(RetryAgent::new(GeminiAgent::new(), 3)),  // Recommended
);
```

Without `RetryAgent`, a single transient error (network timeout, rate limiting) could cause strategy generation to fail completely.

**Complete Offline Example:**

See `examples/orchestrator_with_mock.rs` for a complete example that runs entirely offline with mock agents:

```bash
cargo run --example orchestrator_with_mock --features agent,derive
```

