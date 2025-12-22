#### Two-Layer Error Recovery: RetryAgent + Orchestrator


The recommended pattern is to combine `RetryAgent` (agent-level retry) with Orchestrator (workflow-level recovery) for robust error handling:

```rust
use llm_toolkit::agent::impls::{ClaudeCodeAgent, RetryAgent};
use llm_toolkit::orchestrator::{Orchestrator, BlueprintWorkflow};

// Layer 1: Agent-level retry (transient errors)
let claude = ClaudeCodeAgent::new();
let retry_agent = RetryAgent::new(claude, 3);  // Up to 3 retries

// Layer 2: Orchestrator-level recovery (structural errors)
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.add_agent(Box::new(retry_agent));

// Now you have two layers of error recovery:
// - Agent layer: Network errors, 429 rate limits, parse errors
// - Orchestrator layer: Wrong agent selection, strategy issues
```

**Responsibility Separation:**

| Error Type | Layer | Recovery Strategy |
|------------|-------|-------------------|
| Network timeout | Agent (RetryAgent) | Wait + retry (linear backoff) |
| 429 rate limit | Agent (RetryAgent) | Wait retry_after (exponential, max 60s) |
| Parse error | Agent (RetryAgent) | Immediate retry (linear backoff) |
| Agent capability mismatch | Orchestrator | Try different agent (step remediation) |
| Strategy design flaw | Orchestrator | Redesign workflow (tactical/full) |

**Per-Agent Customization:**

You can customize retry behavior for each agent based on importance:

```rust
// Critical agent: More retries
let writer = WriterAgent::default();
let retry_writer = RetryAgent::new(writer, 5);  // 5 retries

// Lightweight agent: Fewer retries
let validator = ValidatorAgent::default();
let retry_validator = RetryAgent::new(validator, 2);  // 2 retries

orchestrator.add_agent(Box::new(retry_writer));
orchestrator.add_agent(Box::new(retry_validator));
```

**Cost Control:**

Worst case: Agent retries × Orchestrator remediations
- Agent: 3 attempts (1 initial + 2 retries)
- Orchestrator: 3 remediations
- Maximum: 3 × 3 = 9 agent calls per step

This is **intentional design**:
- Agent retries handle transient errors (network, API)
- Orchestrator remediations handle structural errors (strategy, capability)
- Both limits are independently configurable for cost control

**Why This Pattern Works:**

- ✅ **Clear Separation**: Transient vs structural errors handled at appropriate levels
- ✅ **DRY Principle**: Same retry logic (RetryAgent) used everywhere
- ✅ **Flexible Control**: Independent configuration of agent and orchestrator retries
- ✅ **No Additional Code**: Uses existing RetryAgent decorator
- ✅ **Production-Ready**: 429 rate limiting, Full Jitter, retry_after support

**When NOT to use RetryAgent:**

If you want the Orchestrator to immediately try a different agent on first failure (no agent-level retry), add agents directly without wrapping:

```rust
// Direct agent addition - no agent-level retry
orchestrator.add_agent(Box::new(ClaudeCodeAgent::new()));

// First error → Orchestrator immediately tries different agent or redesigns
```

