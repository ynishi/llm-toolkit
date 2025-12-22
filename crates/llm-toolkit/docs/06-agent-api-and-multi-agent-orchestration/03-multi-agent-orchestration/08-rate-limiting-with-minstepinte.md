#### Rate Limiting with `min_step_interval`


The orchestrator provides proactive rate limiting to prevent API rate limit errors (429 Too Many Requests).

**Problem**: Each orchestrator step typically makes 2+ API calls (intent generation + execution). Without delays, a 6-step workflow can make 12+ calls in 30 seconds, exceeding many LLM API rate limits (e.g., 10 requests/minute for Gemini).

**Solution**: Set `min_step_interval` to introduce a delay after each step completes:

```rust
use std::time::Duration;
use llm_toolkit::orchestrator::{Orchestrator, OrchestratorConfig};

let mut orchestrator = Orchestrator::new(blueprint);

// Method 1: Set entire configuration at once
let config = OrchestratorConfig {
    min_step_interval: Duration::from_millis(500),  // 500ms delay between steps
    ..Default::default()
};
orchestrator.set_config(config);

// Method 2: Use convenience method
orchestrator.set_min_step_interval(Duration::from_secs(1));  // 1 second delay
```

**How It Works:**
- Applied **after** each step completes (before starting next step)
- **Not applied** after the last step (no unnecessary delay)
- `Duration::ZERO` means no delay (default, backward compatible)

**Choosing Good Values:**
- **10 req/min limit** (e.g., Gemini): Use `Duration::from_secs(6)` or higher
- **60 req/min limit** (e.g., Claude): Use `Duration::from_millis(500)` to `Duration::from_secs(1)`
- **Conservative approach**: Start with `Duration::from_secs(1)`, reduce if no errors occur

**Combining with RetryAgent:**

For maximum resilience, combine proactive rate limiting (min_step_interval) with reactive retry (RetryAgent):

```rust
use llm_toolkit::agent::impls::{GeminiAgent, RetryAgent};

// Layer 1: Proactive rate limiting (prevents errors)
orchestrator.set_min_step_interval(Duration::from_secs(1));

// Layer 2: Reactive retry with retry_after support (handles errors)
let gemini = GeminiAgent::new();
let retry_gemini = RetryAgent::new(gemini, 5);  // Respects server retry_after
orchestrator.add_agent(retry_gemini);

// Result: Minimal API errors and automatic recovery if they occur
```

