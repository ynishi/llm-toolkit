#### Configuring Error Recovery Limits


The orchestrator provides configurable limits for error recovery to prevent infinite loops and control API costs:

```rust
use llm_toolkit::orchestrator::{Orchestrator, OrchestratorConfig};

let mut orchestrator = Orchestrator::new(blueprint);

// Method 1: Set entire configuration at once
let config = OrchestratorConfig {
    max_step_remediations: 5,     // Maximum 5 attempts per step (initial + 4 retries)
    max_total_redesigns: 15,       // Maximum 15 redesigns (initial strategy not counted)
};
orchestrator.set_config(config);

// Method 2: Modify individual limits
orchestrator.set_max_step_remediations(5);
orchestrator.set_max_total_redesigns(15);

// Method 3: Use partial configuration with defaults
let config = OrchestratorConfig {
    max_step_remediations: 5,
    ..Default::default()  // Use default for max_total_redesigns (10)
};
orchestrator.set_config(config);
```

**Default Limits:**
- `max_step_remediations`: 3
  - Allows **3 execution attempts** per step (initial attempt + 2 retries)
  - Prevents infinite loops on a single failing step
- `max_total_redesigns`: 10
  - Allows **10 redesign operations** (initial strategy generation not counted)
  - Controls overall workflow redesign attempts across all steps

**How Counting Works:**

*Step-level counting:*
```
Step fails → count incremented → check if count >= max_step_remediations
- Attempt 1 (initial): Fails → count=1 → 1>=3? No → Retry
- Attempt 2: Fails → count=2 → 2>=3? No → Retry
- Attempt 3: Fails → count=3 → 3>=3? Yes → Error: MaxStepRemediationsExceeded
Result: max_step_remediations=3 allows 3 total attempts (2 retries)
```

*Total redesigns counting:*
```
Initial strategy generation → redesigns_triggered=0 (not counted)
Retry/TacticalRedesign/FullRegenerate → redesigns_triggered incremented
- First redesign: redesigns_triggered=1
- ...
- 10th redesign: redesigns_triggered=10 → 10>=10? Yes → Error: MaxTotalRedesignsExceeded
Result: max_total_redesigns=10 allows up to 11 total strategy executions
```

**When Limits Are Exceeded:**
- **Step limit exceeded**: Returns `OrchestratorError::MaxStepRemediationsExceeded { step_index, max_remediations }`
- **Total limit exceeded**: Returns `OrchestratorError::MaxTotalRedesignsExceeded(limit)`

**Choosing Good Values:**
- **Small workflows (2-3 steps)**: Default values work well
- **Large workflows (5+ steps)**: Consider increasing `max_total_redesigns` to 15-20
- **Critical steps**: If certain steps are known to be unstable, increase `max_step_remediations` to 5
- **Cost-sensitive**: Reduce both limits to fail faster (e.g., max_step_remediations=2, max_total_redesigns=5)

