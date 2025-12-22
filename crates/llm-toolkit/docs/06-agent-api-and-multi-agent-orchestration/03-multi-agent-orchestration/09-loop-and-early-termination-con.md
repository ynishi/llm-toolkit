#### Loop and Early Termination Control Flow


The orchestrator supports advanced control flow with loops and early termination, enabling iterative refinement and conditional workflow exit.

**Status**: ✅ **Complete and tested** (160 tests passing)

**Features:**
- ✅ Loop blocks with configurable iteration limits
- ✅ Early termination instructions with conditional evaluation
- ✅ Single-level loops only (nested loops rejected via validation)
- ✅ Optional fields for simplified LLM generation
- ✅ Execution engine with recursive instruction processing
- ✅ Condition template evaluation with MiniJinja
- ✅ Loop aggregation modes (LastSuccess, FirstSuccess, CollectAll)
- ✅ Global loop iteration limits (prevents runaway costs)
- ✅ Integrated with execute_strategy() (automatic legacy migration)

**Data Model:**

```rust
use llm_toolkit::orchestrator::{StrategyInstruction, LoopBlock, TerminateInstruction};

// Example 1: Minimal loop (optimal for LLM generation)
let loop_instruction = StrategyInstruction::Loop(LoopBlock {
    loop_id: "refine".to_string(),
    description: None,  // Optional
    loop_type: None,    // Optional (defaults to While)
    max_iterations: 3,
    condition_template: Some("{{ needs_improvement }}".to_string()),
    body: vec![/* nested instructions */],
    aggregation: None,  // Optional
});

// Example 2: Early termination
let terminate = StrategyInstruction::Terminate(TerminateInstruction {
    terminate_id: "early_exit".to_string(),
    description: None,  // Optional
    condition_template: Some("{{ success }}".to_string()),
    final_output_template: None,  // Optional
});
```

**Minimal JSON Example** (hand-written or LLM-generated):

```json
{
  "goal": "Iteratively refine design",
  "elements": [
    {
      "type": "step",
      "step_id": "initial_design",
      "description": "Create initial design",
      "assigned_agent": "DesignAgent",
      "intent_template": "Create design for {{ task }}",
      "expected_output": "Design document"
    },
    {
      "type": "loop",
      "loop_id": "refine_loop",
      "max_iterations": 5,
      "condition_template": "{{ feedback.needs_improvement }}",
      "body": [
        {
          "type": "step",
          "step_id": "get_feedback",
          "description": "Get design feedback",
          "assigned_agent": "ReviewAgent",
          "intent_template": "Review design",
          "expected_output": "Feedback"
        },
        {
          "type": "terminate",
          "terminate_id": "approved",
          "condition_template": "{{ feedback.approved }}"
        },
        {
          "type": "step",
          "step_id": "improve",
          "description": "Apply improvements",
          "assigned_agent": "DesignAgent",
          "intent_template": "Improve design based on {{ feedback }}",
          "expected_output": "Improved design"
        }
      ]
    }
  ]
}
```

**Configuration:**

```rust
use llm_toolkit::orchestrator::OrchestratorConfig;

let config = OrchestratorConfig {
    max_total_loop_iterations: 50,  // Global limit across all loops (default: 50)
    ..Default::default()
};
orchestrator.set_config(config);
```

**Safety Constraints:**
- Single-level loops only (nested loops are rejected with validation error)
- Global `max_total_loop_iterations` limit prevents runaway costs
- Each loop requires `max_iterations` (per-loop limit)
- Validation via `StrategyMap::validate()` before execution

**Design Decisions:**
- `description` and `loop_type` are **optional** to reduce LLM generation failures
- No `controller_agent` field (reuses existing `internal_agent` for LLM-driven control)
- `condition_template` uses MiniJinja for deterministic evaluation
- Backward compatible: legacy `steps` format still supported via `migrate_legacy_steps()`

**Performance Impact:**
- **6-step workflow with 1s delay**: Adds ~5 seconds total (6 steps - 1 last step)
- **Trade-off**: Slightly slower execution vs. no rate limit errors
- **Best practice**: Use only when targeting rate-limited APIs

