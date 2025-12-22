#### Using Predefined Strategies


By default, the orchestrator automatically generates execution strategies from your blueprint using an internal LLM. However, you can also provide a predefined strategy to:

- **Reuse known-good strategies** that have been validated
- **Test specific execution paths** with deterministic workflows
- **Implement custom strategy generation logic** outside the orchestrator
- **Skip strategy generation costs** when you already know the optimal plan

**Basic Usage:**

```rust
use llm_toolkit::orchestrator::{Orchestrator, StrategyMap, StrategyStep};

// Create orchestrator
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.add_agent(ClaudeCodeAgent::new());

// Define a custom strategy manually
let mut strategy = StrategyMap::new("Write a technical article".to_string());

// Step 1: Create outline
let mut step1 = StrategyStep::new(
    "step_1".to_string(),
    "Create article outline".to_string(),
    "ClaudeCodeAgent".to_string(),
    "Create an outline for: {{ task }}".to_string(),
    "Article outline".to_string(),
);
step1.output_key = Some("outline".to_string());  // Custom alias
strategy.add_step(step1);

// Step 2: Write introduction - can reference using custom alias
let mut step2 = StrategyStep::new(
    "step_2".to_string(),
    "Write introduction".to_string(),
    "ClaudeCodeAgent".to_string(),
    "Based on {{ outline }}, write an introduction".to_string(),  // Using custom alias
    "Introduction paragraph".to_string(),
);
step2.output_key = Some("introduction".to_string());
strategy.add_step(step2);

// Set the predefined strategy
orchestrator.set_strategy_map(strategy);

// Execute - strategy generation is skipped
let result = orchestrator.execute("Rust ownership system").await;
```

**Retrieving Current Strategy:**

```rust
// Check if strategy is set
if let Some(strategy) = orchestrator.strategy_map() {
    println!("Strategy has {} steps", strategy.steps.len());
    for (i, step) in strategy.steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step.description);
    }
}
```

**Backward Compatibility:**

When no predefined strategy is set, the orchestrator behaves exactly as before - automatically generating strategies from the blueprint:

```rust
// Traditional usage - automatic strategy generation
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.add_agent(ClaudeCodeAgent::new());
let result = orchestrator.execute(task).await; // Auto-generates strategy
```

**When to Use Predefined Strategies:**

| Scenario | Use Auto-Generation | Use Predefined Strategy |
|----------|---------------------|------------------------|
| Exploring new workflows | ✅ Yes | ❌ No |
| Production with validated flows | ❌ No | ✅ Yes |
| Testing specific error scenarios | ❌ No | ✅ Yes |
| Cost optimization (reuse strategies) | ❌ No | ✅ Yes |
| Prototyping and experimentation | ✅ Yes | ❌ No |

**Generating Strategies Without Execution:**

If you want to generate a strategy but not execute it immediately (e.g., to save it as a template), use `generate_strategy_only()`:

```rust
// Generate strategy without executing
let strategy = orchestrator.generate_strategy_only("Process documents").await?;

// Save to file for reuse
let json = serde_json::to_string_pretty(&strategy)?;
std::fs::write("my_workflow.json", json)?;

// Later: Load and execute
let json = std::fs::read_to_string("my_workflow.json")?;
let strategy: StrategyMap = serde_json::from_str(&json)?;
orchestrator.set_strategy_map(strategy);
orchestrator.execute("Process documents").await?;
```

This is useful for creating workflow templates that can be reused across multiple runs.

**Example Code:**

See the complete example at `examples/orchestrator_with_predefined_strategy.rs`:

```bash
cargo run --example orchestrator_with_predefined_strategy --features agent,derive
```

