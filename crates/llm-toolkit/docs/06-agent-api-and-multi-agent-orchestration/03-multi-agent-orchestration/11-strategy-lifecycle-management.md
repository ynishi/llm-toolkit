#### Strategy Lifecycle Management


The `StrategyLifecycle` trait provides a unified interface for managing execution strategies across both `Orchestrator` and `ParallelOrchestrator`, enabling advanced use cases like pre-generated strategies, strategy caching, and testing.

**Status**: ✅ **Complete and tested** (6 dedicated tests passing)

**Features:**
- ✅ Unified trait for strategy management
- ✅ Strategy injection (bypass LLM generation)
- ✅ Strategy retrieval (inspect current strategy)
- ✅ Strategy-only generation (preview without execution)
- ✅ Implemented by both orchestrator types
- ✅ Trait object support for polymorphism

**Basic Usage:**

```rust
use llm_toolkit::orchestrator::{
    Orchestrator,
    StrategyLifecycle,
    StrategyMap,
    StrategyStep
};

let mut orchestrator = Orchestrator::new(blueprint);

// Method 1: Let orchestrator generate strategy automatically
let result = orchestrator.execute("Analyze data").await?;

// Method 2: Inject pre-generated strategy
let mut custom_strategy = StrategyMap::new("Custom Analysis".to_string());
custom_strategy.add_step(StrategyStep::new(
    "step_1".to_string(),
    "Load data".to_string(),
    "DataLoader".to_string(),
    "Load from {{ source }}".to_string(),
    "Data loaded".to_string(),
));
orchestrator.set_strategy_map(custom_strategy);

// Retrieve current strategy
if let Some(strategy) = orchestrator.strategy_map() {
    println!("Current strategy: {}", strategy.goal);
    println!("Steps: {}", strategy.steps.len());
}

// Method 3: Preview strategy without execution
let strategy = orchestrator
    .generate_strategy_only("Analyze data")
    .await?;

println!("Generated strategy has {} steps", strategy.steps.len());
// Decision: execute now or modify first
```

**ParallelOrchestrator Usage:**

```rust
use llm_toolkit::orchestrator::{ParallelOrchestrator, StrategyLifecycle};

let mut orchestrator = ParallelOrchestrator::new(blueprint);

// Generate strategy without execution
let strategy = orchestrator
    .generate_strategy_only("Process files in parallel")
    .await?;

// Inspect and optionally modify strategy
println!("Will execute {} steps in parallel", strategy.steps.len());

// Execute with the generated (or modified) strategy
let result = orchestrator
    .execute("Process files", token, None, None)
    .await?;
```

**Advanced: Strategy Caching**

```rust
use std::collections::HashMap;
use llm_toolkit::orchestrator::{StrategyLifecycle, StrategyMap};

// Strategy cache for common tasks
let mut strategy_cache: HashMap<String, StrategyMap> = HashMap::new();

async fn execute_with_cache(
    orchestrator: &mut impl StrategyLifecycle,
    task: &str,
) -> Result<()> {
    // Check cache
    if let Some(cached_strategy) = strategy_cache.get(task) {
        orchestrator.set_strategy_map(cached_strategy.clone());
    } else {
        // Generate and cache
        let strategy = orchestrator.generate_strategy_only(task).await?;
        strategy_cache.insert(task.to_string(), strategy.clone());
        orchestrator.set_strategy_map(strategy);
    }

    // Execute with cached/generated strategy
    // ... execution logic
    Ok(())
}
```

**Trait Definition:**

```rust
#[async_trait]
pub trait StrategyLifecycle {
    /// Injects a pre-built strategy map, bypassing automatic generation
    fn set_strategy_map(&mut self, strategy: StrategyMap);

    /// Returns the currently active strategy map, if any
    fn strategy_map(&self) -> Option<&StrategyMap>;

    /// Generates a strategy map for the given task without executing it
    async fn generate_strategy_only(
        &mut self,
        task: &str,
    ) -> Result<StrategyMap, OrchestratorError>;
}
```

**Use Cases:**
- **Testing**: Inject controlled strategies for deterministic tests
- **Caching**: Reuse strategies for common tasks to reduce LLM costs
- **Strategy Templates**: Build libraries of proven strategies
- **Preview & Modify**: Generate strategy, review/modify, then execute
- **Multi-Orchestrator Workflows**: Share strategies across orchestrator instances
- **Strategy Versioning**: Store and version successful strategies
- **Debugging**: Inspect strategy before execution

