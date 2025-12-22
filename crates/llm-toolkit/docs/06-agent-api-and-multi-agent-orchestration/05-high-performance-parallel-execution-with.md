### High-Performance Parallel Execution with ParallelOrchestrator


For workflows with independent tasks (e.g., multiple research steps) that can be run concurrently, `llm-toolkit` offers a high-performance `ParallelOrchestrator`. It analyzes the dependencies between steps in a `StrategyMap` and executes independent steps in parallel "waves," significantly reducing total execution time.

**Key Benefits:**
- **Performance**: Drastically speeds up workflows with high degrees of parallelism.
- **Robustness**: Supports per-step timeouts and concurrency limits to prevent stalls and manage resources.
- **Observability**: Integrates with the `tracing` crate to provide clear, correlated logs for concurrent operations.

**Example Usage:**

The API is nearly identical to the sequential `Orchestrator`, but requires a pre-defined `StrategyMap` as it does not generate strategies on its own.

```rust
use llm_toolkit::orchestrator::{ParallelOrchestrator, StrategyMap, StrategyStep, ParallelOrchestratorConfig};
use llm_toolkit::agent::Agent;
use std::sync::Arc;
use std::time::Duration;

// Assume ResearchAgent and WriterAgent are defined and implement Agent + Send + Sync.
// For example:
// #[derive(Clone)]
// struct ResearchAgent;
// #[async_trait::async_trait]
// impl Agent for ResearchAgent { /* ... */ type Output = String; }
//
// struct WriterAgent;
// #[async_trait::async_trait]
// impl Agent for WriterAgent { /* ... */ type Output = String; }


#[tokio::main]
async fn main() {
    // Define a strategy where step 1 and 2 can run in parallel.
    let mut strategy = StrategyMap::new("Write article based on parallel research");

    strategy.add_step(StrategyStep::new(
        "step_1", "Research Topic A", "ResearchAgent",
        "Research the benefits of Rust for systems programming.", "topic_a_research",
    ));
    strategy.add_step(StrategyStep::new(
        "step_2", "Research Topic B", "ResearchAgent",
        "Research the benefits of Rust for web assembly.", "topic_b_research",
    ));

    // Step 3 depends on the outputs of step 1 and 2.
    strategy.add_step(StrategyStep::new(
        "step_3", "Write Article", "WriterAgent",
        r#"Write a comprehensive article based on the following research:
Topic A: {{ topic_a_research }}
Topic B: {{ topic_b_research }}"#,
        "final_article",
    ));

    // Configure the orchestrator with a 5-minute timeout per step.
    let config = ParallelOrchestratorConfig::new()
        .with_step_timeout(Duration::from_secs(300));

    let mut orchestrator = ParallelOrchestrator::with_config(strategy, config);

    // IMPORTANT: Agents MUST be thread-safe (Send + Sync).
    // orchestrator.add_agent("ResearchAgent", Arc::new(ResearchAgent));
    // orchestrator.add_agent("WriterAgent", Arc::new(WriterAgent));

    // let result = orchestrator.execute("Write an article about Rust's versatility.").await.unwrap();
    // assert!(result.success);
    // println!("Final article: {:?}", result.context.get("final_article"));
}
```

#### ⚠️ Important: Agent Thread-Safety (`Send + Sync`)

To ensure thread safety, any agent added to the `ParallelOrchestrator` **must** implement the `Send` and `Sync` traits. The `add_agent` method enforces this at compile time, so you will get a clear error if you try to add a non-thread-safe agent.

This is necessary because the orchestrator may need to share agents across multiple threads to execute them concurrently. For agents that share internal state, use thread-safe primitives like `Arc` and `Mutex`.

