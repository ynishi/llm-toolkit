#### Monitoring Orchestrator Execution with Tracing


The orchestrator emits structured logs using the `tracing` crate, allowing you to monitor workflow execution in real-time. You can capture these logs in JSON format and stream them to any destination.

**Example: JSON Log Streaming**

See `examples/orchestrator_streaming.rs` for a complete example that demonstrates:
- Setting up a custom `tracing` layer to capture orchestrator events
- Streaming logs to a channel in JSON format
- Pretty-printing execution events in real-time

```bash
cargo run --example orchestrator_streaming --features agent,derive
```

**Key Features:**
- **Structured Logging**: All orchestrator events (step execution, errors, redesigns) are emitted as structured logs
- **JSON Format**: Easy integration with log aggregation tools (e.g., ELK, Datadog, CloudWatch)
- **Real-time Streaming**: Monitor workflow progress as it happens using `tokio::sync::mpsc` channels
- **Custom Layers**: Implement your own `tracing::Layer` to route logs to any destination

**Basic Setup:**

```rust
use tracing_subscriber::prelude::*;
use tokio::sync::mpsc;

// Create a channel for log streaming
let (tx, mut rx) = mpsc::channel::<String>(100);

// Set up tracing subscriber with custom layer
let subscriber = tracing_subscriber::registry()
    .with(YourCustomLayer { sender: tx })
    .with(tracing_subscriber::filter::EnvFilter::new("info"));

tracing::subscriber::set_global_default(subscriber)?;

// Listen for events
tokio::spawn(async move {
    while let Some(event) = rx.recv().await {
        println!("{}", event); // Process log event
    }
});

// Execute orchestrator - logs will be streamed automatically
let result = orchestrator.execute(task).await;
```

For the complete implementation, see the example file at `crates/llm-toolkit/examples/orchestrator_streaming.rs`.

