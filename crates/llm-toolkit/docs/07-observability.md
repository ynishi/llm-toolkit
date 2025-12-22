## Observability


The `observability` module (available with the `agent` feature) provides a simple and powerful way to gain visibility into your LLM agent workflows. Built on the industry-standard `tracing` crate, it enables you to capture detailed execution traces, performance metrics, and contextual metadata with minimal setup.

### Features

- **One-Line Initialization**: Get started with a single function call
- **Zero Boilerplate**: No need to manually configure `tracing_subscriber`
- **Automatic Instrumentation**: All agents created with `#[derive(Agent)]` or `#[agent]` are automatically instrumented
- **Structured Logging**: Captures agent names, expertise, and execution spans
- **Flexible Output**: Log to console or file
- **OpenTelemetry Ready**: Built on `tracing`, making it easy to integrate with observability platforms like Jaeger, Datadog, and Honeycomb in the future

### Quick Start

```rust
use llm_toolkit::observability::{self, ObservabilityConfig, LogTarget};
use tracing::Level;

fn main() {
    // Initialize observability with DEBUG level logging to console
    observability::init(ObservabilityConfig {
        level: Level::DEBUG,
        target: LogTarget::Console,
    }).expect("Failed to initialize observability");

    // All agent executions will now emit detailed traces
    // ...rest of your application
}
```

### Configuration Options

#### Log Levels

Choose the appropriate log level for your needs:

```rust
use tracing::Level;

// Most verbose - captures all execution details
Level::TRACE

// Detailed information useful for debugging
Level::DEBUG

// General informational messages (default)
Level::INFO

// Warnings about potential issues
Level::WARN

// Only errors
Level::ERROR
```

#### Output Targets

Log to console or file:

```rust
use llm_toolkit::observability::LogTarget;

// Console output (stdout)
LogTarget::Console

// File output
LogTarget::File("logs/agent_execution.log".to_string())
```

### What Gets Traced

With observability enabled, you'll see:

- **Agent Execution Spans**: Each agent execution creates a span with:
  - `agent.name`: The agent's struct name
  - `agent.expertise`: The agent's expertise description
  - `agent.role`: For `PersonaAgent`, the persona's role

- **Timing Information**: Duration of each agent execution
- **Hierarchical Context**: Nested spans for composed agents (e.g., `PersonaAgent` wrapping another agent)

### Example Output

```
2024-01-15T10:30:00.123Z DEBUG agent.execute{agent.name="ContentWriter" agent.expertise="Writing articles"}: llm_toolkit::agent: executing agent
2024-01-15T10:30:02.456Z DEBUG agent.execute{agent.name="ContentWriter" agent.expertise="Writing articles"}: llm_toolkit::agent: agent completed duration=2.333s
```

### Future Enhancements

- **Orchestrator Instrumentation**: Tracing for workflow steps and strategies
- **Dialogue Instrumentation**: Visibility into multi-turn conversations
- **OpenTelemetry Integration**: Direct export to observability platforms
- **Custom Metrics**: Performance counters and histograms for agent execution

