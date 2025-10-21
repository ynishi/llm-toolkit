//! Tracing tests for ParallelOrchestrator
//!
//! These tests verify that structured tracing events and spans are properly
//! emitted during workflow execution.

use llm_toolkit::agent::{Agent, AgentError, DynamicAgent, Payload};
use llm_toolkit::orchestrator::{ParallelOrchestrator, StrategyMap, StrategyStep};
use serde_json::{Value as JsonValue, json};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::Level;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::fmt::format::FmtSpan;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Captures tracing output to a string for verification
#[derive(Clone)]
struct TestWriter {
    output: Arc<std::sync::Mutex<Vec<u8>>>,
}

impl TestWriter {
    fn new() -> Self {
        Self {
            output: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn get_output(&self) -> String {
        let bytes = self.output.lock().unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }
}

impl std::io::Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.output.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.output.lock().unwrap().flush()
    }
}

impl<'a> MakeWriter<'a> for TestWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

// ============================================================================
// Mock Agents
// ============================================================================

#[derive(Clone)]
struct SimpleAgent {
    name: String,
    output: JsonValue,
}

impl SimpleAgent {
    fn new(name: impl Into<String>, output: JsonValue) -> Self {
        Self {
            name: name.into(),
            output,
        }
    }
}

#[async_trait::async_trait]
impl Agent for SimpleAgent {
    type Output = JsonValue;

    fn expertise(&self) -> &str {
        "Simple test agent"
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.output.clone())
    }
}

#[async_trait::async_trait]
impl DynamicAgent for SimpleAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn expertise(&self) -> &str {
        "Simple test agent"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
        self.execute(input).await
    }
}

// ============================================================================
// Tracing Tests
// ============================================================================

#[tokio::test]
async fn test_top_level_span_created() {
    let writer = TestWriter::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_ansi(false)
        .with_writer(writer.clone())
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);

    let mut strategy = StrategyMap::new("Test".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(SimpleAgent::new("Agent1", json!({"ok": true}))),
    );

    let _result = orchestrator
        .execute("test task", CancellationToken::new(), None, None)
        .await
        .unwrap();

    let output = writer.get_output();

    // Verify top-level span exists
    assert!(
        output.contains("parallel_orchestrator_execute"),
        "Top-level span 'parallel_orchestrator_execute' not found in output:\n{}",
        output
    );

    // Verify span has task attribute
    assert!(
        output.contains("task=test task") || output.contains("task=\"test task\""),
        "Span should include task attribute in output:\n{}",
        output
    );
}

#[tokio::test]
async fn test_wave_spans_created() {
    let writer = TestWriter::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_ansi(false)
        .with_writer(writer.clone())
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);

    let mut strategy = StrategyMap::new("Wave Test".to_string());

    // Two independent steps -> wave 1
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Step 2".to_string(),
        "Agent2".to_string(),
        "{{ task }}".to_string(),
        "Output 2".to_string(),
    ));

    // Dependent step -> wave 2
    strategy.add_step(StrategyStep::new(
        "step_3".to_string(),
        "Step 3".to_string(),
        "Agent3".to_string(),
        "{{ step_1_output }}".to_string(),
        "Output 3".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(SimpleAgent::new("Agent1", json!({"ok": 1}))),
    );
    orchestrator.add_agent(
        "Agent2",
        Arc::new(SimpleAgent::new("Agent2", json!({"ok": 2}))),
    );
    orchestrator.add_agent(
        "Agent3",
        Arc::new(SimpleAgent::new("Agent3", json!({"ok": 3}))),
    );

    let _result = orchestrator
        .execute("wave test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    let output = writer.get_output();

    // Verify wave spans exist
    assert!(
        output.contains("wave"),
        "Wave spans not found in output:\n{}",
        output
    );

    // Should have at least 2 waves
    assert!(
        output.matches("wave_number=1").count() >= 1,
        "Wave 1 not found in output:\n{}",
        output
    );
}

#[tokio::test]
async fn test_step_spans_created() {
    let writer = TestWriter::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_ansi(false)
        .with_writer(writer.clone())
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);

    let mut strategy = StrategyMap::new("Step Span Test".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(SimpleAgent::new("Agent1", json!({"ok": true}))),
    );

    let _result = orchestrator
        .execute("step test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    let output = writer.get_output();

    // Verify per-step span exists
    assert!(
        output.contains("parallel_step"),
        "Per-step span 'parallel_step' not found in output:\n{}",
        output
    );

    // Verify step_id attribute
    assert!(
        output.contains("step_id=step_1") || output.contains("step_id=\"step_1\""),
        "Step span should include step_id attribute in output:\n{}",
        output
    );

    // Verify agent_name attribute
    assert!(
        output.contains("agent_name=Agent1") || output.contains("agent_name=\"Agent1\""),
        "Step span should include agent_name attribute in output:\n{}",
        output
    );
}

#[tokio::test]
async fn test_state_transition_events() {
    let writer = TestWriter::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(writer.clone())
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);

    let mut strategy = StrategyMap::new("State Events Test".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(SimpleAgent::new("Agent1", json!({"ok": true}))),
    );

    let _result = orchestrator
        .execute("state test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    let output = writer.get_output();

    // Verify state transition events
    assert!(
        output.contains("Step marked as Ready"),
        "Ready state event not found in output:\n{}",
        output
    );

    assert!(
        output.contains("Step execution started"),
        "Running state event not found in output:\n{}",
        output
    );

    assert!(
        output.contains("Step completed successfully"),
        "Completed state event not found in output:\n{}",
        output
    );
}

#[tokio::test]
async fn test_failure_events() {
    let writer = TestWriter::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(writer.clone())
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);

    #[derive(Clone)]
    struct FailingAgent;

    #[async_trait::async_trait]
    impl Agent for FailingAgent {
        type Output = JsonValue;

        fn expertise(&self) -> &str {
            "Failing agent"
        }

        async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
            Err(AgentError::ExecutionFailed(
                "Intentional failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl DynamicAgent for FailingAgent {
        fn name(&self) -> String {
            "FailingAgent".to_string()
        }

        fn expertise(&self) -> &str {
            "Failing agent"
        }

        async fn execute_dynamic(&self, input: Payload) -> Result<JsonValue, AgentError> {
            self.execute(input).await
        }
    }

    let mut strategy = StrategyMap::new("Failure Test".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Failing Step".to_string(),
        "FailAgent".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Dependent Step".to_string(),
        "Agent2".to_string(),
        "{{ step_1_output }}".to_string(),
        "Output 2".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent("FailAgent", Arc::new(FailingAgent));
    orchestrator.add_agent(
        "Agent2",
        Arc::new(SimpleAgent::new("Agent2", json!({"ok": 2}))),
    );

    let _result = orchestrator
        .execute("failure test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    let output = writer.get_output();

    // Verify failure event
    assert!(
        output.contains("Step failed"),
        "Step failure event not found in output:\n{}",
        output
    );

    // Verify skipped event
    assert!(
        output.contains("Step skipped due to failed dependency"),
        "Skipped dependency event not found in output:\n{}",
        output
    );
}

#[tokio::test]
async fn test_span_hierarchy() {
    let writer = TestWriter::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_span_events(FmtSpan::NEW | FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE)
        .with_ansi(false)
        .with_writer(writer.clone())
        .finish();

    let _guard = tracing::subscriber::set_default(subscriber);

    let mut strategy = StrategyMap::new("Hierarchy Test".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Step 1".to_string(),
        "Agent1".to_string(),
        "{{ task }}".to_string(),
        "Output 1".to_string(),
    ));

    let mut orchestrator = ParallelOrchestrator::new(strategy);
    orchestrator.add_agent(
        "Agent1",
        Arc::new(SimpleAgent::new("Agent1", json!({"ok": true}))),
    );

    let _result = orchestrator
        .execute("hierarchy test", CancellationToken::new(), None, None)
        .await
        .unwrap();

    let output = writer.get_output();

    // Verify all three levels of spans exist
    assert!(
        output.contains("parallel_orchestrator_execute"),
        "Top-level span missing"
    );
    assert!(output.contains("wave"), "Wave span missing");
    assert!(output.contains("parallel_step"), "Step span missing");

    // The output should show nested structure (exact format depends on subscriber configuration)
    // We just verify all levels are present
}
