### Context-Aware Detection & Enrichment


The orchestrator provides a sophisticated context detection system that automatically infers high-level context (task health, task type, user states) from execution patterns. This context is then used to enrich agent payloads, enabling agents to provide more contextually appropriate responses.

#### Architecture

The context detection system uses a **hierarchical three-layer architecture**:

1. **EnvContext** (Raw Runtime Context)
   - Raw execution metrics: `redesign_count`, `step_failures`, `success_rate`
   - Journal summary: consecutive failures, total steps
   - Strategy phase tracking
   - Timeline-based: each step execution can have its own EnvContext snapshot

2. **DetectedContext** (Inferred Semantic Context)
   - `task_type`: Inferred task type (e.g., "security-review", "debug", "implementation")
   - `task_health`: Health status (`OnTrack`, `AtRisk`, `OffTrack`)
   - `user_states`: Inferred user states (e.g., "confused", "frustrated", "on-track")
   - `confidence`: Confidence scores for each inference
   - `detected_by`: List of detectors that contributed to this context

3. **ExecutionContext** (Unified Wrapper)
   - Combines EnvContext and DetectedContext
   - Provides unified access to both raw and inferred context
   - Stored as timeline: `Vec<ExecutionContext>` in Payload

#### Layered Detection Pattern

The system supports **progressive context enrichment** through multiple detector passes:

**Layer 1: Rule-Based Detection** (Fast, ~1ms overhead)
- Hardcoded heuristics for deterministic detection
- Examples:
  - `redesign_count > 2` → `TaskHealth::AtRisk`
  - `consecutive_failures > 3` → `user_states: ["frustrated"]`
  - `success_rate < 0.5` → `TaskHealth::OffTrack`

**Layer 2: Agent-Based Detection** (Semantic, 1 LLM call per enrichment)
- Uses any `Agent<Output = String>` implementation (typically lightweight LLM like Haiku)
- Performs semantic analysis of Payload contents
- Can detect nuanced patterns beyond keyword matching
- Enriches context with LLM's language understanding

**Layer 3: Custom Detection** (Domain-Specific)
- Implement the `ContextDetector` trait for specialized detection
- Can leverage domain knowledge or external systems

#### Orchestrator Integration

The orchestrator provides three detection modes via the `DetectionMode` enum:

```rust
use llm_toolkit::orchestrator::{Orchestrator, DetectionMode};

// Mode 1: No detection (default, zero overhead)
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.set_detection_mode(DetectionMode::None);

// Mode 2: Rule-based detection only (~1ms overhead per step)
orchestrator.set_detection_mode(DetectionMode::RuleBased);

// Mode 3: Agent-based semantic detection (1 LLM call per step)
orchestrator.set_detection_mode(DetectionMode::AgentBased);
```

**How it works:**

1. Before each step execution, the orchestrator captures the current `EnvContext` (redesign count, journal metrics, etc.)
2. If detection is enabled, the orchestrator runs the configured detector(s) to infer `DetectedContext`
3. The enriched Payload (with both EnvContext and DetectedContext) is passed to the step's agent
4. The agent can access context via `payload.latest_env_context()` and `payload.latest_detected_context()`

#### Manual Detection Usage

You can also use detectors manually for more control:

```rust
use llm_toolkit::agent::{Payload, EnvContext, DetectedContext};
use llm_toolkit::agent::context_detector::{ContextDetector, DetectContextExt};
use llm_toolkit::agent::rule_based_detector::RuleBasedDetector;
use llm_toolkit::agent::agent_based_detector::AgentBasedDetector;
use llm_toolkit::agent::impls::ClaudeCodeAgent;

// Layer 1: Rule-based detection
let rule_detector = RuleBasedDetector::new();
let env_context = EnvContext::new().with_redesign_count(3);
let payload = Payload::text("The authentication system keeps failing")
    .with_env_context(env_context)
    .detect_with(&rule_detector).await?;

// Layer 2: Agent-based semantic detection
let agent = ClaudeCodeAgent::new();
let llm_detector = AgentBasedDetector::new(agent);
let payload = payload.detect_with(&llm_detector).await?;

// Access detected context
if let Some(detected) = payload.latest_detected_context() {
    println!("Task type: {:?}", detected.task_type);
    println!("Task health: {:?}", detected.task_health);
    println!("User states: {:?}", detected.user_states);
    println!("Detected by: {:?}", detected.detected_by);
}
```

#### Custom Detectors

Implement the `ContextDetector` trait for domain-specific detection:

```rust
use llm_toolkit::agent::{ContextDetector, Payload, DetectedContext, AgentError};
use llm_toolkit::context::TaskHealth;
use async_trait::async_trait;

struct SecurityReviewDetector;

#[async_trait]
impl ContextDetector for SecurityReviewDetector {
    async fn detect(&self, payload: &Payload) -> Result<DetectedContext, AgentError> {
        let mut detected = DetectedContext::new();

        // Check for security-related keywords
        let text = payload.as_text().unwrap_or("");
        if text.contains("security") || text.contains("vulnerability") {
            detected = detected.with_task_type("security-review");
        }

        // Check environment context
        if let Some(env) = payload.latest_env_context() {
            if env.redesign_count > 2 {
                detected = detected.with_task_health(TaskHealth::AtRisk);
                detected = detected.with_user_state("needs-guidance");
            }
        }

        Ok(detected.detected_by("SecurityReviewDetector"))
    }

    fn name(&self) -> &str {
        "SecurityReviewDetector"
    }
}

// Use it
let detector = SecurityReviewDetector;
let payload = payload.detect_with(&detector).await?;
```

#### Performance Characteristics

| Detection Mode | Overhead per Step | LLM Calls | Best For |
|----------------|-------------------|-----------|----------|
| **None** | 0ms | 0 | Production default, cost-sensitive workflows |
| **RuleBased** | ~1ms | 0 | Fast detection, deterministic patterns |
| **AgentBased** | ~500-2000ms | 1 | Rich semantic understanding, complex patterns |

**Cost Analysis:**
- **RuleBased**: Zero API cost, negligible CPU overhead
- **AgentBased**: 1 LLM call per step execution (use lightweight models like Haiku for cost efficiency)

#### Timeline-Based Context Management

Context is stored as a timeline in the Payload, preserving the full history of context evolution:

```rust
// Access the latest context
if let Some(latest) = payload.latest_detected_context() {
    println!("Current task health: {:?}", latest.task_health);
}

// Access the full timeline (all past contexts)
for exec_ctx in payload.execution_context() {
    println!("Step: redesign_count={}, task_health={:?}",
             exec_ctx.env.redesign_count,
             exec_ctx.detected.as_ref().and_then(|d| d.task_health));
}
```

#### Use Cases

**Use Case 1: Adaptive Agent Behavior**
Agents can adjust their responses based on detected context:
```rust
#[derive(Agent)]
#[agent(expertise = "Help users debug issues", output = "String")]
struct DebuggerAgent;

impl DebuggerAgent {
    async fn execute_with_context(&self, payload: Payload) -> Result<String, AgentError> {
        // Check if user is frustrated
        let is_frustrated = payload.latest_detected_context()
            .map(|d| d.user_states.contains(&"frustrated".to_string()))
            .unwrap_or(false);

        if is_frustrated {
            // Provide more empathetic, step-by-step guidance
            return Ok("I understand this is frustrating. Let's break this down step by step...".to_string());
        }

        // Normal execution
        Ok("Here's the fix...".to_string())
    }
}
```

**Use Case 2: Workflow Monitoring**
Track workflow health over time:
```rust
let result = orchestrator.execute(task).await?;

// Analyze final context
if let Some(exec_ctx) = result.final_payload.as_ref()
    .and_then(|p| p.latest_execution_context())
{
    match exec_ctx.detected.as_ref().and_then(|d| d.task_health) {
        Some(TaskHealth::OffTrack) => {
            eprintln!("⚠️ Workflow completed but task is off-track!");
        }
        Some(TaskHealth::AtRisk) => {
            eprintln!("⚠️ Workflow completed but task is at risk");
        }
        _ => println!("✅ Workflow completed successfully"),
    }
}
```

**Use Case 3: Custom Detector Chains**
Combine multiple detectors for rich context:
```rust
let payload = Payload::text(user_request)
    .with_env_context(env_context)
    .detect_with(&rule_detector).await?        // Fast heuristics
    .detect_with(&security_detector).await?    // Domain-specific
    .detect_with(&llm_detector).await?;        // Semantic enrichment

// All detectors contribute to the final context
let detected = payload.latest_detected_context().unwrap();
println!("Detected by: {:?}", detected.detected_by);
// Output: ["RuleBasedDetector", "SecurityReviewDetector", "AgentBasedDetector"]
```

#### Key Design Decisions

1. **Timeline-Based**: Contexts are never overwritten; they form a timeline
   - Enables historical analysis
   - Preserves full evolution of workflow state

2. **Progressive Enrichment**: Detectors merge their results
   - Multiple detectors can contribute to a single DetectedContext
   - Later detectors add/override fields rather than replacing the entire context

3. **Zero-Cost Default**: Detection is opt-in
   - `DetectionMode::None` (default) has zero overhead
   - Production workflows can run without any detection cost

4. **Separation of Concerns**:
   - `EnvContext` = raw metrics (objective)
   - `DetectedContext` = inferences (subjective, with confidence scores)
   - Agents choose which to trust

5. **Async-First**: All detectors use `async fn` for consistency
   - Even rule-based detectors are async for trait uniformity
   - Enables future extensions (e.g., database lookups, API calls)

#### Testing

The context detection system includes comprehensive tests:

```bash
# Run context detection tests
cargo test --package llm-toolkit --lib agent::agent_based_detector
cargo test --package llm-toolkit --lib agent::rule_based_detector
cargo test --package llm-toolkit --lib agent::context_detector

# Run orchestrator integration tests with detection
cargo test --package llm-toolkit --lib orchestrator::tests::test_detection_mode
```

