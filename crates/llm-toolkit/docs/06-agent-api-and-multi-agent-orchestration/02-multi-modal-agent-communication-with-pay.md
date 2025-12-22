### Multi-Modal Agent Communication with Payload


The `execute()` method accepts a `Payload` type that supports multi-modal content including text and images. This enables agents to process both textual instructions and visual inputs.

**Basic Usage (Text Only):**

```rust
use llm_toolkit::agent::Agent;

// String automatically converts to Payload for backward compatibility
let result = agent.execute("Analyze this text".to_string().into()).await?;

// Or use Payload explicitly
use llm_toolkit::agent::Payload;
let payload = Payload::text("Analyze this text");
let result = agent.execute(payload).await?;
```

**Multi-Modal Usage (Text + Images):**

```rust
use llm_toolkit::agent::Payload;
use llm_toolkit::attachment::Attachment;

// Combine text and attachments
let payload = Payload::text("What's in this image?")
    .with_attachment(Attachment::local("/path/to/image.png"));

let result = agent.execute(payload).await?;

// Or from raw image data
let image_bytes = std::fs::read("/path/to/image.png")?;
let payload = Payload::text("Describe this screenshot")
    .with_attachment(Attachment::in_memory(image_bytes));
```

**Backward Compatibility:**

All existing code using `String` continues to work thanks to automatic conversion:

```rust
// This still works unchanged
let result = agent.execute("Simple text intent".to_string().into()).await?;
```

**CLI Agents Attachment Support:**

All CLI agents (`GeminiAgent`, `ClaudeCodeAgent`, and `CodexAgent`) support attachments by automatically writing them to temporary files and passing the file paths to the CLI tools:

```rust
use llm_toolkit::agent::impls::{GeminiAgent, ClaudeCodeAgent, CodexAgent};
use llm_toolkit::agent::{Agent, Payload};
use llm_toolkit::attachment::Attachment;

// GeminiAgent with attachments
let gemini = GeminiAgent::new();
let payload = Payload::text("Analyze this diagram")
    .with_attachment(Attachment::local("architecture.png"));
let result = gemini.execute(payload).await?;

// ClaudeCodeAgent with multiple attachments
let claude = ClaudeCodeAgent::new();
let payload = Payload::text("Review these files")
    .with_attachment(Attachment::local("code.rs"))
    .with_attachment(Attachment::local("tests.rs"));
let result = claude.execute(payload).await?;

// CodexAgent with image attachments and sandbox
let codex = CodexAgent::new()
    .with_sandbox("workspace-write")
    .with_approval_policy("on-failure");
let payload = Payload::text("Fix the bug shown in this screenshot")
    .with_attachment(Attachment::local("error.png"));
let result = codex.execute(payload).await?;

// Custom attachment directory (useful for debugging)
let gemini = GeminiAgent::new()
    .with_attachment_dir("/tmp/my-attachments")
    .with_keep_attachments(true);  // Don't auto-delete temp files

// In-memory attachments are also supported
let image_data = std::fs::read("screenshot.png")?;
let payload = Payload::text("What's in this screenshot?")
    .with_attachment(Attachment::in_memory(image_data));
let result = claude.execute(payload).await?;

// CodexAgent full-auto mode with web search
let codex = CodexAgent::new()
    .full_auto()  // on-failure approval + workspace-write sandbox
    .with_search(true);
let result = codex.execute("Research the latest Rust async patterns".to_string().into()).await?;
```

**How it works:**
- Attachments are written to a temporary directory (either system temp or a custom dir via `with_attachment_dir()`)
- Each session gets a unique subdirectory to prevent conflicts
- File paths are appended to the prompt text before sending to the CLI tool
- Temporary files are automatically cleaned up after execution (unless `with_keep_attachments(true)` is set)
- Local files are copied to temp dir; in-memory data is written to temp files
- Remote URLs are logged with a warning and skipped (not yet supported)

#### Defining Agents: Two Approaches

`llm-toolkit` provides two ways to define agents, each optimized for different use cases:

##### 1. ⚠️ Simple Agents with `#[derive(Agent)]` (DEPRECATED - Use `#[agent(...)]` instead)

**⚠️ DEPRECATED:** This approach is deprecated as of v0.59.0 and will be removed in v0.60.0 (Q2 2025).

**Critical Issue:** The `#[derive(Agent)]` macro does NOT inject expertise into prompts, meaning the LLM never sees your expertise definition. This leads to poor results.

**Migration:** Simply remove `#[derive(Agent)]` and use only `#[agent(...)]` (see section 2 below).

For historical reference, the old derive macro syntax was:

```rust
use llm_toolkit::Agent;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct ArticleDraft {
    title: String,
    body: String,
    references: Vec<String>,
}

// Simple stateless agent
#[derive(Agent)]
#[agent(
    expertise = "Research topics and generate well-structured article drafts with citations",
    description = "Article research and drafting specialist",  // Optional: lightweight summary
    capabilities = ["research", "writing", "citations"],  // Optional: explicit capabilities
    output = "ArticleDraft"
)]
struct ContentSynthesizerAgent;

// Usage - extremely simple
#[tokio::main]
async fn main() {
    let agent = ContentSynthesizerAgent;
    let result: ArticleDraft = agent.execute("Write about Rust async/await".to_string().into()).await.unwrap();
    println!("Generated: {}", result.title);
}
```

**Best Practice: Writing Effective Expertise Descriptions**

The `expertise` field should describe the agent's capabilities in **natural language only**. Do NOT include template placeholder syntax like `{{ variable }}` in the expertise string.

❌ **Incorrect:**
```rust
#[agent(
    expertise = "Processes {{ strategy_json }} and generates reports",
    output = "Report"
)]
```
**Problem:** When the orchestrator generates strategies, the LLM sees these `{{ }}` patterns and may confuse them with actual placeholders that need to be filled, leading to incorrect intent generation.

✅ **Correct:**
```rust
#[agent(
    expertise = "Processes strategy details provided in the input and generates comprehensive reports. \
                 Input should include strategy goals, constraints, and context data.",
    output = "Report"
)]
```
**Why this works:** The orchestrator's strategy generation LLM reads this natural language description and automatically creates appropriate intent templates like `"Process the following strategy: {{ strategy_data }}"`. The LLM understands what inputs the agent needs and generates the correct placeholders in the strategy's `intent_template` field.

**Key principle:** The `expertise` describes capabilities; the orchestrator creates the actual intent templates dynamically based on those capabilities.

**Description and Capabilities Attributes:**

The `#[agent]` macro supports two additional optional attributes for better orchestrator integration:

```rust
#[derive(Agent)]
#[agent(
    expertise = "You are a file operations specialist with expertise in reading, writing, and managing files...",
    description = "File operations specialist",  // Optional: lightweight routing summary
    capabilities = ["read_file", "write_file", "delete_file"],  // Optional: explicit capabilities
    output = "FileOperationResult"
)]
struct FileAgent;
```

- **`description`** (Optional): A lightweight 1-2 sentence summary used by orchestrators for agent selection
  - If omitted, automatically generated from the first 100 characters of `expertise`
  - Orchestrators use this for routing decisions instead of reading the full expertise text
  - Example: `"File operations specialist"` vs. full expertise paragraph

- **`capabilities`** (Optional): Explicit list of capabilities the agent provides
  - Declared as string array: `capabilities = ["tool1", "tool2"]`
  - If omitted, `capabilities()` returns `None` (no capabilities displayed)
  - Orchestrators can filter agents based on required capabilities
  - Priority: Explicit declaration > (Future: Expertise type auto-extraction) > None

**Behavior:**
- `agent.description()` → Returns lightweight summary (for orchestrator routing)
- `agent.expertise()` → Returns full expertise text (for LLM execution)
- `agent.capabilities()` → Returns `Option<Vec<Capability>>` (for capability-based selection)

**Note:** The `description` and `capabilities` attributes work the same way as in the `#[agent]` macro (see above). Both attributes are optional and follow the same behavior:
- `description` auto-generates from the first 100 characters of `expertise` if omitted
- `capabilities` returns `None` if not specified

**Features:**
- ✅ Simplest possible interface
- ✅ Minimal boilerplate
- ✅ Perfect for prototyping
- ✅ Supports `description` and `capabilities` attributes for orchestrator integration
- ⚠️ Creates internal agent on each `execute()` call (stateless)

**Automatic JSON Schema Enforcement:**

When using `#[derive(Agent)]` with a structured output type (non-String), the macro automatically adds JSON schema instructions to the agent's expertise. This dramatically improves LLM compliance and reduces parse errors.

```rust
use llm_toolkit::{Agent, ToPrompt};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, ToPrompt)]
#[prompt(mode = "full")]
struct ReviewResult {
    /// Overall quality score from 0 to 100
    quality_score: u8,

    /// List of identified issues
    issues: Vec<String>,

    /// Actionable recommendations for improvement
    recommendations: Vec<String>,
}

#[derive(Agent)]
#[agent(
    expertise = "Review code quality and provide detailed feedback",
    output = "ReviewResult"
)]
struct CodeReviewAgent;

// The agent's expertise() method automatically returns:
// "Review code quality and provide detailed feedback
//
// IMPORTANT: Respond with valid JSON matching this schema:
//
// /**
//  * (struct documentation if present)
//  */
// type ReviewResult = {
//   quality_score: number;  // Overall quality score from 0 to 100
//   issues: string[];  // List of identified issues
//   recommendations: string[];  // Actionable recommendations for improvement
// }"
```

**Schema Generation Strategy (3-Tier Auto-Inference):**

1. **With `ToPrompt` + doc comments** → Detailed schema with field descriptions
   - Requires: `#[derive(ToPrompt)]` + `#[prompt(mode = "full")]`
   - Best experience: Full field-level documentation

2. **With `ToPrompt` (no doc comments)** → Basic schema with field names
   - Requires: `#[derive(ToPrompt)]` + `#[prompt(mode = "full")]`
   - Good: Type-safe field names

3. **String output** → No JSON enforcement
   - For plain text responses
   - The macros skip `extract_json()`/`serde_json::from_str()` and return the LLM text as-is
   - If the response happens to be a JSON string (e.g., `"hello world"` or inside ```` ```json` blocks), only the surrounding quotes are stripped; otherwise no additional parsing is applied
   - Helper: `llm_toolkit::agent::normalize_string_output` (available if you want the same behavior in custom agents)

**Recommendation:** Always use `#[derive(ToPrompt)]` with `#[prompt(mode = "full")]` for structured outputs to get the best LLM compliance.

**Nested Schema Expansion:**

The schema generation automatically includes complete type definitions for nested types that implement `ToPrompt`, including both `Vec<T>` and regular nested objects. This ensures LLMs receive all necessary schema information in a single call:

```rust
#[derive(Serialize, Deserialize, ToPrompt)]
#[prompt(mode = "full")]
pub struct EvaluationResult {
    /// The rule being checked
    pub rule: String,
    /// Whether this specific rule passed
    pub passed: bool,
}

#[derive(Serialize, Deserialize, ToPrompt)]
#[prompt(mode = "full")]
pub struct ProducerOutput {
    /// Whether the evaluation passed all checks
    pub evaluation_passed: bool,
    /// List of evaluation results for each rule
    pub results: Vec<EvaluationResult>,
}

// Generated schema for ProducerOutput (single call):
// type EvaluationResult = {
//   rule: string;  // The rule being checked
//   passed: boolean;  // Whether this specific rule passed
// }
//
// type ProducerOutput = {
//   evaluation_passed: boolean;  // Whether the evaluation passed all checks
//   results: EvaluationResult[];  // List of evaluation results for each rule
// }
```

**How it works:**

- The macro detects nested types (both `Vec<T>` and regular fields) at compile time
- At runtime (first call only), it collects `prompt_schema()` from all nested types
- Nested type definitions are placed **before** the main type definition
- Duplicates are automatically removed (same type used multiple times)
- Result is cached with `OnceLock` for performance (zero cost after first call)
- LLM receives complete schema information with all necessary type definitions

**Nested Objects:**

The same expansion works for regular nested objects (not just Vec):

```rust
#[derive(Serialize, Deserialize, ToPrompt)]
#[prompt(mode = "full")]
pub struct Emblem {
    /// The name of the emblem
    pub name: String,
    /// A description of the emblem
    pub description: String,
}

#[derive(Serialize, Deserialize, ToPrompt)]
#[prompt(mode = "full")]
pub struct EmblemResponse {
    /// An obvious, straightforward emblem
    pub obvious_emblem: Emblem,
    /// A creative, unexpected emblem
    pub creative_emblem: Emblem,
}

// Generated schema for EmblemResponse (single call):
// type Emblem = {
//   name: string;  // The name of the emblem
//   description: string;  // A description of the emblem
// }
//
// type EmblemResponse = {
//   obvious_emblem: Emblem;  // An obvious, straightforward emblem
//   creative_emblem: Emblem;  // A creative, unexpected emblem
// }
```

**Collections and Option Types:**

The schema expansion also supports `Option<T>`, `HashMap<K, V>`, `HashSet<T>`, and their combinations:

```rust
#[derive(Serialize, Deserialize, ToPrompt)]
pub enum Priority {
    Low,
    Medium,
    High,
}

#[derive(Serialize, Deserialize, ToPrompt)]
#[prompt(mode = "full")]
pub struct TaskCollection {
    /// Optional list of tags
    pub tags: Option<Vec<String>>,
    /// Map of task IDs to their priorities
    pub priorities: HashMap<String, Priority>,
    /// Set of assigned user IDs
    pub assigned_users: HashSet<String>,
    /// Optional map of metadata
    pub metadata: Option<HashMap<String, Priority>>,
}

// Generated schema for TaskCollection:
// type Priority =
//   | "Low"
//   | "Medium"
//   | "High";
//
// type TaskCollection = {
//   tags: string[] | null;  // Optional list of tags
//   priorities: Record<string, Priority>;  // Map of task IDs to their priorities
//   assigned_users: string[];  // Set of assigned user IDs
//   metadata: Record<string, Priority> | null;  // Optional map of metadata
// }
```

**Note:** For `HashMap<K, V>` and `BTreeMap<K, V>`, only the value type `V` is expanded if it's a custom type. The key type `K` is always treated as `string` in the TypeScript schema (using `Record<string, V>`). If you need custom enum keys, consider using the enum as a value instead.

**How it works:**

- The macro detects field types at compile time
- For `Vec<T>`: generates TypeScript array syntax `T[]` and includes `T` definition
- For `Option<T>`: generates TypeScript nullable syntax `T | null` and includes `T` definition if non-primitive
- For `HashMap<K, V>` / `BTreeMap<K, V>`: generates TypeScript `Record<string, V>` and includes `V` definition if non-primitive (Note: Key type `K` is always treated as `string` in the schema)
- For `HashSet<T>` / `BTreeSet<T>`: generates TypeScript array syntax `T[]` and includes `T` definition if non-primitive
- For nested collections (e.g., `Option<HashMap<String, T>>`): recursively unwraps and includes inner type `T` definition
- For nested objects: generates TypeScript type reference `TypeName` and includes its full definition
- For primitives: generates TypeScript primitive types (`string`, `number`, `boolean`, etc.)
- All type definitions are bundled together in the correct dependency order

**Benefits:**

- ✅ **Complete schema information** - LLM receives all type definitions in one call
- ✅ **Zero manual work** - No need to manually concatenate schemas
- ✅ **Type-driven design** - Rust types directly translate to LLM-friendly schemas
- ✅ **Prevents parse errors** - LLM knows exactly what fields are required in nested objects
- ✅ **Clean, readable output** - TypeScript-style syntax that LLMs understand well
- ✅ **Industry-standard format** - Uses familiar TypeScript syntax for better LLM comprehension

**Why This Matters:**

Without complete type definitions, LLMs guess field names and types, leading to parse errors like:
- `missing field 'age'` - LLM didn't know the field was required
- Wrong field names - LLM invented fields not in the schema
- Wrong types - LLM used `string` instead of `number`

With complete type definitions included, the LLM has perfect information and generates correct output.

**Automatic Retry on Transient Errors:**

All agents automatically retry on transient errors (ParseError, ProcessError, IoError) without any configuration:

```rust
#[derive(Agent)]
#[agent(
    expertise = "Extract data from documents",
    output = "ExtractedData"
)]
struct DataExtractorAgent;

// Automatically retries up to 3 times on:
// - ParseError: LLM output malformed
// - ProcessError: Process communication issues (including 429 rate limiting)
// - IoError: Temporary I/O failures
//
// Intelligent Retry Delay (3-Priority System):
// Priority 1: Server-provided retry_after (e.g., 90s from Retry-After header)
// Priority 2: 429 fallback - exponential backoff capped at 60s (2^attempt, max 60s)
// Priority 3: Other errors - linear backoff (100ms × attempt)
// All delays use Full Jitter (random 0~delay) to prevent thundering herd
//
// Example with 429 rate limiting:
// - Attempt 1 fails (429 + retry_after=60s) → wait ~30s (jittered) → retry
// - Attempt 2 fails (429, no retry_after) → wait ~1-2s (exponential + jitter) → retry
// - Attempt 3 fails → return error
```

**Customizing Retry Behavior:**

```rust
// Increase retry attempts for critical operations
#[agent(
    expertise = "...",
    output = "MyOutput",
    max_retries = 5  // Default is 3
)]
struct ResilientAgent;

// Disable retry for fast-fail scenarios
#[agent(
    expertise = "...",
    output = "MyOutput",
    max_retries = 0  // No retry
)]
struct NoRetryAgent;
```

**RetryAgent Wrapper - Add Retry to Any Agent:**

For production use cases where you need more control over retry behavior, use the `RetryAgent` decorator to wrap any existing agent:

```rust
use llm_toolkit::agent::impls::{ClaudeCodeAgent, RetryAgent};

// Wrap any agent with retry logic
let base_agent = ClaudeCodeAgent::new();
let retry_agent = RetryAgent::new(base_agent, 5); // Max 5 retries

// The wrapper handles all retry logic automatically
let result = retry_agent.execute(payload).await?;

// RetryAgent follows the same 3-priority delay system:
// - Server retry_after takes highest priority
// - 429 errors use exponential backoff (capped at 60s)
// - Other errors use linear backoff (100ms × attempt)
```

**Benefits of RetryAgent:**
- ✅ **Decorator Pattern**: Wrap any `Agent` implementation without modification
- ✅ **Unified Retry Logic**: Same retry mechanism used by macros (DRY principle)
- ✅ **Production-Ready**: Full control over max_retries and retry behavior
- ✅ **429 Rate Limiting**: Intelligent handling of server-provided retry delays
- ✅ **Zero Configuration**: Works out-of-the-box with sensible defaults

**Design Philosophy:**

Agent-level retries are intentionally **simple and limited** (2-3 attempts by default):
- **Fail fast**: Quickly report errors to the orchestrator
- **Orchestrator is smarter**: Has broader context for complex error recovery
  - Try different agents
  - Redesign strategy
  - Escalate to human
- **System stability**: Simple local retries + complex orchestration at the top = robust system

This design aligns with the Orchestrator's 3-stage error recovery (Tactical → Full Redesign → Human Escalation).

**Advanced: Server-Provided Retry Delays**

When LLM APIs return 429 rate limiting errors with a `Retry-After` header, agents automatically respect the server-specified delay:

```rust
use llm_toolkit::agent::{AgentError, ProcessError};
use std::time::Duration;

// Example: Creating a 429 error with retry_after
let error = AgentError::process_error_with_retry_after(
    429,
    "Rate limit exceeded",
    true,
    Duration::from_secs(90)
);

// The retry mechanism will:
// 1. Extract retry_after (90s)
// 2. Apply Full Jitter (random 0~90s)
// 3. Wait before retrying
//
// This prevents overwhelming rate-limited APIs and respects server guidance
```

##### 2. ✅ Production Agents with `#[agent(...)]` (RECOMMENDED - Use This!)

**This is the recommended way to define agents.** The `#[agent(...)]` attribute macro:
- ✅ Automatically injects expertise into prompts (the LLM sees your expertise!)
- ✅ Generates `Default` implementation for easy instantiation
- ✅ Supports generic inner agents for testing with mocks
- ✅ Provides better composability with PersonaAgent
- ✅ Full feature set for production use

**Quick Start:**

```rust
use llm_toolkit::agent::impls::ClaudeCodeAgent;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct ArticleDraft {
    title: String,
    body: String,
    references: Vec<String>,
}

// Advanced agent with Generic support
#[llm_toolkit_macros::agent(
    expertise = "Research topics and generate well-structured article drafts with citations",
    output = "ArticleDraft"
)]
struct ContentSynthesizerAgent;

#[tokio::main]
async fn main() {
    // Method 1: Using Default
    let agent = ContentSynthesizerAgent::default();

    // Method 2: Convenience constructor with specific model
    let agent = ContentSynthesizerAgent::with_claude_model("opus-4");

    // Method 3: Inject custom agent
    let custom_claude = ClaudeCodeAgent::new().with_model_str("sonnet-4.5");
    let agent = ContentSynthesizerAgent::new(custom_claude);

    let result: ArticleDraft = agent.execute("Write about Rust async/await".to_string().into()).await.unwrap();
    println!("Generated: {}", result.title);
}
```

**Practical Injection Examples:**

```rust
use llm_toolkit::agent::impls::{ClaudeCodeAgent, GeminiAgent};

// Example 1: Environment-based agent selection
fn create_agent(env: &str) -> ContentSynthesizerAgent {
    match env {
        "production" => {
            let claude = ClaudeCodeAgent::new().with_model_str("opus-4");
            ContentSynthesizerAgent::new(claude)
        },
        "development" => {
            let claude = ClaudeCodeAgent::new().with_model_str("sonnet-4.5");
            ContentSynthesizerAgent::new(claude)
        },
        _ => ContentSynthesizerAgent::default()
    }
}

// Example 2: Switching between different LLM providers
fn create_agent_with_provider(provider: &str) -> ContentSynthesizerAgent {
    match provider {
        "claude" => {
            let inner = ClaudeCodeAgent::new().with_model_str("sonnet-4.5");
            ContentSynthesizerAgent::new(inner)
        },
        "gemini" => {
            let inner = GeminiAgent::new().with_model_str("gemini-2.0-flash");
            ContentSynthesizerAgent::new(inner)
        },
        _ => ContentSynthesizerAgent::default()
    }
}

// Example 3: Custom configuration injection
fn create_configured_agent() -> ContentSynthesizerAgent {
    let claude = ClaudeCodeAgent::new()
        .with_model_str("opus-4")
        .with_system_prompt("You are an expert technical writer focused on clarity and accuracy.");
    ContentSynthesizerAgent::new(claude)
}
```

**Features:**
- ✅ Agent injection support (great for testing with mocks)
- ✅ Reuses internal agent (efficient)
- ✅ Static dispatch (compile-time optimization)
- ✅ Multiple constructor patterns
- ✅ Suitable for production use

**Testing Example:**

Agent injection makes testing simple and deterministic:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use llm_toolkit::agent::{Agent, AgentError, Payload};

    // Define a mock agent for testing
    struct MockAgent {
        response: String,
        call_count: std::cell::RefCell<usize>,
    }

    #[async_trait::async_trait]
    impl Agent for MockAgent {
        type Output = String;
        type Expertise = &'static str;
        fn expertise(&self) -> &&'static str {
            const EXPERTISE: &str = "mock";
            &EXPERTISE
        }
        async fn execute(&self, _: Payload) -> Result<String, AgentError> {
            *self.call_count.borrow_mut() += 1;
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_with_mock() {
        // Inject deterministic mock for testing
        let mock = MockAgent {
            response: r#"{"title": "Test Article", "body": "Test content", "references": ["source1"]}"#.to_string(),
            call_count: std::cell::RefCell::new(0),
        };
        let agent = ContentSynthesizerAgent::new(mock);

        // Execute and verify
        let result = agent.execute("test".to_string().into()).await.unwrap();
        assert_eq!(result.title, "Test Article");
        assert_eq!(result.references.len(), 1);
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Mock that returns an error
        struct ErrorAgent;

        #[async_trait::async_trait]
        impl Agent for ErrorAgent {
            type Output = String;
            type Expertise = &'static str;
            fn expertise(&self) -> &&'static str {
                const EXPERTISE: &str = "error mock";
                &EXPERTISE
            }
            async fn execute(&self, _: Payload) -> Result<String, AgentError> {
                Err(AgentError::ExecutionError("Simulated failure".to_string()))
            }
        }

        let agent = ContentSynthesizerAgent::new(ErrorAgent);
        let result = agent.execute("test".to_string().into()).await;
        assert!(result.is_err());
    }
}
```

**Using Custom Agent Backends:**

You can specify custom agent implementations (like Olama, local models, etc.) using `default_inner`:

```rust
// Define your custom agent
#[derive(Default, Clone)]
struct OlamaAgent {
    model: String,
}

impl OlamaAgent {
    fn new() -> Self { /* ... */ }
    fn with_model(self, model: &str) -> Self { /* ... */ }
}

use llm_toolkit::agent::Payload;

#[async_trait::async_trait]
impl Agent for OlamaAgent {
    type Output = String;
    type Expertise = &'static str;
    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Olama agent";
        &EXPERTISE
    }
    async fn execute(&self, intent: Payload) -> Result<String, AgentError> {
        // Call Olama API
    }
}

// Create specialized agents using OlamaAgent as backend
#[llm_toolkit_macros::agent(
    expertise = "Writing technical articles",
    output = "ArticleDraft",
    default_inner = "OlamaAgent"  // Custom backend!
)]
struct ArticleWriterAgent;

#[llm_toolkit_macros::agent(
    expertise = "Reviewing Rust code",
    output = "CodeReview",
    default_inner = "OlamaAgent"  // Same backend, different expertise!
)]
struct CodeReviewerAgent;

// Usage:
let olama = OlamaAgent::new().with_model("llama3.1");
let writer = ArticleWriterAgent::new(olama.clone());
let reviewer = CodeReviewerAgent::new(olama);
```

This pattern lets you:
- ✅ Reuse one backend (Olama, etc.) for multiple specialized agents
- ✅ Each agent has unique expertise
- ✅ Share configuration or customize per-agent
- ✅ Easy testing with mock backends

**Using Structured Expertise with `llm-toolkit-expertise`:**

Beyond simple string descriptions, you can use the `llm-toolkit-expertise` crate for composition-based, structured expertise definitions. The `expertise` parameter accepts any expression that implements `ToPrompt`:

```rust
use llm_toolkit_expertise::{Expertise, WeightedFragment, KnowledgeFragment, Priority};

// Define reusable expertise
fn rust_reviewer_expertise() -> Expertise {
    Expertise::new("rust-reviewer", "1.0")
        .with_tag("lang:rust")
        .with_tag("role:reviewer")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Always run cargo check before reviewing".to_string()
            ))
            .with_priority(Priority::Critical)
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Check for security issues".to_string(),
                steps: vec![
                    "Scan for unsafe code".to_string(),
                    "Verify input validation".to_string(),
                ],
            })
            .with_priority(Priority::High)
        )
}

// Use structured expertise in agent macro
#[llm_toolkit_macros::agent(
    expertise = rust_reviewer_expertise(),
    output = "ReviewResult"
)]
struct RustCodeReviewerAgent;
```

**Note on `Expertise::description` (Optional Field):**

When creating `Expertise` instances, the `description` field is optional. Use `Expertise::new(id, version)` for the basic two-argument constructor, and add an explicit description with `with_description()` if needed:

```rust
// Without explicit description - auto-generates from first fragment (~100 chars)
let expertise = Expertise::new("rust-reviewer", "1.0")
    .with_tag("lang:rust")
    .with_fragment(/* ... */);

// With explicit description
let expertise = Expertise::new("rust-reviewer", "1.0")
    .with_description("Expert Rust code reviewer with security focus")
    .with_tag("lang:rust")
    .with_fragment(/* ... */);
```

The structured approach enables:
- ✅ **Composition over inheritance**: Build expertise from reusable fragments
- ✅ **Priority-based prompts**: Control emphasis with Critical/High/Normal/Low weights
- ✅ **Context-aware behavior**: Activate fragments based on task health and conditions
- ✅ **Version control**: Track expertise evolution with semantic versioning

See the [llm-toolkit-expertise documentation](https://docs.rs/llm-toolkit-expertise) for details.

**Customizing Default Initialization with `init`:**

When using `default_inner`, you can specify an `init` function to customize the default initialization:

```rust
// Define an init function that configures the agent
fn init_with_workspace(agent: ClaudeCodeAgent) -> ClaudeCodeAgent {
    agent
        .with_cwd("/workspace/project")
        .with_env("PATH", enhanced_path())
}

// Agent with custom default initialization
#[llm_toolkit_macros::agent(
    expertise = "Generate project documentation",
    output = "Documentation",
    default_inner = "ClaudeCodeAgent",
    init = "init_with_workspace"  // Applied during Default::default()
)]
struct DocGeneratorAgent;

// Usage:
let agent = DocGeneratorAgent::default();  // init_with_workspace is applied automatically
// Or with custom inner agent:
let custom_agent = ClaudeCodeAgent::new().with_model("claude-opus-4-5");
let agent = DocGeneratorAgent::new(custom_agent);  // init is NOT applied
```

The `init` function:
- ✅ **Signature**: Must be `Fn(InnerAgent) -> InnerAgent`
- ✅ **Applied automatically**: Only when using `::default()`
- ✅ **Not applied**: When using `::new(custom_agent)`
- ✅ **Use cases**: Environment-based config, workspace setup, default model selection

**Proxy Builder Methods with `proxy_methods`:**

When using `default_inner`, you can also proxy the inner agent's builder methods to the outer agent:

```rust
// Agent with proxied builder methods
#[llm_toolkit_macros::agent(
    expertise = "Code analysis agent",
    output = "AnalysisResult",
    default_inner = "ClaudeCodeAgent",
    proxy_methods = ["with_cwd", "with_env", "with_model_str"]  // Proxy specific methods
)]
struct CodeAnalyzerAgent;

// Usage - builder pattern on the outer agent!
let agent = CodeAnalyzerAgent::default()
    .with_cwd("/project/src")
    .with_env("RUST_LOG", "debug")
    .with_model_str("claude-opus-4");
```

Available methods to proxy:
- ✅ `with_cwd` / `with_directory` / `with_attachment_dir` - Set working directory
- ✅ `with_env` / `with_envs` - Set environment variables
- ✅ `with_arg` / `with_args` - Set CLI arguments
- ✅ `with_model_str` - Set model
- ✅ `with_execution_profile` - Set execution profile
- ✅ `with_keep_attachments` - Control attachment retention

The `proxy_methods` feature:
- ✅ **Array of method names**: Choose only the methods you need
- ✅ **Type-safe**: Generated with correct signatures
- ✅ **Builder pattern**: Chainable and consistent with inner agent API
- ✅ **Combines with `init`**: Use both for maximum flexibility

**Combining `init` and `proxy_methods`:**

```rust
// Init function for defaults
fn init_analyzer(agent: ClaudeCodeAgent) -> ClaudeCodeAgent {
    agent.with_execution_profile(ExecutionProfile::Balanced)
}

// Agent with both init and proxy_methods
#[llm_toolkit_macros::agent(
    expertise = "Advanced code analyzer",
    output = "AnalysisResult",
    default_inner = "ClaudeCodeAgent",
    init = "init_analyzer",                              // Default config
    proxy_methods = ["with_cwd", "with_env"]             // Runtime config
)]
struct AdvancedAnalyzerAgent;

// Usage:
let agent = AdvancedAnalyzerAgent::default()  // init_analyzer applied
    .with_cwd("/custom/path")                 // Runtime override via proxy
    .with_env("DEBUG", "1");                  // Runtime override via proxy
```

**When to use which:**
- **`#[agent(...)]` with `backend`**: ✅ **RECOMMENDED** - Production with Claude/Gemini
- **`#[agent(...)]` with `default_inner`**: Custom backends (Ollama, local models, mocks)
- **`#[derive(Agent)]`**: ⚠️ **DEPRECATED** - Do not use (expertise not injected)

