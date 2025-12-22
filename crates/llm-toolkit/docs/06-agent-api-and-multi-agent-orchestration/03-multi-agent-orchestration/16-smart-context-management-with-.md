#### Smart Context Management with `ToPrompt`


The orchestrator automatically manages context between agent steps. When an agent produces output, the orchestrator stores it and makes it available to subsequent steps. If the output type implements `ToPrompt`, the orchestrator intelligently uses the human-readable prompt representation instead of raw JSON.

**Why This Matters:**

When you have complex output types (like enums with variant descriptions, or structs with rich formatting), you want the orchestrator to pass them to the next agent in a readable, LLM-friendly format—not as opaque JSON.

**Example: Enum with ToPrompt**

```rust
use llm_toolkit::{ToPrompt, Agent};
use serde::{Serialize, Deserialize};

// Define an enum with rich documentation
#[derive(ToPrompt, Serialize, Deserialize)]
pub enum AnalysisResult {
    /// The topic is technically sound and ready to proceed
    Approved,
    /// The topic needs revision due to technical inaccuracies
    NeedsRevision { reasons: Vec<String> },
    /// The topic is rejected as out of scope
    Rejected,
}

// Agent that produces this enum
#[derive(Agent)]
#[agent(
    expertise = "Analyze technical topics for accuracy and scope",
    output = "AnalysisResult"
)]
struct AnalyzerAgent;
```

**How it works:**

1. **Step 1**: `AnalyzerAgent` produces `AnalysisResult::NeedsRevision { reasons: [...] }`
2. **Orchestrator stores two versions**:
   - `step_1_output`: JSON representation `{"NeedsRevision": {"reasons": [...]}}`
   - `step_1_output_prompt`: ToPrompt representation with full descriptions
3. **Step 2**: When building intent for the next agent, the orchestrator prefers the `_prompt` version
4. **Result**: Next agent receives rich, human-readable context instead of cryptic JSON

**Setup:**

To enable `ToPrompt` support for your agent outputs, use `add_agent_with_to_prompt`:

```rust
// ✅ Correct: Use add_agent_with_to_prompt for types implementing ToPrompt
orchestrator.add_agent_with_to_prompt(MyAnalyzerAgent::new());

// ❌ Common Mistake: Using add_agent() - ToPrompt won't be used!
// orchestrator.add_agent(MyAnalyzerAgent::new());
```

**Benefits:**

- **Better LLM Understanding**: Complex types are presented in natural language, not JSON
- **Automatic Fallback**: If `ToPrompt` is not implemented, JSON is used (backward compatible)
- **Type-Safe**: The conversion is compile-time verified through the type system
- **Zero Overhead**: Only computed once per step and cached in context

