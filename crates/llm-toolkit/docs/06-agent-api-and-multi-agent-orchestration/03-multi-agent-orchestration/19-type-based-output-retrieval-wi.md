#### Type-Based Output Retrieval with `TypeMarker` (v0.13.9+)


**Problem**: The orchestrator's strategy LLM generates non-deterministic step IDs (`step_1`, `world_generation`, `analysis_phase`, etc.), making it difficult to retrieve specific outputs by step ID. You want to retrieve outputs by *type*, not by guessing step names.

**Solution**: Use the `TypeMarker` pattern to retrieve outputs based on their type, regardless of which step produced them.

**How It Works:**

There are two ways to add `__type` field for type-based retrieval:

**Method 1: Using `#[type_marker]` attribute macro (Recommended)**

```rust
use llm_toolkit::{type_marker, ToPrompt};
use serde::{Deserialize, Serialize};

// IMPORTANT: #[type_marker] must be placed BEFORE #[derive(...)]
#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct HighConceptResponse {
    pub reasoning: String,
    pub high_concept: String,
}
```

The `#[type_marker]` attribute macro automatically:
- Adds `__type: String` field with `#[serde(default = "default_high_concept_response_type")]`
- Generates the default function that returns the struct name
- Implements the `TypeMarker` trait
- The `__type` field is **automatically excluded from LLM schema** (ToPrompt skips fields named `__type`)

**Method 2: Manual `__type` field definition (For custom configurations)**

Use this method when you need special configurations:
- Custom field name or type
- Complex default function logic
- Integration with existing code

```rust
use llm_toolkit::{TypeMarker, ToPrompt};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt, TypeMarker)]
#[prompt(mode = "full", type_marker)]  // 👈 Optional marker to document TypeMarker usage
pub struct HighConceptResponse {
    #[serde(default = "default_high_concept_type")]
    __type: String,  // Manually defined for custom configuration
    pub reasoning: String,
    pub high_concept: String,
}

fn default_high_concept_type() -> String {
    "HighConceptResponse".to_string()
}
```

**Note:** The `#[prompt(type_marker)]` parameter is optional and serves as documentation/marker. The `__type` field will be automatically excluded from LLM schema regardless.

**Complete Example:**

```rust
use llm_toolkit::{type_marker, ToPrompt, Agent};
use serde::{Deserialize, Serialize};

// Define your response types
#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct HighConceptResponse {
    pub reasoning: String,
    pub high_concept: String,
}

#[type_marker]
#[derive(Serialize, Deserialize, Debug, Clone, ToPrompt)]
#[prompt(mode = "full")]
pub struct ProfileResponse {
    pub name: String,
    pub role: String,
}

// Define agents
#[derive(Agent)]
#[agent(
    expertise = "Generate high-level concepts",
    output = "HighConceptResponse"
)]
struct ConceptAgent;

#[derive(Agent)]
#[agent(
    expertise = "Create character profiles",
    output = "ProfileResponse"
)]
struct ProfileAgent;

// Register agents and execute
orchestrator.add_agent_with_to_prompt(ConceptAgent::default());
orchestrator.add_agent_with_to_prompt(ProfileAgent::default());

let result = orchestrator.execute(&intent).await?;

// Retrieve outputs by type - no need to know step IDs!
let concept: HighConceptResponse = orchestrator.get_typed_output()?;
let profile: ProfileResponse = orchestrator.get_typed_output()?;

println!("Concept: {}", concept.high_concept);
println!("Profile: {} - {}", profile.name, profile.role);
```

**Key Points:**

- **`#[type_marker]`**: Attribute macro that automatically adds `__type` field and implements `TypeMarker`
  - ⚠️ **Must be placed FIRST** (before `#[derive(...)]`) due to Rust macro processing order
  - Generates: field, default function, and trait implementation
  - The `__type` field is **excluded from the JSON schema** sent to LLMs (prevents confusion)
- **`#[derive(TypeMarker)]`**: Only implements the trait (use with manual `__type` field)
- **`get_typed_output<T>()`**: Type-safe retrieval that returns `Result<T, OrchestratorError>`
- **Schema exclusion**: ToPrompt automatically skips fields named `__type` (Line 154 in macro implementation)

**Benefits:**

- ✅ **No Step ID Guessing**: Retrieve outputs by type, not by unpredictable step names
- ✅ **Type-Safe**: Compile-time verification of output types
- ✅ **Clean Schema**: `__type` is excluded from schema to prevent LLM confusion
- ✅ **Automatic Deserialization**: `__type` is added during JSON parsing via `#[serde(default)]`
- ✅ **DRY Principle**: No manual field definition or JSON schema duplication needed
- ✅ **Works with Dynamic Workflows**: Strategy LLM can name steps anything; your code still works

**Common Pattern:**

```rust
// 1. Execute orchestrated workflow
let result = orchestrator.execute(&intent).await?;

// 2. Retrieve all needed outputs by type
let world_concept: WorldConceptResponse = orchestrator.get_typed_output()?;
let high_concept: HighConceptResponse = orchestrator.get_typed_output()?;
let emblem: EmblemResponse = orchestrator.get_typed_output()?;
let profile: ProfileResponse = orchestrator.get_typed_output()?;

// 3. Assemble final result
let spirit = Spirit {
    world_concept: world_concept.into(),
    high_concept: high_concept.high_concept,
    emblems: vec![emblem.obvious_emblem, emblem.creative_emblem],
    profile: profile.into(),
};
```

**Comparison with Step-Based Retrieval:**

```rust
// ❌ Step-based retrieval (fragile)
let concept_json = orchestrator.get_step_output("step_1")?; // What if it's "concept_generation"?
let concept: HighConceptResponse = serde_json::from_value(concept_json.clone())?;

// ✅ Type-based retrieval (robust)
let concept: HighConceptResponse = orchestrator.get_typed_output()?; // Always works!
```

**Run the examples:**
```bash
# See TypeMarker schema generation in action
cargo run --example type_marker_schema_test --features agent,derive

# Full orchestrator example
cargo run --example orchestrator_basic --features agent,derive
```

