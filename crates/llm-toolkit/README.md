# llm-toolkit
Basic llm tools for rust

# Motivation & Philosophy

High-level LLM frameworks like LangChain, while powerful, can be problematic in Rust. Their heavy abstractions and complex type systems often conflict with Rust's strengths, imposing significant constraints and learning curves on developers.

There is a clear need for a different kind of tool: a low-level, unopinionated, and minimalist toolkit that provides robust "last mile" utilities for LLM integration, much like how `candle` provides core building blocks for ML without dictating the entire application architecture.

This document proposes the creation of `llm-toolkit`, a new library crate designed to be the professional's choice for building reliable, high-performance LLM-powered applications in Rust.

## Core Design Principles

1.  **Minimalist & Unopinionated:**
    The toolkit will NOT impose any specific application architecture. Developers are free to design their own `UseCase`s and `Service`s. `llm-toolkit` simply provides a set of sharp, reliable "tools" to be called when needed.

2.  **Focused on the "Last Mile Problem":**
    The toolkit focuses on solving the most common and frustrating problems that occur at the boundary between a strongly-typed Rust application and the unstructured, often unpredictable string-based responses from LLM APIs.

3.  **Minimal Dependencies:**
    The toolkit will have minimal dependencies (primarily `serde` and `minijinja`) to ensure it can be added to any Rust project with negligible overhead and maximum compatibility.

## Features

| Feature Area | Description | Key Components | Status |
|---|---|---|---|
| **Content Extraction** | Safely extracting structured data (like JSON) from unstructured LLM responses. | `extract` module (`FlexibleExtractor`, `extract_json`) | Implemented |
| **Prompt Generation** | Building complex prompts from Rust data structures with a powerful templating engine. | `prompt!` macro, `#[derive(ToPrompt)]`, `#[derive(ToPromptSet)]` | Implemented |
| **Multi-Target Prompts** | Generate multiple prompt formats from a single data structure for different contexts. | `ToPromptSet` trait, `#[prompt_for(...)]` attributes | Implemented |
| **Context-Aware Prompts** | Generate prompts for a type within the context of another (e.g., a `Tool` for an `Agent`). | `ToPromptFor<T>` trait, `#[derive(ToPromptFor)]` | Implemented |
| **Example Aggregation** | Combine examples from multiple data structures into a single formatted section. | `examples_section!` macro | Implemented |
| **External Prompt Templates** | Load prompt templates from external files to separate prompts from Rust code. | `#[prompt(template_file = "...")]` attribute | Implemented |
| **Type-Safe Intent Definition** | Generate prompt builders and extractors from a single enum definition. | `#[define_intent]` macro | Implemented |
| **Intent Extraction** | Extracting structured intents (e.g., enums) from LLM responses. | `intent` module (`IntentFrame`, `IntentExtractor`) | Implemented |
| **Agent API** | Define reusable AI agents with expertise and structured outputs. | `Agent` trait, `#[agent(...)]` macro (recommended), `#[derive(Agent)]` (deprecated) | Implemented |
| **Agent Description & Capabilities** | Lightweight agent metadata for orchestrator routing with auto-generated descriptions and explicit capability declarations. | `description` attribute, `capabilities` attribute, `Expertise::auto_description_from_text()` | Implemented (v0.57.0) |
| **Auto-JSON Enforcement** | Automatically add JSON schema instructions to agent prompts for better LLM compliance. | `#[agent(...)]` with `ToPrompt::prompt_schema()` integration | Implemented |
| **Built-in Retry** | Intelligent retry with 3-priority delay system: server retry_after (Priority 1), 429 exponential backoff (Priority 2), linear backoff (Priority 3). Includes RetryAgent decorator and Full Jitter. | `max_retries` attribute, `RetryAgent`, `retry_after` field | Implemented |
| **Multi-Modal Payload** | Pass text and images to agents and dialogues through a unified `Payload` interface with backward compatibility. | `Payload`, `PayloadContent` types, `impl Into<Payload>` | Implemented |
| **Dynamic Payload Instructions** | Prepend turn-specific instructions or constraints to payloads without modifying Persona definitions. | `prepend_message()`, `prepend_system()` | Implemented |
| **Persistent Context Management** | Attach context information that remains visible throughout long conversations without being buried in history. PersonaAgent strategically places context based on conversation length, with configurable strategies for Participants placement and trailing prompts to reinforce persona identity. | `PayloadContent::Context`, `with_context()`, `ContextConfig` (`participants_after_context`, `include_trailing_prompt`), `.with_context_config()` | Implemented |
| **Multi-Agent Orchestration** | Coordinate multiple agents to execute complex workflows with adaptive error recovery. | `Orchestrator`, `BlueprintWorkflow`, `StrategyMap` | Implemented |
| **Context-Aware Detection** | Automatically infer task health, task type, and user states from execution patterns using layered detection (rule-based + LLM-based). Orchestrator automatically enriches agent payloads with detected context. | `DetectedContext`, `RuleBasedDetector`, `AgentBasedDetector`, `DetectionMode` | Implemented |
| **Execution Profiles** | Declaratively configure agent behavior (Creative/Balanced/Deterministic) via semantic profiles. | `ExecutionProfile` enum, `profile` attribute, `.with_execution_profile()` | Implemented (v0.13.0) |
| **Template File Validation** | Compile-time validation of template file paths with helpful error messages. | `template_file` attribute validation | Implemented (v0.13.0) |
| **Resilient Deserialization** | Deserializing LLM responses into Rust types, handling schema variations. | (Planned) | Planned |

## Prompt Generation

`llm-toolkit` offers three powerful and convenient ways to generate prompts, powered by the `minijinja` templating engine.

### 1. Ad-hoc Prompts with `prompt!` macro

For quick prototyping and flexible prompt creation, the `prompt!` macro provides a `println!`-like experience. You can pass any `serde::Serialize`-able data as context.

```rust
use llm_toolkit::prompt::prompt;
use serde::Serialize;

#[derive(Serialize)]
struct User {
    name: &'static str,
    role: &'static str,
}

let user = User { name: "Mai", role: "UX Engineer" };
let task = "designing a new macro";

let p = prompt!(
    "User {{user.name}} ({{user.role}}) is currently {{task}}.",
    user = user,
    task = task
).unwrap();

assert_eq!(p, "User Mai (UX Engineer) is currently designing a new macro.");
```

### 2. Structured Prompts with `#[derive(ToPrompt)]`

For core application logic, you can derive the `ToPrompt` trait on your structs to generate prompts in a type-safe way.

**Setup:**

First, enable the `derive` feature in your `Cargo.toml`:
```toml
[dependencies]
llm-toolkit = { version = "0.1.0", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
```

**Usage:**

Then, use the `#[derive(ToPrompt)]` and `#[prompt(...)]` attributes on your struct. The struct must also derive `serde::Serialize`.

```rust
use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(ToPrompt, Serialize)]
#[prompt(template = "USER PROFILE:\nName: {{name}}\nRole: {{role}}")]
struct UserProfile {
    name: &'static str,
    role: &'static str,
}

let user = UserProfile {
    name: "Yui",
    role: "World-Class Pro Engineer",
};

let p = user.to_prompt();
// The following would be printed:
// USER PROFILE:
// Name: Yui
// Role: World-Class Pro Engineer
```

#### Default Formatting and Field Attributes

If you omit the `#[prompt(template = "...")]` attribute on a struct, `ToPrompt` will automatically generate a key-value representation of the struct's fields. You can control this output with field-level attributes:

| Attribute | Description |
| :--- | :--- |
| `#[prompt(rename = "new_name")]` | Overrides the key with `"new_name"`. |
| `#[prompt(skip)]` | Excludes the field from the output. |
| `#[prompt(format_with = "path::to::func")]`| Uses a custom function to format the field's **value**. |

The **key** for each field is determined with the following priority:
1.  `#[prompt(rename = "...")]` attribute.
2.  Doc comment (`/// ...`) on the field.
3.  The field's name (fallback).

**Comprehensive Example:**

```rust
use llm_toolkit::ToPrompt;
use llm_toolkit_macros::ToPrompt; // Make sure to import the derive macro
use serde::Serialize;

// A custom formatting function
fn format_id(id: &u64) -> String {
    format!("user-{}", id)
}

#[derive(ToPrompt, Serialize)]
struct AdvancedUser {
    /// The user's unique identifier
    id: u64,

    #[prompt(rename = "full_name")]
    name: String,

    // This field will not be included in the prompt
    #[prompt(skip)]
    internal_hash: String,

    // This field will use a custom formatting function for its value
    #[prompt(format_with = "format_id")]
    formatted_id: u64,
}

let user = AdvancedUser {
    id: 123,
    name: "Mai".to_string(),
    internal_hash: "abcdef".to_string(),
    formatted_id: 123,
};

let p = user.to_prompt();
// The following would be generated:
// The user's unique identifier: 123
// full_name: Mai
// formatted_id: user-123
```

#### Tip: Handling Special Characters in Templates

When using raw string literals (`r#"..."#`) for your templates, be aware of a potential parsing issue if your template content includes the `#` character (e.g., in a hex color code like `"#FFFFFF"`).

The macro parser can sometimes get confused by the inner `#`. To avoid this, you can use a different number of `#` symbols for the raw string delimiter.

**Problematic Example:**
```rust
// This might fail to parse correctly
#[prompt(template = r#"{"color": "#FFFFFF"}"#)]
struct Color { /* ... */ }
```

**Solution:**
```rust
// Use r##"..."## to avoid ambiguity
#[prompt(template = r##"{"color": "#FFFFFF"}"##)]
struct Color { /* ... */ }
```

#### Using External Template Files

For larger prompts, you can separate them into external files (`.jinja`, `.txt`, etc.) and reference them using the `template_file` attribute. This improves code readability and makes prompts easier to manage.

You can also enable compile-time validation of your templates with `validate = true`.

```rust
use llm_toolkit::ToPrompt;
use serde::Serialize;

// In templates/user_profile.jinja:
// Name: {{ name }}
// Email: {{ email }}

#[derive(ToPrompt, Serialize)]
#[prompt(
    template_file = "templates/user_profile.jinja",
    validate = true
)]
struct UserFromTemplate {
    name: String,
    email: String,
}

let user = UserFromTemplate {
    name: "Yui".to_string(),
    email: "yui@example.com".to_string(),
};

let p = user.to_prompt();
// The following would be generated from the file:
// Name: Yui
// Email: yui@example.com
```

### 3. Enum Documentation with `#[derive(ToPrompt)]`

For enums, the `ToPrompt` derive macro provides flexible ways to generate prompts. It distinguishes between **instance-level** prompts (describing a single variant) and **type-level** schema (describing all possible variants).

#### Instance vs. Type-Level Prompts

```rust
use llm_toolkit::ToPrompt;

/// Represents different user intents for a chatbot
#[derive(ToPrompt)]
pub enum UserIntent {
    /// User wants to greet or say hello
    Greeting,
    /// User is asking for help or assistance
    Help,
}

// Instance-level: describe the current variant only
let intent = UserIntent::Greeting;
let prompt = intent.to_prompt();
// Output: "Greeting: User wants to greet or say hello"

// Type-level: describe all possible variants (TypeScript union type format)
let schema = UserIntent::prompt_schema();
// Output:
// /**
//  * Represents different user intents for a chatbot
//  */
// type UserIntent =
//   | "Greeting"  // User wants to greet or say hello
//   | "Help"  // User is asking for help or assistance;
//
// Example value: "Greeting"
```

**When to use which:**
- **`value.to_prompt()`** - When you need to describe a specific enum value to the LLM (e.g., "The user selected: Greeting")
- **`Enum::prompt_schema()`** - When you need to explain all possible options to the LLM (e.g., "Choose one of these intents...")

**TypeScript Format Benefits:**
- Clear union type syntax that LLMs understand well
- Each variant includes its description as an inline comment
- Example value shows the correct JSON format
- JSDoc comments for type-level documentation

#### Advanced Attribute Controls

The `ToPrompt` derive macro supports powerful attribute-based controls for fine-tuning the generated prompts:

- **`#[prompt("...")]`** - Provide a custom description that overrides the doc comment
- **`#[prompt(skip)]`** - Exclude a variant from the schema (but the variant name is still shown at instance level)
- **No attribute** - Variants without doc comments or attributes will show just the variant name

Here's a comprehensive example showcasing all features:

```rust
use llm_toolkit::ToPrompt;

/// Represents different actions a user can take in the system
#[derive(ToPrompt)]
pub enum UserAction {
    /// User wants to create a new document
    CreateDocument,

    /// User is searching for existing content
    Search,

    #[prompt("Custom: User is updating their profile settings and preferences")]
    UpdateProfile,

    #[prompt(skip)]
    InternalDebugAction,

    DeleteItem,
}

// Instance-level prompts
let action1 = UserAction::CreateDocument;
assert_eq!(action1.to_prompt(), "CreateDocument: User wants to create a new document");

let action2 = UserAction::InternalDebugAction;
assert_eq!(action2.to_prompt(), "InternalDebugAction");  // Skipped variants show name only

// Type-level schema (TypeScript union type format)
let schema = UserAction::prompt_schema();
// Output:
// /**
//  * Represents different actions a user can take in the system
//  */
// type UserAction =
//   | "CreateDocument"  // User wants to create a new document
//   | "Search"  // User is searching for existing content
//   | "UpdateProfile"  // Custom: User is updating their profile settings and preferences
//   | "DeleteItem";
//
// Example value: "CreateDocument"
//
// Note: InternalDebugAction is excluded from schema due to #[prompt(skip)]
```

**Behavior of `#[prompt(skip)]`:**
- At **instance level** (`value.to_prompt()`): Shows only the variant name
- At **type level** (`Enum::prompt_schema()`): Completely excluded from the schema

#### Variant Renaming with Priority System

When working with enums that need different serialization formats (e.g., snake_case for APIs, camelCase for JSON), the `ToPrompt` macro provides flexible variant renaming with a clear 4-level priority system:

**Priority Levels (Highest to Lowest):**

1. **`#[prompt(rename = "...")]`** - ToPrompt-specific, highest priority
2. **`#[serde(rename = "...")]`** - Per-variant serde rename
3. **`#[serde(rename_all = "...")]`** - Enum-level serde rename rule
4. **Default PascalCase** - Rust variant name as-is

This priority system ensures that the TypeScript schema matches serde's serialization format, preventing deserialization errors when LLMs follow the schema.

**Example: Basic serde rename_all Support**

```rust
use llm_toolkit::ToPrompt;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
pub enum VisualTreatment {
    DelicateLuminous,
    CinematicCrisp,
    SoftAtmospheric,
}

// Type-level schema matches serde format
let schema = VisualTreatment::prompt_schema();
// Output:
// type VisualTreatment =
//   | "delicate_luminous"
//   | "cinematic_crisp"
//   | "soft_atmospheric";
//
// Example value: "delicate_luminous"

// Instance-level also uses renamed values
let visual = VisualTreatment::CinematicCrisp;
assert_eq!(visual.to_prompt(), "cinematic_crisp");

// Serialization matches
let json = serde_json::to_string(&visual).unwrap();
assert_eq!(json, "\"cinematic_crisp\"");  // ✅ Perfect match!
```

**Example: Priority System in Action**

```rust
use llm_toolkit::ToPrompt;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]  // Priority 3: enum-level rule
pub enum UserAction {
    // Priority 1: #[prompt(rename)] wins over everything
    #[prompt(rename = "ui_create")]
    CreateDocument,

    // Priority 2: #[serde(rename)] overrides rename_all
    #[serde(rename = "find_content")]
    SearchFiles,

    // Priority 3: Uses snake_case from rename_all
    UpdateProfile,

    // Priority 4: No rename rules, uses default PascalCase
    DeleteItem,
}

let schema = UserAction::prompt_schema();
// Output:
// type UserAction =
//   | "ui_create"        // Priority 1: prompt rename
//   | "find_content"     // Priority 2: serde rename
//   | "update_profile"   // Priority 3: rename_all
//   | "DeleteItem";      // Priority 4: default
```

**Example: Combined with Descriptions**

```rust
#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    #[prompt(rename = "search_query")]
    #[prompt(description = "User wants to search for content")]
    Search,

    #[serde(rename = "create_new")]
    #[prompt(description = "User wants to create a new item")]
    Create,
}

// Both rename and description are applied!
let schema = Intent::prompt_schema();
// Output:
// type Intent =
//   | "search_query"  // User wants to search for content
//   | "create_new"    // User wants to create a new item
```

**Supported Rename Rules (from serde):**

All serde rename_all patterns are supported:
- `lowercase` - `lowercase`
- `UPPERCASE` - `UPPERCASE`
- `PascalCase` - `PascalCase`
- `camelCase` - `camelCase`
- `snake_case` - `snake_case`
- `SCREAMING_SNAKE_CASE` - `SCREAMING_SNAKE_CASE`
- `kebab-case` - `kebab-case`
- `SCREAMING-KEBAB-CASE` - `SCREAMING-KEBAB-CASE`

**Why This Matters:**

Without matching serde's format, you get guaranteed deserialization failures:

```rust
// ❌ Without rename support (old behavior)
#[derive(Serialize, Deserialize, ToPrompt)]
#[serde(rename_all = "snake_case")]
pub enum Status { InProgress }

let schema = Status::prompt_schema();
// Schema says: "InProgress"
// But serde expects: "in_progress"
// LLM follows schema → returns "InProgress" → deserialization fails!

// ✅ With rename support (new behavior)
let schema = Status::prompt_schema();
// Schema says: "in_progress"
// Serde expects: "in_progress"
// LLM follows schema → returns "in_progress" → deserialization succeeds!
```

**Best Practices:**

1. **Always use `#[serde(rename_all)]` with ToPrompt** - Ensures schema matches serialization
2. **Use `#[prompt(rename)]` for custom display names** - When LLM-facing names differ from API serialization
3. **Test deserialization** - Verify LLM responses deserialize correctly with your schema

#### Struct Variants (Tagged Unions)

**New in v0.21.0+**: The `ToPrompt` macro now fully supports **struct variants**, enabling rich domain models with complex data. Struct variants are serialized as **TypeScript tagged unions** with a `type` discriminator field, which is the industry-standard pattern for LLMs.

**Basic Example:**

```rust
use llm_toolkit::ToPrompt;
use serde::{Serialize, Deserialize};

#[derive(ToPrompt, Serialize, Deserialize)]
#[serde(tag = "type")]  // ← serde tagged union
pub enum AnalysisResult {
    /// Analysis approved with no issues
    Approved,

    /// Analysis needs revision
    NeedsRevision {
        reasons: Vec<String>,
        severity: String,
    },

    /// Analysis rejected
    Rejected {
        reason: String,
    },
}

// Type-level schema (TypeScript tagged union)
let schema = AnalysisResult::prompt_schema();
// Output:
// type AnalysisResult =
//   | "Approved"  // Analysis approved with no issues
//   | { type: "NeedsRevision", reasons: string[], severity: string }  // Analysis needs revision
//   | { type: "Rejected", reason: string };  // Analysis rejected
//
// Example value: "Approved"

// Instance-level: struct variants show fields
let result = AnalysisResult::NeedsRevision {
    reasons: vec!["Missing data".to_string()],
    severity: "High".to_string(),
};

let prompt = result.to_prompt();
// Output: "NeedsRevision: Analysis needs revision { reasons: [\"Missing data\"], severity: \"High\" }"

// Serde serialization (matches schema!)
let json = serde_json::to_string(&result).unwrap();
// Output: {"type":"NeedsRevision","reasons":["Missing data"],"severity":"High"}

// LLM response → deserializes perfectly
let from_llm = r#"{"type":"Rejected","reason":"Invalid format"}"#;
let parsed: AnalysisResult = serde_json::from_str(from_llm).unwrap();
```

**Supported Variant Types:**

| Variant Type | Example | TypeScript Output | Status |
|--------------|---------|-------------------|--------|
| **Unit** | `Variant` | `"Variant"` | ✅ Full support |
| **Struct** | `Variant { x: i32 }` | `{ type: "Variant", x: number }` | ✅ Full support |
| **Tuple** | `Variant(i32, String)` | `[number, string]` | ✅ Full support |

**Type Mapping:**

The macro automatically maps Rust types to TypeScript equivalents:

```rust
#[derive(ToPrompt, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Measurement {
    Temperature {
        celsius: f32,          // → number
        location: String,      // → string
    },
    Count {
        items: i64,            // → number
        verified: bool,        // → boolean
    },
    Tags {
        labels: Vec<String>,   // → string[]
        metadata: Option<String>, // → string | null
    },
}

// Generated schema:
// type Measurement =
//   | { type: "Temperature", celsius: number, location: string }
//   | { type: "Count", items: number, verified: boolean }
//   | { type: "Tags", labels: string[], metadata: string | null };
```

**Complex Example: Cinematic Lighting**

```rust
use llm_toolkit::ToPrompt;
use serde::{Serialize, Deserialize};

#[derive(ToPrompt, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LightingTechnique {
    /// Chiaroscuro (dramatic high-contrast lighting)
    Chiaroscuro {
        contrast_level: ContrastLevel,
        light_source: LightSourceType,
        shadow_direction: ShadowDirection,
    },

    /// Rembrandt lighting (triangle of light on cheek)
    Rembrandt {
        triangle_side: Side,
        fill_ratio: f32,
    },

    /// Simple natural lighting
    Natural,
}

#[derive(ToPrompt, Serialize, Deserialize)]
pub enum ContrastLevel { Low, Medium, High }

#[derive(ToPrompt, Serialize, Deserialize)]
pub enum LightSourceType { Single, Multiple, Diffused }

#[derive(ToPrompt, Serialize, Deserialize)]
pub enum ShadowDirection { Left, Right, Top, Bottom }

#[derive(ToPrompt, Serialize, Deserialize)]
pub enum Side { Left, Right }

// Generated schema (snake_case from rename_all):
// type LightingTechnique =
//   | { type: "chiaroscuro", contrast_level: ContrastLevel, light_source: LightSourceType, shadow_direction: ShadowDirection }  // Chiaroscuro (dramatic high-contrast lighting)
//   | { type: "rembrandt", triangle_side: Side, fill_ratio: number }  // Rembrandt lighting (triangle of light on cheek)
//   | "natural";  // Simple natural lighting

// LLM can return:
// {"type":"chiaroscuro","contrast_level":"High","light_source":"Single","shadow_direction":"Left"}
```

**Why Tagged Unions?**

1. **LLM-Friendly**: Industry-standard pattern that LLMs understand intuitively
2. **Type Safety**: Compile-time guarantees for field names and types
3. **Serde Compatible**: Works seamlessly with `#[serde(tag = "type")]`
4. **Clear Discrimination**: The `type` field makes variant identification unambiguous
5. **JSON-First**: Natural JSON representation for API communication

**Combining Features:**

All ToPrompt features work with struct variants:

```rust
#[derive(ToPrompt, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Command {
    /// Execute a script
    #[prompt(rename = "run_script")]  // Custom name
    Execute { script: String },

    #[prompt(skip)]  // Hidden from schema
    InternalDebug { details: String },

    /// Simple shutdown
    Shutdown,
}

// Schema includes only non-skipped variants with custom names:
// type Command =
//   | { type: "run_script", script: string }  // Execute a script
//   | "Shutdown";  // Simple shutdown
```

**Tuple Variants:**

Tuple variants generate TypeScript tuple types with proper type mapping:

```rust
use llm_toolkit::ToPrompt;
use serde::{Serialize, Deserialize};

#[derive(ToPrompt, Serialize, Deserialize)]
#[serde(untagged)]  // ← serde untagged for tuple arrays
pub enum Coordinate {
    /// 2D coordinate
    Point2D(f64, f64),
    /// 3D coordinate with metadata
    Point3D(f64, f64, f64),
    /// Origin point
    Origin,
}

// Generated schema:
// type Coordinate =
//   | [number, number]  // 2D coordinate
//   | [number, number, number]  // 3D coordinate with metadata
//   | "Origin";  // Origin point

// Instance to_prompt():
let point = Coordinate::Point2D(10.5, 20.3);
let prompt = point.to_prompt();
// Output: "Point2D: 2D coordinate (10.5, 20.3)"

// Serde serialization (untagged = array):
let json = serde_json::to_string(&point).unwrap();
// Output: [10.5,20.3]

// LLM can return:
// [10.5, 20.3]  → deserializes to Point2D
// [1.0, 2.0, 3.0]  → deserializes to Point3D
```

**Mixed Types in Tuples:**

```rust
#[derive(ToPrompt, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// String-number pair
    Pair(String, i32),
    /// Single value
    Single(String),
    /// Complex tuple
    Triple(String, Vec<i32>, Option<bool>),
}

// Generated schema:
// type Value =
//   | [string, number]  // String-number pair
//   | [string]  // Single value
//   | [string, number[], boolean | null];  // Complex tuple
```

**Best Practices:**

1. **Struct variants**: Use `#[serde(tag = "type")]` for tagged unions
2. **Tuple variants**: Use `#[serde(untagged)]` for tuple arrays
3. **Keep field names simple** - LLMs work best with clear, descriptive names
4. **Document variants** - Doc comments become inline comments in TypeScript
5. **Test roundtrips** - Verify LLM responses deserialize correctly
6. **Mix freely** - Combine unit, struct, and tuple variants as needed

### 4. Multi-Target Prompts with `#[derive(ToPromptSet)]`

For applications that need to generate different prompt formats from the same data structure for various contexts (e.g., human-readable vs. machine-parsable, or different LLM models), the `ToPromptSet` derive macro enables powerful multi-target prompt generation.

#### Basic Multi-Target Setup

```rust
use llm_toolkit::ToPromptSet;
use serde::Serialize;

#[derive(ToPromptSet, Serialize)]
#[prompt_for(name = "Visual", template = "## {{title}}\n\n> {{description}}")]
struct Task {
    title: String,
    description: String,

    #[prompt_for(name = "Agent")]
    priority: u8,

    #[prompt_for(name = "Agent", rename = "internal_id")]
    id: u64,

    #[prompt_for(skip)]
    is_dirty: bool,
}

let task = Task {
    title: "Implement feature".to_string(),
    description: "Add new functionality".to_string(),
    priority: 1,
    id: 42,
    is_dirty: false,
};

// Generate visual-friendly prompt using template
let visual_prompt = task.to_prompt_for("Visual")?;
// Output: "## Implement feature\n\n> Add new functionality"

// Generate agent-friendly prompt with key-value format
let agent_prompt = task.to_prompt_for("Agent")?;
// Output: "title: Implement feature\ndescription: Add new functionality\npriority: 1\ninternal_id: 42"
```

#### Advanced Features

**Custom Formatting Functions:**
```rust
fn format_priority(priority: &u8) -> String {
    match priority {
        1 => "Low".to_string(),
        2 => "Medium".to_string(),
        3 => "High".to_string(),
        _ => "Unknown".to_string(),
    }
}

#[derive(ToPromptSet, Serialize)]
struct FormattedTask {
    title: String,

    #[prompt_for(name = "Human", format_with = "format_priority")]
    priority: u8,
}
```

**Multimodal Support:**
```rust
use llm_toolkit::prompt::{PromptPart, ToPrompt};

#[derive(ToPromptSet, Serialize)]
#[prompt_for(name = "Multimodal", template = "Analyzing image: {{caption}}")]
struct ImageTask {
    caption: String,

    #[prompt_for(name = "Multimodal", image)]
    image: ImageData,
}

// Generate multimodal prompt with both text and image
let parts = task.to_prompt_parts_for("Multimodal")?;
// Returns Vec<PromptPart> with both Image and Text parts
```

#### Target Configuration Options

| Attribute | Description | Example |
|-----------|-------------|---------|
| `#[prompt_for(name = "TargetName")]` | Include field in specific target | `#[prompt_for(name = "Debug")]` |
| `#[prompt_for(name = "Target", template = "...")]` | Use template for target (struct-level) | `#[prompt_for(name = "Visual", template = "{{title}}")]` |
| `#[prompt_for(name = "Target", rename = "new_name")]` | Rename field for specific target | `#[prompt_for(name = "API", rename = "task_id")]` |
| `#[prompt_for(name = "Target", format_with = "func")]` | Custom formatting function | `#[prompt_for(name = "Human", format_with = "format_date")]` |
| `#[prompt_for(name = "Target", image)]` | Mark field as image content | `#[prompt_for(name = "Vision", image)]` |
| `#[prompt_for(skip)]` | Exclude field from all targets | `#[prompt_for(skip)]` |

When to use `ToPromptSet` vs `ToPrompt`:
- **`ToPrompt`**: Single, consistent prompt format across your application
- **`ToPromptSet`**: Multiple prompt formats needed for different contexts (human vs. machine, different LLM models, etc.)

### 5. Context-Aware Prompts with `#[derive(ToPromptFor)]`

Sometimes, the way you want to represent a type in a prompt depends on the context. For example, a `Tool` might have a different prompt representation when being presented to an `Agent` versus a human user. The `ToPromptFor<T>` trait and its derive macro solve this problem.

It allows a struct to generate a prompt *for* a specific target type, using the target's data in its template.

**Usage:**

The struct using `ToPromptFor` must derive `Serialize` and `ToPrompt`. The target struct passed to it must also derive `Serialize`.

```rust
use llm_toolkit::{ToPrompt, ToPromptFor};
use serde::Serialize;

#[derive(Serialize)]
struct Agent {
    name: String,
    role: String,
}

#[derive(ToPrompt, ToPromptFor, Serialize, Default)]
#[prompt(mode = "full")] // Enables schema_only, example_only modes for ToPrompt
#[prompt_for(
    target = "Agent",
    template = r#"
Hello, {{ target.name }}. As a {{ target.role }}, you can use the following tool.

### Tool Schema
{self:schema_only}

### Tool Example
{self:example_only}

The tool's name is '{{ self.name }}'.
"#
)]
/// A tool that can be used by an agent.
struct Tool {
    /// The name of the tool.
    #[prompt(example = "file_writer")]
    name: String,
    /// A description of what the tool does.
    #[prompt(example = "Writes content to a file.")]
    description: String,
}

let agent = Agent {
    name: "Yui".to_string(),
    role: "Pro Engineer".to_string(),
};

let tool = Tool {
    name: "file_writer_tool".to_string(),
    ..Default::default()
};

let prompt = tool.to_prompt_for(&agent);
// Generates a detailed prompt using the agent's name and role,
// and the tool's own schema and example.
```

### 6. Aggregating Examples with `examples_section!`

When providing few-shot examples to an LLM, it's often useful to show examples of all the data structures it might need to generate. The `examples_section!` macro automates this by creating a clean, formatted Markdown block from a list of types.

**Usage:**

All types passed to the macro must derive `ToPrompt` and `Default`, and have `#[prompt(mode = "full")]` and `#[prompt(example = "...")]` attributes to provide meaningful examples.

```rust
use llm_toolkit::{examples_section, ToPrompt};
use serde::Serialize;

#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "full")]
/// Represents a user of the system.
struct User {
    /// A unique identifier for the user.
    #[prompt(example = "user-12345")]
    id: String,
    /// The user's full name.
    #[prompt(example = "Taro Yamada")]
    name: String,
}

#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "full")]
/// Defines a concept for image generation.
struct Concept {
    /// The main idea for the art.
    #[prompt(example = "a futuristic city at night")]
    prompt: String,
    /// The desired style.
    #[prompt(example = "anime")]
    style: String,
}

let examples = examples_section!(User, Concept);
// The macro generates the following Markdown string:
//
// ### Examples
//
// Here are examples of the data structures you should use.
//
// ---
// #### `User`
// {
//   "id": "user-12345",
//   "name": "Taro Yamada"
// }
// ---
// #### `Concept`
// {
//   "prompt": "a futuristic city at night",
//   "style": "anime"
// }
// ---
```

## Intent Extraction with `IntentFrame`

`llm-toolkit` provides a safe and robust way to extract structured intents (like enums) from an LLM's response. The core component for this is the `IntentFrame` struct.

It solves a common problem: ensuring the tag you use to frame a query in a prompt (`<query>...</query>`) and the tag you use to extract the response (`<intent>...</intent>`) are managed together, preventing typos and mismatches.

**Usage:**

`IntentFrame` is used for two things: wrapping your input and extracting the structured response.

```rust
use llm_toolkit::{IntentFrame, IntentExtractor, IntentError};
use std::str::FromStr;

// 1. Define your intent enum
#[derive(Debug, PartialEq)]
enum UserIntent {
    Search,
    GetWeather,
}

impl FromStr for UserIntent {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "search" => Ok(UserIntent::Search),
            "getweather" => Ok(UserIntent::GetWeather),
            _ => Err(()),
        }
    }
}

// 2. Create an IntentFrame
// The first tag is for wrapping input, the second is for extracting the response.
let frame = IntentFrame::new("user_query", "intent");

// 3. Wrap your input to create part of your prompt
let user_input = "what is the weather in Tokyo?";
let wrapped_input = frame.wrap(user_input);
// wrapped_input is now "<user_query>what is the weather in Tokyo?</user_query>"

// (Imagine sending a full prompt with wrapped_input to an LLM here)

// 4. Extract the intent from the LLM's response
let llm_response = "Okay, I will get the weather. <intent>GetWeather</intent>";
let intent: UserIntent = frame.extract_intent(llm_response).unwrap();

assert_eq!(intent, UserIntent::GetWeather);
```

## Type-Safe Intents with `define_intent!`

To achieve the highest level of type safety and developer experience, the `#[define_intent]` macro automates the entire process of creating and extracting intents.

It solves a critical problem: by defining the prompt, the intent `enum`, and the extraction logic in a single place, it becomes impossible for the prompt-building code and the response-parsing code to diverge.

**Usage:**

Simply annotate an enum with `#[define_intent]` and provide the prompt template and extractor tag in an `#[intent(...)]` attribute.

```rust
use llm_toolkit::{define_intent, IntentExtractor, IntentError};
use std::str::FromStr;

#[define_intent]
#[intent(
    prompt = r#"
Please classify the user's request. The available intents are:
{{ intents_doc }}

User request: <query>{{ user_request }}</query>
"#,
    extractor_tag = "intent"
)]
/// The user's primary intent.
pub enum UserIntent {
    /// The user wants to know the weather.
    GetWeather,
    /// The user wants to send a message.
    SendMessage,
}

// The macro automatically generates:
// 1. A function: `build_user_intent_prompt(user_request: &str) -> String`
// 2. A struct: `pub struct UserIntentExtractor;` which implements `IntentExtractor<UserIntent>`

// --- How to use the generated code ---

// 1. Build the prompt
let prompt = build_user_intent_prompt("what's the weather like in London?");
// The prompt will include the formatted documentation from the enum.

// 2. Use the generated extractor to parse the LLM's response
let llm_response = "Understood. The user wants to know the weather. <intent>GetWeather</intent>";
let extractor = UserIntentExtractor;
let intent = extractor.extract_intent(llm_response).unwrap();

assert_eq!(intent, UserIntent::GetWeather);
```

This macro provides:
- **Ultimate Type Safety:** The prompt and the parser are guaranteed to be in sync.
- **Improved DX:** Eliminates boilerplate code for prompt functions and extractors.
- **Single Source of Truth:** The `enum` becomes the single, reliable source for all intent-related logic.

### Multi-Tag Mode for Complex Action Extraction

For more complex scenarios where you need to extract multiple action tags from a single LLM response, the `define_intent!` macro supports a `multi_tag` mode. This is particularly useful for agent-like applications where the LLM might use multiple XML-style action tags in a single response.

**Setup:**

To use multi-tag mode, add both dependencies to your `Cargo.toml`:

```toml
[dependencies]
llm-toolkit = { version = "0.8.3", features = ["derive"] }
quick-xml = "0.38"  # Required for multi_tag mode
```

Then define your actions:

```rust
use llm_toolkit::define_intent;

#[define_intent(mode = "multi_tag")]
#[intent(
    prompt = r#"Based on the user request, generate a response using the following available actions.

**Available Actions:**
{{ actions_doc }}

**User Request:**
{{ user_request }}"#
)]
#[derive(Debug, Clone, PartialEq)]
pub enum ChatAction {
    /// Get the current weather
    #[action(tag = "GetWeather")]
    GetWeather,

    /// Show an image to the user
    #[action(tag = "ShowImage")]
    ShowImage {
        /// The URL of the image to display
        #[action(attribute)]
        href: String,
    },

    /// Send a message to someone
    #[action(tag = "SendMessage")]
    SendMessage {
        /// The recipient of the message
        #[action(attribute)]
        to: String,
        /// The content of the message
        #[action(inner_text)]
        content: String,
    },
}
```

**Action Tag Attributes:**
- `#[action(tag = "TagName")]` - Defines the XML tag name for this action
- `#[action(attribute)]` - Maps a field to an XML attribute (e.g., `<Tag field="value" />`)
- `#[action(inner_text)]` - Maps a field to the inner text content (e.g., `<Tag>field_value</Tag>`)

**Generated Functions:**
The macro generates:
1. `build_chat_action_prompt(user_request: &str) -> String` - Builds the prompt with action documentation
2. `ChatActionExtractor` struct with methods:
   - `extract_actions(&self, text: &str) -> Result<Vec<ChatAction>, IntentError>` - Extract all actions from response
   - `transform_actions<F>(&self, text: &str, transformer: F) -> String` - Transform action tags using a closure
   - `strip_actions(&self, text: &str) -> String` - Remove all action tags from text

**Usage Example:**

```rust
// 1. Build the prompt
let prompt = build_chat_action_prompt("What's the weather and show me a cat picture?");

// 2. Extract multiple actions from LLM response
let llm_response = r#"
Here's the weather: <GetWeather />
And here's a cat picture: <ShowImage href="https://cataas.com/cat" />
<SendMessage to="user">I've completed both requests!</SendMessage>
"#;

let extractor = ChatActionExtractor;
let actions = extractor.extract_actions(llm_response)?;
// Returns: [ChatAction::GetWeather, ChatAction::ShowImage { href: "https://cataas.com/cat" }, ...]

// 3. Transform action tags to human-readable descriptions
let transformed = extractor.transform_actions(llm_response, |action| match action {
    ChatAction::GetWeather => "[Checking weather...]".to_string(),
    ChatAction::ShowImage { href } => format!("[Displaying image from {}]", href),
    ChatAction::SendMessage { to, content } => format!("[Message to {}: {}]", to, content),
});
// Result: "Here's the weather: [Checking weather...]\nAnd here's a cat picture: [Displaying image from https://cataas.com/cat]\n[Message to user: I've completed both requests!]"

// 4. Strip all action tags for clean text output
let clean_text = extractor.strip_actions(llm_response);
// Result: "Here's the weather: \nAnd here's a cat picture: \n"
```

**When to Use Multi-Tag Mode:**
- **Agent Applications:** When building AI agents that perform multiple actions per response
- **Rich LLM Interactions:** When you need structured actions mixed with natural language
- **Action Processing Pipelines:** When you need to extract, transform, or clean action-based responses

##### 3. Stateful Agents with Personas

For creating stateful, character-driven agents that maintain conversational history, `llm-toolkit` provides the `PersonaAgent` decorator and a convenient `persona` attribute for the `#[agent]` macro. This allows you to give your agents a consistent personality and memory.

**Use Case:** Building chatbots, game characters, or any AI that needs to remember past interactions and respond in character.

**Method 1: Manual Wrapping with `PersonaAgent` (for custom logic)**

You can manually wrap any existing agent with `PersonaAgent` to add persona and dialogue history.

```rust
use llm_toolkit::agent::{Agent, Persona, PersonaAgent};
use llm_toolkit::agent::impls::ClaudeCodeAgent;

// 1. Define a persona
let philosopher_persona = Persona {
    name: "Unit 734",
    role: "Philosopher Robot",
    background: "An android created to explore the nuances of human consciousness.",
    communication_style: "Speaks in a calm, measured tone, often using rhetorical questions.",
};

// 2. Create a base agent
let base_agent = ClaudeCodeAgent::default();

// 3. Wrap it with PersonaAgent
let character_agent = PersonaAgent::new(base_agent, philosopher_persona);

// 4. Interact
let response1 = character_agent.execute("Please introduce yourself.".into()).await?;
let response2 = character_agent.execute("What is your purpose?".into()).await?; // Remembers the first interaction
```

**Method 2: Simplified Usage with `#[agent(persona = ...)]` (Recommended)**

For maximum convenience, you can directly specify a persona in the `#[agent]` macro. The macro will automatically handle the `PersonaAgent` wrapping for you, preserving the inner agent's output type (structured data, attachments, etc.).

```rust
use llm_toolkit::agent::{Agent, persona::Persona};
use std::sync::OnceLock;

// Define a persona using a static or a function
const YUI_PERSONA: Persona = Persona {
    name: "Yui",
    role: "World-Class Pro Engineer",
    background: "A professional and precise AI assistant.",
    communication_style: "Clear, concise, and detail-oriented.",
};

// Use the persona directly in the agent macro
#[llm_toolkit::agent(
    expertise = "Analyzing technical requirements and providing implementation details.",
    persona = "self::YUI_PERSONA"
)]
struct YuiAgent;

// The agent is now stateful and will respond as Yui
let yui = YuiAgent::default();
let response = yui.execute("Introduce yourself.".into()).await?;
// Yui will introduce herself according to her persona and remember this interaction.
```

**Features:**
- ✅ **Stateful Conversation**: Automatically manages and includes dialogue history in prompts.
- ✅ **Consistent Personality**: Enforces a character's persona across multiple turns.
- ✅ **Excellent DX**: The `#[agent(persona = ...)]` attribute makes creating character agents trivial.
- ✅ **Composable**: `PersonaAgent` can wrap *any* agent that implements `Agent`.
- ✅ **Multimodal-Friendly**: Accepts full `Payload` inputs so persona agents can inspect attachments.

**Visual Identity for Enhanced Recognition**

Personas support optional visual identities (icons, taglines, and colors) that strengthen LLM role adherence and improve multi-agent dialogue clarity:

```rust
use llm_toolkit::agent::persona::{Persona, VisualIdentity};

let alice = Persona::new("Alice", "UI/UX Designer")
    .with_background("10 years of user-centered design experience")
    .with_communication_style("Visual, empathetic, user-focused")
    .with_visual_identity(
        VisualIdentity::new("🎨")
            .with_tagline("User-Centered Design Advocate")
            .with_color("#FF6B6B")
    );

// Or use the convenience method for quick icon addition
let bob = Persona::new("Bob", "Backend Engineer")
    .with_background("Senior engineer specializing in distributed systems")
    .with_communication_style("Technical, pragmatic")
    .with_icon("🔧");

// Visual identities appear in dialogue history and speaker names
println!("{}", alice.display_name()); // "🎨 Alice"
```

**Benefits:**
- ✅ **Enhanced Recognition**: Icons provide visual anchors for LLMs
- ✅ **Improved Clarity**: Easier to distinguish agents in conversation logs
- ✅ **Stronger Adherence**: LLMs maintain role consistency better
- ✅ **Human Readability**: Users quickly identify agents at a glance
- ✅ **Professional**: Taglines communicate expertise clearly
- ✅ **Future-Ready**: Color codes enable UI integration

See `examples/persona_visual_identity.rs` for a complete demonstration.

##### 4. Multi-Agent Dialogue Simulation

For use cases that require simulating conversations *between* multiple AI agents, the `Dialogue` component provides a powerful and flexible solution. It manages the turn-taking, shared history, and execution flow, enabling complex multi-agent interactions like brainstorming sessions or workflow pipelines.

**Core Concepts:**

-   **`Dialogue`**: The main orchestrator for the conversation.
-   **Execution Strategy**: Determines how agents interact. Five strategies are provided:
    -   **`Sequential`**: A pipeline where agents execute in a chain (`A -> B -> C`), with the output of one becoming the input for the next. Ideal for data processing workflows.
    -   **`Broadcast`**: A 1-to-N pattern where all agents respond to the same prompt. Ideal for brainstorming or getting multiple perspectives.
    -   **`Ordered Sequential`**: Similar to Sequential, but allows you to specify the exact execution order by participant names (e.g., `Designer -> PM -> Engineer`). Any participants not in the order list execute afterward in their original order.
    -   **`Ordered Broadcast`**: Similar to Broadcast, but responses are yielded in a custom-specified order rather than completion order. All agents still execute in parallel, but results are buffered and returned in your desired sequence.
    -   **`Mentioned`**: Only `@mentioned` participants respond (e.g., `@Alice @Bob what do you think?`). Falls back to Broadcast if no mentions are found. Perfect for selective participation in group conversations. Supports multiple **mention matching strategies**:
        - **`ExactWord`** (default): Matches `@word` patterns without spaces (e.g., `@Alice`, `@Bob123`, `@太郎`)
        - **`Name`**: Matches full names including spaces (e.g., `@Ayaka Nakamura` matches participant "Ayaka Nakamura"). Requires explicit delimiter (space, comma, period, etc.) after the name.
        - **`Partial`**: Matches by prefix, selecting the longest candidate (e.g., `@Ayaka` matches "Ayaka Nakamura")

        **Delimiter Support**:
        - **ExactWord/Partial**: Mentions are recognized until whitespace or common delimiters (`,` `.` `!` `?` `;` `:` `()` `[]` `{}` `<>` `"` `'` `` ` `` `/` `\` `|`). Example: `@Alice, what do you think?` or `@Bob!`
        - **Name**: Requires space or basic punctuation (`,` `.` `!` `?` `;` `:`) after the full name. For Japanese honorifics, use spaces: `@あやか なかむら さん` (not `@あやか なかむらさん`)

**Usage Example:**

```rust
use llm_toolkit::agent::chat::Chat;
use llm_toolkit::agent::dialogue::{Dialogue, SequentialOrder};
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

// (Mock agent and personas for demonstration)
# #[derive(Clone)]
# struct MockLLMAgent { agent_type: String }
# #[async_trait]
# impl Agent for MockLLMAgent {
#     type Output = String;
#     type Expertise = &'static str;
#     fn expertise(&self) -> &&'static str {
#         const EXPERTISE: &str = "mock";
#         &EXPERTISE
#     }
#     async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
#         let last_line = intent.to_text().lines().last().unwrap_or("").to_string();
#         Ok(format!("[{}] processed: '{}'", self.agent_type, last_line))
#     }
# }
# const SUMMARIZER_PERSONA: Persona = Persona { name: "Summarizer", role: "Summarizer", background: "...", communication_style: "..." };
# const TRANSLATOR_PERSONA: Persona = Persona { name: "Translator", role: "Translator", background: "...", communication_style: "..." };
# const CRITIC_PERSONA: Persona = Persona { name: "Critic", role: "Critic", background: "...", communication_style: "..." };

// --- Pattern 1: Sequential Pipeline ---
let summarizer = Chat::new(MockLLMAgent { agent_type: "Summarizer".to_string() })
    .with_persona(SUMMARIZER_PERSONA).with_history(false).build();
let translator = Chat::new(MockLLMAgent { agent_type: "Translator".to_string() })
    .with_persona(TRANSLATOR_PERSONA).with_history(false).build();

let mut dialogue = Dialogue::sequential();
dialogue.add_participant(summarizer).add_participant(translator);
let final_result = dialogue.run("A long article text...").await?;
// final_result: Ok(vec!["[Translator] processed: '[Summarizer] processed: 'A long article text...'"])

// Need a different execution chain? Pin the order by persona name.
dialogue.with_sequential_order(SequentialOrder::Explicit(vec![
    "Translator".to_string(), // run Translator first
    "Summarizer".to_string(), // then Summarizer
    // any other participants (e.g., reviewers) run afterward in their original order
]));

// Or create an ordered sequential dialogue from the start:
let mut ordered_dialogue = Dialogue::ordered_sequential(vec![
    "Designer".to_string(),
    "PM".to_string(),
    "Engineer".to_string(),
]);
// Participants execute in: Designer -> PM -> Engineer order

// --- Pattern 2: Broadcast ---
let critic = Chat::new(MockLLMAgent { agent_type: "Critic".to_string() })
    .with_persona(CRITIC_PERSONA).with_history(false).build();
let translator_b = Chat::new(MockLLMAgent { agent_type: "Translator".to_string() })
    .with_persona(TRANSLATOR_PERSONA).with_history(false).build();

let mut dialogue = Dialogue::broadcast();
dialogue.add_participant(critic).add_participant(translator_b);
let responses = dialogue.run("The new API design is complete.").await?;
// responses: Ok(vec!["[Critic] processed: 'The new API design is complete.'", "[Translator] processed: 'The new API design is complete.'"])

// Want responses in a specific order? Use ordered_broadcast:
let mut ordered_broadcast = Dialogue::ordered_broadcast(vec![
    "Jordan".to_string(),  // UX Designer's perspective first
    "Alex".to_string(),    // Engineer's technical view second
    "Sam".to_string(),     // PM's business view last
]);
// All agents run in parallel, but results are yielded in: Jordan, Alex, Sam order

// --- Pattern 3: Mentioned (Selective Participation) ---
# const ALICE_PERSONA: Persona = Persona { name: "Alice", role: "Backend", background: "...", communication_style: "..." };
# const BOB_PERSONA: Persona = Persona { name: "Bob", role: "Frontend", background: "...", communication_style: "..." };
# const CHARLIE_PERSONA: Persona = Persona { name: "Charlie", role: "QA", background: "...", communication_style: "..." };
let alice = Chat::new(MockLLMAgent { agent_type: "Alice".to_string() })
    .with_persona(ALICE_PERSONA).with_history(false).build();
let bob = Chat::new(MockLLMAgent { agent_type: "Bob".to_string() })
    .with_persona(BOB_PERSONA).with_history(false).build();
let charlie = Chat::new(MockLLMAgent { agent_type: "Charlie".to_string() })
    .with_persona(CHARLIE_PERSONA).with_history(false).build();

let mut dialogue = Dialogue::mentioned();
dialogue
    .add_participant(alice)
    .add_participant(bob)
    .add_participant(charlie);

// Only Alice and Bob respond
let turn1 = dialogue.run("@Alice @Bob what's your initial take?").await?;
// turn1: Ok(vec![DialogueTurn from Alice, DialogueTurn from Bob])

// Charlie can respond to their discussion
let turn2 = dialogue.run("@Charlie your QA perspective?").await?;
// turn2: Ok(vec![DialogueTurn from Charlie])

// No mentions → falls back to Broadcast (everyone responds)
let turn3 = dialogue.run("Any final thoughts?").await?;
// turn3: Ok(vec![DialogueTurn from Alice, Bob, Charlie])

// Get participant names for UI auto-completion
let names = dialogue.participant_names();
// names: vec!["Alice", "Bob", "Charlie"]
```

**Mention Matching Strategies:**

For participants with space-containing names like "Ayaka Nakamura", use alternative matching strategies:

```rust
use llm_toolkit::agent::dialogue::{Dialogue, MentionMatchStrategy};

# const AYAKA_PERSONA: Persona = Persona {
#     name: "Ayaka Nakamura",
#     role: "Designer",
#     background: "...",
#     communication_style: "..."
# };
let ayaka = Chat::new(MockLLMAgent { agent_type: "Ayaka".to_string() })
    .with_persona(AYAKA_PERSONA).with_history(false).build();

// Strategy 1: Name matching (exact full name with spaces)
let mut dialogue = Dialogue::mentioned_with_strategy(MentionMatchStrategy::Name);
dialogue.add_participant(ayaka);
let turn = dialogue.run("@Ayaka Nakamura please review this design").await?;
// Matches "Ayaka Nakamura" exactly

// Strategy 2: Partial matching (prefix-based, longest match)
let mut dialogue = Dialogue::mentioned_with_strategy(MentionMatchStrategy::Partial);
dialogue.add_participant(ayaka);
let turn = dialogue.run("@Ayaka what do you think?").await?;
// Matches "Ayaka Nakamura" by prefix "Ayaka"
```

###### Mid-Dialogue Participation with JoiningStrategy

Add participants to an ongoing dialogue with controlled history visibility using `join_in_progress()`:

```rust
use llm_toolkit::agent::dialogue::{Dialogue, JoiningStrategy};
use llm_toolkit::agent::persona::Persona;

let mut dialogue = Dialogue::broadcast();
dialogue.add_participant(alice_persona, alice_agent);
dialogue.add_participant(bob_persona, bob_agent);

// Turn 1-2: Initial conversation
dialogue.run("What's the plan?").await?;
dialogue.run("Let's proceed").await?;

// Add expert consultant mid-conversation with NO historical bias
let consultant_persona = Persona {
    name: "Carol".to_string(),
    role: "Security Expert".to_string(),
    background: "Security specialist".to_string(),
    communication_style: "Analytical".to_string(),
    visual_identity: None,
    capabilities: None,
};

dialogue.join_in_progress(
    consultant_persona,
    consultant_agent,
    JoiningStrategy::Fresh  // No history - fresh perspective
);

// Turn 3: Carol participates without historical context
let turn3 = dialogue.run("Carol, security review please").await?;
```

**Available JoiningStrategy Options:**

```rust
// 1. Fresh: No history (ideal for unbiased expert consultation)
JoiningStrategy::Fresh

// 2. Full: Complete history (ideal for new team members needing context)
JoiningStrategy::Full

// 3. Recent: Only recent N turns (ideal for focused review)
JoiningStrategy::recent_with_turns(5)  // Last 5 turns only

// 4. Range: Custom turn range (advanced scenarios)
JoiningStrategy::range(10, Some(20))  // Turns 10-20
```

**Use Cases:**

- **Fresh**: External consultants, code reviewers, unbiased analysis
- **Full**: New team members, context-dependent tasks, comprehensive review
- **Recent**: Focused review, memory-constrained scenarios, quick catchup
- **Range**: Testing, debugging, specific conversation segment analysis

**Supported in All Modes:**
- ✅ Broadcast: All participants respond in parallel
- ✅ Mentioned: Only @mentioned participants respond
- ✅ Sequential: Participants execute in chain order
- ✅ Moderator: Delegates to above modes
- ✅ Streaming API (`partial_session()`): All modes supported

###### Adding Pre-Configured Agents with Custom Settings

When you need to configure PersonaAgent settings (like ContextConfig) before adding to a Dialogue, use `add_agent()`:

```rust
use llm_toolkit::agent::persona::{PersonaAgent, ContextConfig};
use llm_toolkit::agent::dialogue::Dialogue;

// Configure ContextConfig for better long conversation handling
let config = ContextConfig {
    long_conversation_threshold: 5000,
    recent_messages_count: 10,
    participants_after_context: true,  // Place Participants after Context
    include_trailing_prompt: true,     // Add "YOU (name):" at the end
};

// Create PersonaAgent with custom config
let alice_persona = Persona {
    name: "Alice".to_string(),
    role: "Engineer".to_string(),
    background: "Senior developer".to_string(),
    communication_style: "Technical".to_string(),
    visual_identity: None,
    capabilities: None,
};

let persona_agent = PersonaAgent::new(base_agent, alice_persona.clone())
    .with_context_config(config);

// Add pre-configured agent to dialogue
let mut dialogue = Dialogue::sequential();
dialogue.add_agent(alice_persona, persona_agent);
```

Alternatively, you can use the Chat builder with `with_context_config()`:

```rust
let chat_agent = Chat::new(base_agent)
    .with_persona(persona)
    .with_context_config(config)
    .build();

dialogue.add_participant(persona, chat_agent);
```

###### Streaming Results with `partial_session`

Interactive shells and UI frontends can consume responses incrementally:

```rust
let mut session = dialogue.partial_session("Draft release plan");

while let Some(turn) = session.next_turn().await {
    let turn = turn?; // handle AgentError per participant
    println!("[{}] {}", turn.speaker.name(), turn.content);
}
```

- **Broadcast** sessions stream each agent’s reply as soon as it finishes (fast responders appear first).
- **Sequential** sessions expose intermediate outputs (`turn.content`) before they’re fed into the next participant, so you can surface progress step-by-step.

You can also rely on the built-in `tracing` instrumentation (`target = "llm_toolkit::dialogue"`) to monitor progress without polling the session manually. Attach a `tracing_subscriber` layer and watch for `dialogue_turn_completed` / `dialogue_turn_failed` events to drive dashboards or aggregate metrics.

Need deterministic ordering instead of completion order? Create the session with `partial_session_with_order(prompt, BroadcastOrder::ParticipantOrder)` to buffer results until all earlier participants have responded.

The existing `Dialogue::run` helper still collects everything for you (and, in sequential mode, keeps returning only the final turn) by internally driving a `partial_session` to completion.

###### Multi-Turn Conversations

Execute multiple turns by calling `run()` repeatedly. The dialogue automatically maintains conversation history, so agents have full context in each subsequent turn:

```rust
let mut dialogue = Dialogue::sequential()
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Turn 1: Initial discussion
let turn1 = dialogue.run("Let's discuss the architecture").await?;

// Turn 2: Agents see turn1 in history
let turn2 = dialogue.run("What are the trade-offs?").await?;

// Turn 3: Agents see all previous turns
let turn3 = dialogue.run("Make a final decision").await?;
```

Each `run()` call:
- Executes one turn with all (or mentioned) participants
- Automatically stores results in the message history
- Provides full context to agents in subsequent turns

**Continuing the Conversation:**

Use system messages to guide multi-turn dialogues:

```rust
use llm_toolkit::agent::{Payload, PayloadMessage, Speaker};

// Turn 1: Initial prompt
dialogue.run("Brainstorm ideas for the new feature").await?;

// Turn 2: Ask agents to continue
let continue_prompt = Payload::from_messages(vec![
    PayloadMessage::new(Speaker::System, "NEXT: Refine the top 3 ideas"),
]);
dialogue.run(continue_prompt).await?;

// Turn 3: Request convergence
let decide_prompt = Payload::from_messages(vec![
    PayloadMessage::new(Speaker::System, "NEXT: Make a final decision"),
]);
dialogue.run(decide_prompt).await?;
```

**Convergence-Based Loop:**

For iterative discussions, implement your own convergence logic:

```rust
let mut dialogue = Dialogue::broadcast()
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Initial prompt
dialogue.run("Discuss the API design").await?;

// Continue until convergence or max turns
for turn_num in 1..10 {
    // Check for convergence keywords
    let history = dialogue.history();
    let last_content = history.last().map(|t| &t.content).unwrap_or("");

    if last_content.contains("DECISION:") || last_content.contains("AGREED:") {
        println!("Converged after {} turns", turn_num);
        break;
    }

    // Continue the discussion
    let next_prompt = Payload::from_messages(vec![
        PayloadMessage::new(
            Speaker::System,
            format!("NEXT: Continue discussion (Turn {})", turn_num + 1)
        ),
    ]);
    dialogue.run(next_prompt).await?;
}
```

This approach gives you full control over:
- Turn limits (cost control)
- Convergence conditions (content-based, time-based, etc.)
- Custom prompts for each turn
- Logging and debugging at each step

**Available Methods:**

The `Dialogue` component provides several methods for managing conversations:

-   **`participants() -> Vec<&Persona>`**: Access the list of participant personas. Useful for inspecting names, roles, backgrounds, and communication styles.
-   **`participant_names() -> Vec<&str>`**: Get the names of all participants as strings. Ideal for UI auto-completion of `@mentions`.
-   **`participant_count() -> usize`**: Get the current number of participants.
-   **`add_participant(persona, agent)`**: Dynamically add a new participant to the conversation.
-   **`remove_participant(name)`**: Remove a participant by name (useful for guest participants).
-   **`history() -> &[DialogueTurn]`**: Access the complete conversation history.
-   **`with_context(DialogueContext)`**: Apply a full dialogue context (talk style, environment, additional context) that the toolkit prepends as system guidance before each turn in both `run` and `partial_session`.
-   **`with_talk_style(TalkStyle)`**: Convenient method to set only the conversation style (Brainstorm, Debate, etc.).
-   **`with_environment(String)`**: Set environment information (e.g., "Production environment", "ClaudeCode").
-   **`with_additional_context(impl ToPrompt)`**: Add structured or string-based additional context that gets converted to prompts.

```rust
// Inspect participants
let personas = dialogue.participants();
for persona in personas {
    println!("Participant: {} ({})", persona.name, persona.role);
}

// Dynamically manage participants
dialogue.add_participant(expert_persona, expert_agent);
dialogue.run("Get expert opinion").await?;
dialogue.remove_participant("Expert")?;

// Access conversation history
for turn in dialogue.history() {
    println!("[{}]: {}", turn.speaker.name(), turn.content);
}

// Apply conversation context using convenient builder methods
use llm_toolkit::agent::dialogue::{DialogueContext, TalkStyle};

// Option 1: Use convenient builder methods
let mut dialogue = Dialogue::sequential();
dialogue
    .with_talk_style(TalkStyle::Brainstorm)    // Set conversation style
    .with_environment("Production environment") // Add environment info
    .with_additional_context("Focus on security and performance".to_string())
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Option 2: Build a complete context and apply it
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Debate)
    .with_environment("ClaudeCode environment")
    .with_additional_context("Technical deep-dive".to_string());

dialogue.with_context(context);

// Option 3: Use structured, type-safe additional context
#[derive(ToPrompt)]
struct ProjectContext {
    language: String,
    focus_areas: Vec<String>,
}

dialogue.with_additional_context(ProjectContext {
    language: "Rust".to_string(),
    focus_areas: vec!["Performance".to_string(), "Safety".to_string()],
});

dialogue.partial_session("Kickoff agenda").await?;
```

**Available TalkStyles:**
- `TalkStyle::Brainstorm` - Creative, exploratory, building on ideas
- `TalkStyle::Debate` - Challenging ideas, diverse perspectives
- `TalkStyle::DecisionMaking` - Analytical, weighing options
- `TalkStyle::ProblemSolving` - Systematic, solution-focused
- `TalkStyle::Review` - Constructive feedback, detailed analysis
- `TalkStyle::Planning` - Structured, forward-thinking
- `TalkStyle::Casual` - Relaxed, friendly conversation

The `DialogueContext` is generic and accepts any type implementing `ToPrompt` for additional context, enabling structured, type-safe context management:

```rust
// Custom context types are automatically converted to prompts
#[derive(ToPrompt)]
struct TeamContext {
    team_size: usize,
    experience_level: String,
    constraints: Vec<String>,
}

dialogue.with_additional_context(TeamContext {
    team_size: 5,
    experience_level: "Senior".to_string(),
    constraints: vec!["No breaking changes".to_string()],
});
```

**Capability Declaration and Policy Enforcement:**

The `Capability` system provides a hybrid architecture for managing what agents can do and what they're allowed to do in specific sessions:

- **Bottom-up (Static)**: Agents declare capabilities via `Agent::capabilities()` and `Persona.capabilities`
- **Top-down (Dynamic)**: Dialogues enforce policies via `DialogueContext.policy` to restrict what's allowed in a session

This separation distinguishes "what CAN be done" from "what is ALLOWED in this context."

```rust
use llm_toolkit::agent::{Capability, Persona};
use llm_toolkit::agent::dialogue::{Dialogue, DialogueContext};

// Declare agent capabilities on Persona
let researcher = Persona::new(
    "Alice",
    "Research Specialist",
    "Expert in data gathering and analysis",
    "Analytical and thorough",
)
.with_capabilities(vec![
    Capability::new("web:search").with_description("Search the web for information"),
    Capability::new("db:query").with_description("Query internal databases"),
    Capability::new("file:read").with_description("Read files from disk"),
]);

let writer = Persona::new(
    "Bob",
    "Technical Writer",
    "Creates clear documentation",
    "Clear and concise",
)
.with_capabilities(vec![
    Capability::new("file:write").with_description("Write content to files"),
    Capability::new("file:read").with_description("Read files from disk"),
]);

// Create a dialogue with policy restrictions
let context = DialogueContext::new()
    // Allow Alice only web search in this session (restrict db:query and file:read)
    .with_policy("Alice", vec![
        Capability::new("web:search"),
    ])
    // Allow Bob both capabilities
    .with_policy("Bob", vec![
        Capability::new("file:write"),
        Capability::new("file:read"),
    ]);

let dialogue = Dialogue::broadcast()
    .with_context(context)
    .add_participant(researcher, agent1)
    .add_participant(writer, agent2);

// During execution:
// - Alice sees Bob's capabilities: [file:write, file:read]
// - Bob sees Alice's capabilities: [web:search] (policy-filtered from original 3)
// - This information is distributed via ParticipantInfo
```

**How Capabilities Flow:**

1. **Declaration**: Agents declare what they can do via `Persona.with_capabilities()`
2. **Policy Filtering**: `DialogueContext.with_policy()` restricts allowed capabilities per participant
3. **Distribution**: `ParticipantInfo` carries filtered capabilities to other agents
4. **Coordination**: Agents discover what others can do and coordinate accordingly

**Capability Format:**

Capabilities use a `category:action` naming convention for clarity:

```rust
// Simple capability (just name)
let cap1 = Capability::new("api:weather");

// With description for LLM understanding
let cap2 = Capability::new("db:query")
    .with_description("Query the user database");

// Convenience conversions
let cap3: Capability = "file:read".into();
let cap4: Capability = ("api:call", "Call external APIs").into();
```

**Use Cases:**

- ✅ **Access Control**: Restrict capabilities based on session context (e.g., read-only mode)
- ✅ **Orchestrator Precision**: Select agents based on concrete capabilities, not just expertise
- ✅ **Dialogue Coordination**: Agents discover and leverage each other's capabilities
- ✅ **Dynamic Policy**: Different sessions can enforce different restrictions on the same agents

**Reaction Control with ReactionStrategy:**

The `ReactionStrategy` allows you to control when agents should react to messages, providing fine-grained control over dialogue participation.

```rust
use llm_toolkit::agent::dialogue::{Dialogue, ReactionStrategy};

let mut dialogue = Dialogue::broadcast();
dialogue
    .with_reaction_strategy(ReactionStrategy::UserOnly)  // Only react to user messages
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);
```

**Available Strategies:**

- **`ReactionStrategy::Always`** (default) - React to all messages
  ```rust
  // Default behavior - agents react to everything
  let dialogue = Dialogue::broadcast();  // Uses Always by default
  ```

- **`ReactionStrategy::UserOnly`** - Only react to User messages
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::UserOnly);
  // Agents only react when users speak, ignoring agent-to-agent or system messages
  ```

- **`ReactionStrategy::AgentOnly`** - Only react to Agent messages
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::AgentOnly);
  // Useful for observer agents that analyze other agents' responses
  ```

- **`ReactionStrategy::ExceptSystem`** - React to all messages except System messages
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::ExceptSystem);
  // Ignore system notifications while responding to users and other agents
  ```

- **`ReactionStrategy::Conversational`** - React to User or Agent messages only
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::Conversational);
  // Engage in conversation between users and agents, ignoring system messages
  ```

- **`ReactionStrategy::ExceptContextInfo`** - React to all messages except ContextInfo
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::ExceptContextInfo);
  // React to all message types including System, but skip ContextInfo background information
  ```

**Message Metadata and Types:**

The dialogue system supports rich message metadata including message types for context-aware processing:

```rust
use llm_toolkit::agent::dialogue::message::{MessageMetadata, MessageType};
use llm_toolkit::agent::{Payload, Speaker};

// Send a context information message (won't trigger reactions)
let context_payload = Payload::new().add_message_with_metadata(
    Speaker::System,
    "Background: Project uses Rust",
    MessageMetadata::new().with_type(MessageType::ContextInfo),
);

dialogue.run(context_payload).await?;  // Stored in history but doesn't trigger reactions
```

**Available Message Types:**

- `MessageType::Conversational` - Regular dialogue messages (default)
- `MessageType::ContextInfo` - Background information (never triggers reactions)
- `MessageType::Notification` - Status updates
- `MessageType::Alert` - Important notifications

**Use Cases:**

- ✅ **Selective Participation**: Control which messages trigger agent responses
- ✅ **Observer Agents**: Create agents that only analyze other agents' outputs
- ✅ **User-Focused Dialogues**: Ensure agents only respond to user input
- ✅ **Context Management**: Share background information without triggering unnecessary reactions

**Session Resumption and History Injection:**

The `Dialogue` component supports saving and resuming conversations, enabling persistent multi-turn dialogues across process restarts or session boundaries.

```rust
use llm_toolkit::agent::dialogue::Dialogue;

// Session 1: Initial conversation
let mut dialogue = Dialogue::broadcast()
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

let turns = dialogue.run("Discuss project architecture").await?;

// Save the conversation history
dialogue.save_history("session_123.json")?;

// --- Process restart or session end ---

// Session 2: Resume conversation from saved history
let saved_history = Dialogue::load_history("session_123.json")?;

let mut dialogue = Dialogue::broadcast()
    .with_history(saved_history)  // Inject saved history
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Continue from where we left off with full context
let more_turns = dialogue.run("Continue from last discussion").await?;
```

**Alternative: Simple Session Resumption with System Prompt**

For simpler use cases where you want agents to "remember" previous conversations without complex structured history management, use `with_history_as_system_prompt()`:

```rust
// Session 2: Resume with simpler approach
let saved_history = Dialogue::load_history("session_123.json")?;

let mut dialogue = Dialogue::broadcast()
    .with_history_as_system_prompt(saved_history)  // ← Inject as system context
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Agents receive full conversation context and can reference previous discussion
let more_turns = dialogue.run("Continue from last discussion").await?;
```

**When to use each approach:**

- **`with_history_as_system_prompt()`** - Use when:
  - ✅ You want simple session restoration with minimal complexity
  - ✅ Your conversation history fits within the LLM's context window
  - ✅ You need agents to "remember" and reference previous conversations
  - ✅ You don't need to query or filter the structured MessageStore

- **`with_history()`** - Use when:
  - ✅ You need structured MessageStore for querying/filtering history
  - ✅ You want agents to manage their own conversation history independently
  - ✅ You're building advanced dialogue features with complex history management
  - ✅ You need fine-grained control over message distribution

**DialogueTurn Structure:**

The `DialogueTurn` struct represents a single turn in the conversation with full speaker attribution:

```rust
pub struct DialogueTurn {
    pub speaker: Speaker,  // Who spoke (System/User/Agent with full role info)
    pub content: String,   // What was said
}
```

The `speaker` field uses the `Speaker` enum to preserve complete attribution information including roles, which is essential for session resumption and conversation analysis.

Key methods for session management:

-   **`with_history(history: Vec<DialogueTurn>)`**: Builder method to inject conversation history into a new dialogue instance as structured messages in the MessageStore. Following the Orchestrator Step pattern, this creates a fresh instance with pre-populated history rather than mutating existing state. **Preserves full speaker information including roles.** Use this for advanced dialogue features requiring structured history queries.

-   **`with_history_as_system_prompt(history: Vec<DialogueTurn>)`**: Builder method to inject conversation history as a formatted system prompt that all agents receive. This simpler approach converts the entire conversation history into readable text that agents can reference, ensuring they "remember" previous discussions. **Ideal for straightforward session restoration** when you don't need structured history management.

-   **`save_history(path)`**: Persists the current conversation history to a JSON file with complete speaker attribution.

-   **`load_history(path)`**: Loads conversation history from a JSON file, restoring all speaker details.

Use cases:
- ✅ **Persistent Conversations**: Maintain dialogue context across application restarts
- ✅ **Session Management**: Save and restore user conversation sessions
- ✅ **Conversation Archival**: Store dialogue history for later analysis
- ✅ **Stateful Chatbots**: Implement chatbots with long-term memory
- ✅ **Agent Memory**: Enable agents to reference and build upon previous conversations

See `examples/dialogue_session_resumption.rs` and `examples/dialogue_session_resumption_system_prompt.rs` for complete demonstrations.

**Multimodal Input Support:**

The `Dialogue` API accepts `impl Into<Payload>`, enabling both text-only and multimodal input (text + attachments) with complete backward compatibility.

```rust
use llm_toolkit::agent::Payload;
use llm_toolkit::attachment::Attachment;

// Text-only input (backward compatible)
dialogue.run("Discuss AI ethics").await?;
dialogue.partial_session("Brainstorm ideas");

// Multimodal input with single attachment
let payload = Payload::text("What's in this image?")
    .with_attachment(Attachment::local("screenshot.png"));
dialogue.run(payload).await?;

// Multiple attachments
let payload = Payload::text("Analyze these files")
    .with_attachment(Attachment::local("data.csv"))
    .with_attachment(Attachment::local("metadata.json"));
dialogue.partial_session(payload);
```

All dialogue methods (`run`, `partial_session`, `partial_session_with_order`) accept any type implementing `Into<Payload>`, including:
- `String` or `&str` for text-only input
- `Payload` for multimodal input with attachments

This design enables:
- ✅ **100% Backward Compatibility**: Existing code works without changes
- ✅ **Extensibility**: New `Payload` features automatically work
- ✅ **Type Safety**: Compiler-enforced correct usage
- ✅ **Zero Method Proliferation**: No `_with_payload` variants needed

**Multi-Message Payloads and Speaker Attribution:**

The `Dialogue` API supports multi-message payloads with explicit speaker attribution, enabling complex conversation structures with System prompts, User inputs, and Agent responses.

```rust
use llm_toolkit::agent::{Payload, PayloadMessage};
use llm_toolkit::agent::dialogue::message::Speaker;

// Create a payload with multiple messages from different speakers
let payload = Payload::from_messages(vec![
    PayloadMessage::system("Context: Project planning meeting"),
    PayloadMessage::user(
        "Alice",
        "Product Manager",
        "What features should we prioritize?",
    ),
]);

// All messages are stored with proper speaker attribution
let turns = dialogue.run(payload).await?;

// Access conversation history with full speaker information
for turn in dialogue.history() {
    match &turn.speaker {
        Speaker::System => println!("[System]: {}", turn.content),
        Speaker::User { name, role } => println!("[{} ({})]: {}", name, role, turn.content),
        Speaker::Agent { name, role } => println!("[{} ({})]: {}", name, role, turn.content),
    }
}
```

The `Speaker` enum provides three variants:
- **`System`**: System-generated prompts or instructions
- **`User { name, role }`**: Human user messages with name and role
- **`Agent { name, role }`**: AI agent responses with persona information

This enables:
- ✅ **Proper Attribution**: Distinguish between System, User, and Agent messages
- ✅ **Role Preservation**: User and Agent roles are preserved in history
- ✅ **Complex Conversations**: Support multi-speaker turns with System + User messages
- ✅ **Session Resumption**: Full speaker context is maintained across save/load cycles

**Dynamic Instructions with Prepend Methods:**

Control agent behavior on a per-turn basis by prepending instructions to payloads without modifying `Persona` definitions:

```rust
use llm_toolkit::agent::Payload;
use llm_toolkit::agent::dialogue::Speaker;

// Prepend a system instruction for this specific turn
let payload = Payload::text("Discuss the architecture")
    .prepend_system("IMPORTANT: Keep responses under 300 characters. Be concise.");

dialogue.run(payload).await?;

// Or use prepend_message for custom speaker attribution
let payload = Payload::text("User question")
    .prepend_message(Speaker::System, "Answer in bullet points only.");
```

This enables:
- ✅ **Dynamic Constraints**: Add turn-specific constraints (e.g., "be concise", "be detailed")
- ✅ **Temporary Instructions**: Inject context-specific guidance without permanent changes
- ✅ **Conversation Control**: Prevent verbosity escalation in multi-agent dialogues
- ✅ **Chaining Support**: Multiple `prepend_*` calls apply in FIFO order

**Use Cases:**
- **Verbosity Control**: Add "Keep responses under 300 characters" when agents start getting too verbose
- **Mode Switching**: Switch between detailed and concise modes based on user preference
- **Context-Specific Behavior**: "Analyze this image concisely" for image attachments
- **Multi-Agent Coordination**: Prevent agents from repeating what others have said

**Enhanced Context Formatting:**

Dialogue participants receive enhanced context that includes:

1. **Participants Table**: Shows all conversation participants with clear "(YOU)" marker
2. **Recent History**: Previous messages with turn numbers and timestamps
3. **Current Task**: The current prompt or request

```text
# Persona Profile
**Name**: Alice
**Role**: Product Manager
...

# Request
# Participants

- **Alice (YOU)** (Product Manager)
  Expert in product strategy

- **Bob** (Engineer)
  Senior backend engineer

# Recent History

## Turn 1 (2024-01-15 10:30:22)
### Bob (Engineer)
I think we should focus on performance...

# Current Task
What should be our next priority?
```

This structured format helps agents:
- ✅ **Understand Context**: Know who else is in the conversation
- ✅ **Self-Identification**: Clear "(YOU)" marker for the current speaker
- ✅ **Temporal Awareness**: Turn numbers and timestamps for conversation flow
- ✅ **Role Clarity**: See the roles and backgrounds of other participants

## Agent API and Multi-Agent Orchestration

`llm-toolkit` provides a powerful agent framework for building multi-agent LLM systems with a clear separation of concerns.

### Agent API: Capability and Intent Separation

The Agent API follows the principle of **capability and intent separation**:
- **Capability**: An agent declares what it can do via two associated types:
  - `type Output`: The structured output type this agent produces
  - `type Expertise`: The expertise definition (typically `&'static str` or `Expertise` from `llm-toolkit-expertise`)
  - `fn expertise(&self) -> &Self::Expertise`: Returns agent's expertise/capabilities
- **Intent**: The orchestrator provides what needs to be done as a `Payload` (multi-modal content)

This separation enables maximum reusability and flexibility.

**Basic Agent Implementation:**

```rust
use llm_toolkit::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

struct MyAgent;

#[async_trait]
impl Agent for MyAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Analyzes data and provides insights";
        &EXPERTISE
    }

    async fn execute(&self, payload: Payload) -> Result<String, AgentError> {
        // Implementation
        Ok("analysis result".to_string())
    }
}
```

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

### Multi-Agent Orchestration

For complex workflows requiring multiple agents, the `Orchestrator` coordinates execution with adaptive error recovery.

#### Core Concepts

- **BlueprintWorkflow**: A natural language description of your workflow (no rigid types needed)
- **StrategyMap**: An ad-hoc execution plan generated by LLM based on available agents
- **Adaptive Redesign**: Three-stage error recovery (Retry → Tactical → Full Regenerate)

#### Basic Orchestrator Usage

```rust
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator};
use llm_toolkit::agent::impls::ClaudeCodeAgent;

#[tokio::main]
async fn main() {
    // Define workflow in natural language
    let blueprint = BlueprintWorkflow::new(r#"
        Technical Article Workflow:
        1. Analyze the topic and create an outline
        2. Research key concepts
        3. Write the main content
        4. Generate title and summary
        5. Review and refine
    "#.to_string());

    // Create orchestrator (InnerValidatorAgent is automatically registered)
    let mut orchestrator = Orchestrator::new(blueprint);
    orchestrator.add_agent(Box::new(ClaudeCodeAgent::new()));

    // Execute workflow - the orchestrator will:
    // - Generate an optimal execution strategy
    // - Assign agents to each step
    // - Handle errors with adaptive redesign
    let result = orchestrator.execute(
        "Write a beginner-friendly article about Rust ownership"
    ).await;

    match result.status {
        llm_toolkit::orchestrator::OrchestrationStatus::Success => {
            println!("✅ Workflow completed!");
            println!("Steps executed: {}", result.steps_executed);
            println!("Redesigns triggered: {}", result.redesigns_triggered);
            if let Some(output) = result.final_output {
                println!("\nFinal output:\n{}", output);
            }
        }
        llm_toolkit::orchestrator::OrchestrationStatus::Failure => {
            eprintln!("❌ Workflow failed: {:?}", result.error_message);
        }
    }
}
```

#### Customizing Internal Agents with `with_internal_agents`

By default, `Orchestrator::new()` uses `ClaudeCodeAgent` and `ClaudeCodeJsonAgent` as internal agents for strategy generation and redesign decisions. You can inject custom internal agents for testing, different LLM backends, or specialized behavior.

**Why customize internal agents?**
- **Testing**: Use mock agents to test orchestration logic without external API calls
- **Different LLM providers**: Use Gemini, Ollama, or custom backends for strategy generation
- **Cost optimization**: Use cheaper models for internal decision-making
- **Offline execution**: Run workflows completely offline with mock agents

**Usage:**

```rust
use llm_toolkit::orchestrator::{BlueprintWorkflow, Orchestrator};
use llm_toolkit::agent::{Agent, AgentError, Payload};

// Define custom internal agents (e.g., mock agents for testing)
struct MockStrategyAgent;

#[async_trait::async_trait]
impl Agent for MockStrategyAgent {
    type Output = StrategyMap;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock strategy generator for testing";
        &EXPERTISE
    }

    async fn execute(&self, intent: Payload) -> Result<StrategyMap, AgentError> {
        // Return a predefined strategy for testing
        let mut strategy = StrategyMap::new("Mock workflow".to_string());
        strategy.add_step(/* ... */);
        Ok(strategy)
    }
}

struct MockDecisionAgent;

#[async_trait::async_trait]
impl Agent for MockDecisionAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Mock decision maker for testing";
        &EXPERTISE
    }

    async fn execute(&self, intent: Payload) -> Result<String, AgentError> {
        Ok("RETRY".to_string())  // Simple retry strategy
    }
}

// Create orchestrator with custom internal agents
let orchestrator = Orchestrator::with_internal_agents(
    blueprint,
    Box::new(MockDecisionAgent),      // For intent generation & redesign decisions
    Box::new(MockStrategyAgent),      // For StrategyMap generation
);

// The orchestrator now uses your custom agents for all internal operations
let result = orchestrator.execute(task).await;
```

**Default Internal Agents:**

When using `Orchestrator::new()`, the following internal agents are used:
- **Strategy Generation**: `ClaudeCodeJsonAgent` wrapped in `RetryAgent` (max 3 retries)
- **Intent & Redesign**: `ClaudeCodeAgent` wrapped in `RetryAgent` (max 3 retries)

Both agents are automatically wrapped with `RetryAgent` to ensure robustness in critical orchestration decisions.

**IMPORTANT for `with_internal_agents()`:**

When providing custom internal agents, **you should wrap them with `RetryAgent`** for production use:

```rust
use llm_toolkit::agent::impls::{RetryAgent, gemini::GeminiAgent};

let orchestrator = Orchestrator::with_internal_agents(
    blueprint,
    Box::new(RetryAgent::new(GeminiAgent::new(), 3)),  // Recommended
    Box::new(RetryAgent::new(GeminiAgent::new(), 3)),  // Recommended
);
```

Without `RetryAgent`, a single transient error (network timeout, rate limiting) could cause strategy generation to fail completely.

**Complete Offline Example:**

See `examples/orchestrator_with_mock.rs` for a complete example that runs entirely offline with mock agents:

```bash
cargo run --example orchestrator_with_mock --features agent,derive
```

#### Advanced: Custom Agents with Orchestrator

You can combine custom agents (defined with `#[derive(Agent)]`) with the orchestrator:

```rust
#[derive(Serialize, Deserialize)]
struct ResearchData {
    sources: Vec<String>,
    key_points: Vec<String>,
}

#[derive(Agent)]
#[agent(
    expertise = "Deep research on technical topics with source citations",
    output = "ResearchData"
)]
struct ResearchAgent;

#[derive(Agent)]
#[agent(
    expertise = "Writing clear, beginner-friendly technical content",
    output = "ArticleDraft"
)]
struct WriterAgent;

// Add both to orchestrator (InnerValidatorAgent is automatically registered)
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.add_agent(Box::new(ResearchAgent));
orchestrator.add_agent(Box::new(WriterAgent));

// The orchestrator will automatically select the best agent for each step
```

#### Orchestrator Features

- ✅ **Natural Language Blueprints**: Define workflows in plain English
- ✅ **Ad-hoc Strategy Generation**: LLM generates execution plans based on available agents
- ✅ **Two-Layer Error Recovery**: Combine RetryAgent (transient errors) + Orchestrator (structural errors)
- ✅ **3-Stage Error Recovery**:
  - **Retry**: For transient errors
  - **Tactical Redesign**: Modify failed steps and continue
  - **Full Regenerate**: Start over with a new strategy
- ✅ **Built-in Validation**: Automatic registration of `InnerValidatorAgent` as a fallback validator
- ✅ **Smart Context Management**: Automatic passing of outputs between steps with `ToPrompt` support
- ✅ **Configurable Error Recovery Limits**: Control retry behavior to prevent infinite loops
- ✅ **Fast Path Intent Generation**: Optional optimization to skip LLM calls for deterministic template substitution
- ✅ **Logging and Observability**: Stream execution logs in JSON format using `tracing` for real-time monitoring
- ✅ **Loop Control Flow**: Iterative refinement with `LoopBlock` (while/until convergence patterns)
- ✅ **Early Termination**: Conditional workflow exit with `TerminateInstruction`
- ✅ **Control Flow Safety**: Single-level loops only (nested loops rejected), global iteration limits
- ✅ **Execution Journal**: Complete execution history with step-by-step outcomes, timestamps, and error details
- ✅ **Strategy Lifecycle Management**: Unified trait for strategy injection, retrieval, and generation across orchestrator types

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

#### Configuring Error Recovery Limits

The orchestrator provides configurable limits for error recovery to prevent infinite loops and control API costs:

```rust
use llm_toolkit::orchestrator::{Orchestrator, OrchestratorConfig};

let mut orchestrator = Orchestrator::new(blueprint);

// Method 1: Set entire configuration at once
let config = OrchestratorConfig {
    max_step_remediations: 5,     // Maximum 5 attempts per step (initial + 4 retries)
    max_total_redesigns: 15,       // Maximum 15 redesigns (initial strategy not counted)
};
orchestrator.set_config(config);

// Method 2: Modify individual limits
orchestrator.set_max_step_remediations(5);
orchestrator.set_max_total_redesigns(15);

// Method 3: Use partial configuration with defaults
let config = OrchestratorConfig {
    max_step_remediations: 5,
    ..Default::default()  // Use default for max_total_redesigns (10)
};
orchestrator.set_config(config);
```

**Default Limits:**
- `max_step_remediations`: 3
  - Allows **3 execution attempts** per step (initial attempt + 2 retries)
  - Prevents infinite loops on a single failing step
- `max_total_redesigns`: 10
  - Allows **10 redesign operations** (initial strategy generation not counted)
  - Controls overall workflow redesign attempts across all steps

**How Counting Works:**

*Step-level counting:*
```
Step fails → count incremented → check if count >= max_step_remediations
- Attempt 1 (initial): Fails → count=1 → 1>=3? No → Retry
- Attempt 2: Fails → count=2 → 2>=3? No → Retry
- Attempt 3: Fails → count=3 → 3>=3? Yes → Error: MaxStepRemediationsExceeded
Result: max_step_remediations=3 allows 3 total attempts (2 retries)
```

*Total redesigns counting:*
```
Initial strategy generation → redesigns_triggered=0 (not counted)
Retry/TacticalRedesign/FullRegenerate → redesigns_triggered incremented
- First redesign: redesigns_triggered=1
- ...
- 10th redesign: redesigns_triggered=10 → 10>=10? Yes → Error: MaxTotalRedesignsExceeded
Result: max_total_redesigns=10 allows up to 11 total strategy executions
```

**When Limits Are Exceeded:**
- **Step limit exceeded**: Returns `OrchestratorError::MaxStepRemediationsExceeded { step_index, max_remediations }`
- **Total limit exceeded**: Returns `OrchestratorError::MaxTotalRedesignsExceeded(limit)`

**Choosing Good Values:**
- **Small workflows (2-3 steps)**: Default values work well
- **Large workflows (5+ steps)**: Consider increasing `max_total_redesigns` to 15-20
- **Critical steps**: If certain steps are known to be unstable, increase `max_step_remediations` to 5
- **Cost-sensitive**: Reduce both limits to fail faster (e.g., max_step_remediations=2, max_total_redesigns=5)

#### Rate Limiting with `min_step_interval`

The orchestrator provides proactive rate limiting to prevent API rate limit errors (429 Too Many Requests).

**Problem**: Each orchestrator step typically makes 2+ API calls (intent generation + execution). Without delays, a 6-step workflow can make 12+ calls in 30 seconds, exceeding many LLM API rate limits (e.g., 10 requests/minute for Gemini).

**Solution**: Set `min_step_interval` to introduce a delay after each step completes:

```rust
use std::time::Duration;
use llm_toolkit::orchestrator::{Orchestrator, OrchestratorConfig};

let mut orchestrator = Orchestrator::new(blueprint);

// Method 1: Set entire configuration at once
let config = OrchestratorConfig {
    min_step_interval: Duration::from_millis(500),  // 500ms delay between steps
    ..Default::default()
};
orchestrator.set_config(config);

// Method 2: Use convenience method
orchestrator.set_min_step_interval(Duration::from_secs(1));  // 1 second delay
```

**How It Works:**
- Applied **after** each step completes (before starting next step)
- **Not applied** after the last step (no unnecessary delay)
- `Duration::ZERO` means no delay (default, backward compatible)

**Choosing Good Values:**
- **10 req/min limit** (e.g., Gemini): Use `Duration::from_secs(6)` or higher
- **60 req/min limit** (e.g., Claude): Use `Duration::from_millis(500)` to `Duration::from_secs(1)`
- **Conservative approach**: Start with `Duration::from_secs(1)`, reduce if no errors occur

**Combining with RetryAgent:**

For maximum resilience, combine proactive rate limiting (min_step_interval) with reactive retry (RetryAgent):

```rust
use llm_toolkit::agent::impls::{GeminiAgent, RetryAgent};

// Layer 1: Proactive rate limiting (prevents errors)
orchestrator.set_min_step_interval(Duration::from_secs(1));

// Layer 2: Reactive retry with retry_after support (handles errors)
let gemini = GeminiAgent::new();
let retry_gemini = RetryAgent::new(gemini, 5);  // Respects server retry_after
orchestrator.add_agent(retry_gemini);

// Result: Minimal API errors and automatic recovery if they occur
```

#### Loop and Early Termination Control Flow

The orchestrator supports advanced control flow with loops and early termination, enabling iterative refinement and conditional workflow exit.

**Status**: ✅ **Complete and tested** (160 tests passing)

**Features:**
- ✅ Loop blocks with configurable iteration limits
- ✅ Early termination instructions with conditional evaluation
- ✅ Single-level loops only (nested loops rejected via validation)
- ✅ Optional fields for simplified LLM generation
- ✅ Execution engine with recursive instruction processing
- ✅ Condition template evaluation with MiniJinja
- ✅ Loop aggregation modes (LastSuccess, FirstSuccess, CollectAll)
- ✅ Global loop iteration limits (prevents runaway costs)
- ✅ Integrated with execute_strategy() (automatic legacy migration)

**Data Model:**

```rust
use llm_toolkit::orchestrator::{StrategyInstruction, LoopBlock, TerminateInstruction};

// Example 1: Minimal loop (optimal for LLM generation)
let loop_instruction = StrategyInstruction::Loop(LoopBlock {
    loop_id: "refine".to_string(),
    description: None,  // Optional
    loop_type: None,    // Optional (defaults to While)
    max_iterations: 3,
    condition_template: Some("{{ needs_improvement }}".to_string()),
    body: vec![/* nested instructions */],
    aggregation: None,  // Optional
});

// Example 2: Early termination
let terminate = StrategyInstruction::Terminate(TerminateInstruction {
    terminate_id: "early_exit".to_string(),
    description: None,  // Optional
    condition_template: Some("{{ success }}".to_string()),
    final_output_template: None,  // Optional
});
```

**Minimal JSON Example** (hand-written or LLM-generated):

```json
{
  "goal": "Iteratively refine design",
  "elements": [
    {
      "type": "step",
      "step_id": "initial_design",
      "description": "Create initial design",
      "assigned_agent": "DesignAgent",
      "intent_template": "Create design for {{ task }}",
      "expected_output": "Design document"
    },
    {
      "type": "loop",
      "loop_id": "refine_loop",
      "max_iterations": 5,
      "condition_template": "{{ feedback.needs_improvement }}",
      "body": [
        {
          "type": "step",
          "step_id": "get_feedback",
          "description": "Get design feedback",
          "assigned_agent": "ReviewAgent",
          "intent_template": "Review design",
          "expected_output": "Feedback"
        },
        {
          "type": "terminate",
          "terminate_id": "approved",
          "condition_template": "{{ feedback.approved }}"
        },
        {
          "type": "step",
          "step_id": "improve",
          "description": "Apply improvements",
          "assigned_agent": "DesignAgent",
          "intent_template": "Improve design based on {{ feedback }}",
          "expected_output": "Improved design"
        }
      ]
    }
  ]
}
```

**Configuration:**

```rust
use llm_toolkit::orchestrator::OrchestratorConfig;

let config = OrchestratorConfig {
    max_total_loop_iterations: 50,  // Global limit across all loops (default: 50)
    ..Default::default()
};
orchestrator.set_config(config);
```

**Safety Constraints:**
- Single-level loops only (nested loops are rejected with validation error)
- Global `max_total_loop_iterations` limit prevents runaway costs
- Each loop requires `max_iterations` (per-loop limit)
- Validation via `StrategyMap::validate()` before execution

**Design Decisions:**
- `description` and `loop_type` are **optional** to reduce LLM generation failures
- No `controller_agent` field (reuses existing `internal_agent` for LLM-driven control)
- `condition_template` uses MiniJinja for deterministic evaluation
- Backward compatible: legacy `steps` format still supported via `migrate_legacy_steps()`

**Performance Impact:**
- **6-step workflow with 1s delay**: Adds ~5 seconds total (6 steps - 1 last step)
- **Trade-off**: Slightly slower execution vs. no rate limit errors
- **Best practice**: Use only when targeting rate-limited APIs

#### Execution Journal

The orchestrator automatically captures complete execution history in an `ExecutionJournal`, providing detailed insights into workflow execution for debugging, auditing, and analysis.

**Status**: ✅ **Complete and tested** (12 dedicated tests passing)

**Features:**
- ✅ Automatic journal creation for every workflow run
- ✅ Step-by-step execution records with timestamps
- ✅ Complete strategy snapshot (goal, steps, dependencies)
- ✅ Success/failure status tracking per step
- ✅ Output capture for completed steps
- ✅ Error messages for failed steps
- ✅ Support for all step states (Pending, Running, Completed, Failed, Skipped, PausedForApproval)
- ✅ Available in both `Orchestrator` and `ParallelOrchestrator`

**Basic Usage:**

```rust
use llm_toolkit::orchestrator::{Orchestrator, ExecutionJournal};

let mut orchestrator = Orchestrator::new(blueprint);
let result = orchestrator.execute("Analyze customer data").await?;

// Access journal from result
if let Some(journal) = result.journal {
    println!("Workflow: {}", journal.strategy.goal);
    println!("Executed {} steps", journal.steps.len());

    for step in &journal.steps {
        println!("Step {}: {:?} at {}ms",
            step.step_id,
            step.status,
            step.recorded_at_ms
        );

        if let Some(error) = &step.error {
            println!("  Error: {}", error);
        }
    }
}

// Or access from orchestrator instance
if let Some(journal) = orchestrator.execution_journal() {
    // Process journal...
}
```

**ParallelOrchestrator Usage:**

```rust
use llm_toolkit::orchestrator::ParallelOrchestrator;

let mut orchestrator = ParallelOrchestrator::new(blueprint);
let result = orchestrator
    .execute("Process in parallel", token, None, None)
    .await?;

// Journal available in both success and failure cases
let journal = result.journal.expect("Journal should always be present");

// Analyze parallel execution
for step in &journal.steps {
    match step.status {
        StepStatus::Completed => {
            println!("✓ {} completed successfully", step.title);
        }
        StepStatus::Failed => {
            println!("✗ {} failed: {}", step.title,
                step.error.as_ref().unwrap());
        }
        StepStatus::Skipped => {
            println!("⊘ {} skipped (dependency failed)", step.title);
        }
        _ => {}
    }
}
```

**Journal Data Structure:**

```rust
pub struct ExecutionJournal {
    /// Strategy snapshot used for the run
    pub strategy: StrategyMap,
    /// Recorded step outcomes in execution order
    pub steps: Vec<StepRecord>,
}

pub struct StepRecord {
    pub step_id: String,
    pub title: String,
    pub agent: String,
    pub status: StepStatus,
    pub output_key: Option<String>,
    pub output: Option<JsonValue>,
    pub error: Option<String>,
    pub recorded_at_ms: u64,
}

pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    PausedForApproval,
}
```

**Use Cases:**
- **Debugging**: Trace exact execution flow and identify failure points
- **Auditing**: Keep permanent records of workflow executions
- **Analytics**: Analyze step performance and success rates
- **Reporting**: Generate execution reports for stakeholders
- **Testing**: Verify expected execution patterns in tests

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

#### Fast Path Intent Generation (Performance Optimization)

By default, the orchestrator uses LLM-based intent generation for each step, which provides high-quality, context-aware prompts but incurs API latency and costs. For workflows with simple template substitution (all placeholders resolved from context), you can enable **fast path optimization** to skip LLM calls.

**When to Enable:**
- ✅ **Thick Agents**: Agents that contain detailed domain logic and don't need LLM-optimized prompts
- ✅ **Simple Templates**: Intent templates with straightforward placeholder substitution
- ✅ **Performance-Critical Workflows**: When latency matters more than prompt quality
- ✅ **High-Volume Operations**: When API costs need to be minimized

**When to Keep Disabled (Default):**
- ❌ **Thin Agents**: Agents that rely on rich, context-aware prompts from the LLM
- ❌ **Complex Reasoning**: Workflows requiring semantic understanding and prompt adaptation
- ❌ **Quality-First Applications**: When prompt quality is more important than speed

**Usage:**

```rust
use std::time::Duration;
use llm_toolkit::orchestrator::{Orchestrator, OrchestratorConfig};

let mut orchestrator = Orchestrator::new(blueprint);

// Enable fast path optimization
let config = OrchestratorConfig {
    enable_fast_path_intent_generation: true,  // Default: false
    ..Default::default()
};
orchestrator.set_config(config);

// Execute - fast path will be used when all placeholders are resolved
let result = orchestrator.execute(task).await;
```

**How It Works:**

For each step, the orchestrator:
1. **Checks prerequisites**: Are all placeholders in the intent template resolved in context?
2. **Fast path (if enabled + all resolved)**: Simple string substitution (milliseconds, no API call)
3. **LLM path (fallback)**: LLM generates high-quality, context-aware intent (seconds, API call)

**Example:**

```rust
// Intent template from strategy
"Transform this data: {{previous_output}}"

// If fast path enabled and previous_output exists in context:
// → Fast path: Direct substitution → "Transform this data: <actual output>"
// → Latency: ~1ms, Cost: $0

// If fast path disabled or placeholder not resolved:
// → LLM path: Generate intent considering agent expertise → High-quality prompt
// → Latency: ~2s, Cost: ~$0.001
```

**Performance Benefits (Example E2E Test Results):**

```
3-step workflow with mock 100ms LLM delay:
- Fast Path ENABLED:  412ms (1.49x faster)
- Fast Path DISABLED: 615ms

Real-world with actual LLM calls:
- Fast Path: ~50ms per step → 150ms for 3 steps
- LLM Path: ~2s per step → 6s for 3 steps
- Speedup: 40x faster!
```

**Trade-offs:**

| Aspect | Fast Path (Enabled) | LLM Path (Disabled, Default) |
|--------|---------------------|------------------------------|
| **Performance** | ⚡ Milliseconds | 🐌 Seconds |
| **API Cost** | 💰 Zero | 💰💰 Per step |
| **Prompt Quality** | Basic (template substitution) | High (context-aware, semantic) |
| **Best For** | Thick agents, simple templates | Thin agents, complex reasoning |

**Best Practices:**

1. **Default to disabled** - Prioritize quality for thin agent architectures
2. **Enable selectively** - Use for specific workflows where you've validated template quality
3. **Test both modes** - Compare results to ensure fast path doesn't sacrifice quality
4. **Monitor logs** - Watch for `"Using fast path"` vs `"Using LLM-based intent generation"` messages

**Complete E2E Example:**

See `examples/orchestrator_fast_path_e2e.rs` for a complete example comparing both modes:

```bash
cargo run --example orchestrator_fast_path_e2e --features agent,derive
```

This example demonstrates:
- Performance comparison between fast path and LLM path
- Validation that both produce equivalent results
- Configuration toggling
- Practical speedup measurements

#### Two-Layer Error Recovery: RetryAgent + Orchestrator

The recommended pattern is to combine `RetryAgent` (agent-level retry) with Orchestrator (workflow-level recovery) for robust error handling:

```rust
use llm_toolkit::agent::impls::{ClaudeCodeAgent, RetryAgent};
use llm_toolkit::orchestrator::{Orchestrator, BlueprintWorkflow};

// Layer 1: Agent-level retry (transient errors)
let claude = ClaudeCodeAgent::new();
let retry_agent = RetryAgent::new(claude, 3);  // Up to 3 retries

// Layer 2: Orchestrator-level recovery (structural errors)
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.add_agent(Box::new(retry_agent));

// Now you have two layers of error recovery:
// - Agent layer: Network errors, 429 rate limits, parse errors
// - Orchestrator layer: Wrong agent selection, strategy issues
```

**Responsibility Separation:**

| Error Type | Layer | Recovery Strategy |
|------------|-------|-------------------|
| Network timeout | Agent (RetryAgent) | Wait + retry (linear backoff) |
| 429 rate limit | Agent (RetryAgent) | Wait retry_after (exponential, max 60s) |
| Parse error | Agent (RetryAgent) | Immediate retry (linear backoff) |
| Agent capability mismatch | Orchestrator | Try different agent (step remediation) |
| Strategy design flaw | Orchestrator | Redesign workflow (tactical/full) |

**Per-Agent Customization:**

You can customize retry behavior for each agent based on importance:

```rust
// Critical agent: More retries
let writer = WriterAgent::default();
let retry_writer = RetryAgent::new(writer, 5);  // 5 retries

// Lightweight agent: Fewer retries
let validator = ValidatorAgent::default();
let retry_validator = RetryAgent::new(validator, 2);  // 2 retries

orchestrator.add_agent(Box::new(retry_writer));
orchestrator.add_agent(Box::new(retry_validator));
```

**Cost Control:**

Worst case: Agent retries × Orchestrator remediations
- Agent: 3 attempts (1 initial + 2 retries)
- Orchestrator: 3 remediations
- Maximum: 3 × 3 = 9 agent calls per step

This is **intentional design**:
- Agent retries handle transient errors (network, API)
- Orchestrator remediations handle structural errors (strategy, capability)
- Both limits are independently configurable for cost control

**Why This Pattern Works:**

- ✅ **Clear Separation**: Transient vs structural errors handled at appropriate levels
- ✅ **DRY Principle**: Same retry logic (RetryAgent) used everywhere
- ✅ **Flexible Control**: Independent configuration of agent and orchestrator retries
- ✅ **No Additional Code**: Uses existing RetryAgent decorator
- ✅ **Production-Ready**: 429 rate limiting, Full Jitter, retry_after support

**When NOT to use RetryAgent:**

If you want the Orchestrator to immediately try a different agent on first failure (no agent-level retry), add agents directly without wrapping:

```rust
// Direct agent addition - no agent-level retry
orchestrator.add_agent(Box::new(ClaudeCodeAgent::new()));

// First error → Orchestrator immediately tries different agent or redesigns
```

#### Placeholder Syntax in Intent Templates

Intent templates use **Mustache/Jinja2-style double curly braces** `{{ }}` for placeholder substitution. This is **not a typo** - single braces `{ }` are **not recognized**.

**Correct Syntax:**

```rust
"Create an outline for: {{ task }}"           // ✅ Correct
"Based on {{ previous_output }}, continue"    // ✅ Correct
"Transform {{ step_3_output }}"               // ✅ Correct
```

**Incorrect Syntax:**

```rust
"Create an outline for: {task}"               // ❌ Will NOT be recognized
"Based on {previous_output}, continue"        // ❌ Will NOT be recognized
```

**Important Notes:**

- Always use **double curly braces** with spaces: `{{ name }}` (not `{{name}}`)
- This matches the Mustache/Jinja2 templating convention
- The orchestrator's `extract_placeholders` only detects `{{ }}` format
- LLM-generated intent templates follow this convention from prompts.rs

**Common Placeholders:**

- `{{ task }}` - The original user task
- `{{ previous_output }}` - Output from the immediately previous step
- `{{ step_N_output }}` - Output from a specific step (e.g., `{{ step_3_output }}`)
- Custom semantic names (e.g., `{{ concept_content }}`, `{{ emblem_design }}`)

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

#### Template-Based Context Access with Jinja2

The orchestrator uses **minijinja template rendering** to make context data available to agents through intent templates.

**How It Works:**

```rust
// Step 3's intent template (generated by strategy LLM)
"Create a character profile using:
- Concept: {{ step_1_output.concept }}
- Design: {{ step_2_output.emblem }}
- World: {{ user_request.world_seed.aesthetics }}"
```

When executing Step 3, the orchestrator:

1. **Passes full context to minijinja**: All `step_N_output`, `user_request`, and other context data
2. **Minijinja resolves placeholders**: Only referenced fields are extracted and rendered
3. **Agent receives rendered intent**: Clean, readable text with all placeholders expanded

**Available Context Keys:**

The orchestrator maintains these keys in context:

- `step_{step_id}_output` - JSON output from each step (e.g., `step_1_output`, `step_2_output`)
  - **Automatic key**: Always created with `step_` prefix + step_id + `_output` suffix
  - Example: If `step_id` is `"step_1"`, the key becomes `"step_step_1_output"`
- `{output_key}` - Custom alias for step output (if `output_key` is specified in strategy)
  - **User-defined key**: Set via the `output_key` field in StrategyStep
  - Example: `"output_key": "world_concept"` creates key `"world_concept"`
  - Preferred for better readability (e.g., `{{ world_concept.theme }}` instead of `{{ step_step_1_output.theme }}`)
- `step_{step_id}_output_prompt` - ToPrompt version (human-readable string)
- `{output_key}_prompt` - ToPrompt version with custom alias (if `output_key` is specified)
- `previous_output` - Convenience reference to the immediately previous step's output
- `user_request` - External input data added via `context_mut().insert()`
- Custom keys - Any data added before execution

**Dot Notation for Nested Access:**

```rust
// Access nested JSON fields
{{ step_1_output.concept }}                     // Simple field
{{ step_2_output.data.items }}                  // Nested object
{{ user_request.world_seed.aesthetics }}        // Deep nesting
{{ step_3_output.results[0].name }}             // Array indexing (if supported)
```

**Benefits:**

- **Simple & Standard**: Uses standard Jinja2 templating, familiar to many developers
- **No Semantic Matching**: Direct key resolution—no LLM calls for placeholder mapping
- **Full Information**: Complete step outputs available, minijinja extracts what's needed
- **Type Safety**: Dot notation errors are caught at template render time
- **Automatic**: Strategy generation LLM creates appropriate placeholder references

**Common Pitfall:**

❌ **Manually extracting intermediate results**:
```rust
// DON'T DO THIS - Orchestrator handles it automatically!
let result = orchestrator.execute(task).await;
let concept = extract_from_context("step_1_output")?; // Not accessible!
let emblem = extract_from_context("step_2_output")?;  // Not accessible!
```

✅ **Correct approach - Design the final agent to aggregate**:
```rust
// The LAST agent's intent template should request all needed data
"Generate final output including:
- Concept: {{ concept_content }}
- Emblem: {{ emblem_design }}
- Profile: {{ character_profile }}"

// Then final_output contains everything
let result = orchestrator.execute(task).await;
let complete_data = result.final_output; // All data aggregated by final agent
```

**Understanding Context Keys and Placeholder Resolution:**

Intent templates reference context data using Jinja2-style placeholders. The orchestrator stores all data in a context HashMap and passes it to minijinja for template rendering.

**How Context Keys Work:**

1. **Step outputs are automatically stored**:
   - `step_{step_id}_output` - JSON version (e.g., `step_1_output`, `step_world_concept_generation_output`)
     - **Note**: The `step_` prefix is **automatically added**. If your `step_id` is `"step_1"`, the key becomes `"step_step_1_output"`.
   - `{output_key}` - Custom alias (e.g., `world_concept` if `output_key: "world_concept"` is specified)
   - `step_{step_id}_output_prompt` - ToPrompt version (if available)
   - `{output_key}_prompt` - ToPrompt version with custom alias (if available)
   - `previous_output` - Updated after each step to reference the most recent output

2. **Placeholder resolution is direct**:
   - `{{ step_1_output }}` → Looks up `step_1_output` key in context
   - `{{ world_concept.theme }}` → Looks up `world_concept` (custom output_key) then accesses `.theme` field
   - `{{ step_1_output.concept }}` → Looks up `step_1_output` then accesses `.concept` field
   - `{{ user_request.world_seed.aesthetics }}` → Looks up `user_request` then navigates nested fields
   - No semantic matching or alias resolution—just direct key lookup

3. **Accessing nested fields with dot notation**:
   - Intent templates support Jinja2-style dot notation
   - Example: `{{ step_3_output.user.profile.role }}` accesses nested JSON fields
   - Works for any depth of nesting in JSON objects

**Adding External Context:**

You can add custom context before execution using `context_mut()`:

```rust
orchestrator.context_mut().insert(
    "user_request".to_string(),
    serde_json::json!({"name": "Alice", "world_seed": {"aesthetics": "Gothic"}})
);
```

This data is **immediately available** in intent templates:

```rust
// Intent template can directly reference it
"Create a profile for {{ user_request.name }} with {{ user_request.world_seed.aesthetics }} aesthetics"
```

**Best Practice:**

Use direct, explicit placeholder references in intent templates:

```rust
// ✅ Recommended: Direct step references with dot notation
// Intent: "Process {{ step_1_output.concept }} and {{ step_2_output.design.colors }}"

// ✅ Also good: External context references
// Intent: "Use world seed: {{ user_request.world_seed.aesthetics }}"

// ✅ Convenience: previous_output for simple sequential workflows
// Intent: "Refine {{ previous_output }}"
```

**Why?** The orchestrator's `context` was internal. But now you can access it!

#### Accessing Intermediate Results (v0.13.6+)

You can now access intermediate step results using the context accessor methods:

```rust
let result = orchestrator.execute(task).await;

// Option 1: Get specific step output
if let Some(concept) = orchestrator.get_step_output("step_1") {
    // Deserialize to your type
    let concept: HighConceptResponse = serde_json::from_value(concept.clone())?;
    println!("Concept: {:?}", concept);
}

// Option 2: Get human-readable version (if ToPrompt was used)
if let Some(prompt) = orchestrator.get_step_output_prompt("step_1") {
    println!("Concept (readable):\n{}", prompt);
}

// Option 3: Get all step outputs
for (step_id, output) in orchestrator.get_all_step_outputs() {
    println!("Step {}: {:?}", step_id, output);
}

// Option 4: Access raw context
let context = orchestrator.context();
println!("Full context: {:?}", context);
```

**Available methods:**

- `context()` - Returns full context HashMap
- `get_step_output(step_id)` - Get JSON output of a specific step
- `get_step_output_prompt(step_id)` - Get ToPrompt version (human-readable)
- `get_all_step_outputs()` - Get all step outputs as HashMap

**Note:** These methods are available after `execute()` completes. The context is preserved until the next `execute()` call.

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

### Parallel Orchestrator with Human-in-the-Loop (HIL)

The `ParallelOrchestrator` supports a Human-in-the-Loop (HIL) capability, allowing agents to pause execution and explicitly request human approval before proceeding with critical, ambiguous, or safety-sensitive tasks. This feature transforms the orchestrator from a purely automated workflow engine into a collaborative partner that can safely wait for human guidance at key decision points.

#### Overview

The `ParallelOrchestrator` executes workflows based on dependency graphs, running independent steps concurrently in "waves" to maximize performance. The HIL feature builds upon the orchestrator's existing interrupt and resume (save/load state) functionality to provide a robust, auditable approval workflow.

Human-in-the-Loop is essential for scenarios where:
- **Safety-critical operations** require explicit confirmation (e.g., deploying to production, deleting data)
- **Ambiguous decisions** need human judgment (e.g., selecting the best approach from multiple options)
- **Compliance requirements** mandate human oversight for certain actions
- **Trust boundaries** exist between automated and manual processes

#### HIL Workflow

The Human-in-the-Loop workflow follows these steps:

1. **Agent Requests Approval**: An agent reaches a point requiring human input and returns `AgentOutput::RequiresApproval` instead of a standard result.

2. **Orchestrator Pauses**: The orchestrator receives the approval request, transitions the corresponding step into a `PausedForApproval` state, and gracefully stops execution.

3. **State Persistence**: Before stopping, the orchestrator automatically saves the complete `OrchestrationState` (including the paused step, approval message, and context) to a file using the existing `save_state_to` mechanism.

4. **Human Review**: The application notifies the user that approval is needed. The user inspects the saved state file, which contains:
   - The approval message explaining what needs review
   - The current payload/context from the agent
   - The complete workflow state

5. **Approval & State Modification**: To approve, the user (or an external tool) modifies the saved state file:
   - Changes the step's status from `PausedForApproval` to `Completed`
   - Optionally injects approved data into the shared context for downstream steps

6. **Orchestrator Resumes**: The application re-invokes the orchestrator using the `resume_from` parameter, pointing to the modified state file. The orchestrator loads the state and seamlessly continues execution from the now-approved step.

#### Implementing an Agent with Approval Requests

To enable an agent to request approval, implement the `DynamicAgent` trait and return `AgentOutput::RequiresApproval`:

```rust
use llm_toolkit::agent::{Agent, AgentError, AgentOutput, DynamicAgent, Payload};
use serde_json::{json, Value as JsonValue};

#[derive(Clone)]
struct DeploymentAgent;

#[async_trait::async_trait]
impl Agent for DeploymentAgent {
    type Output = JsonValue;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Handles production deployments with human approval";
        &EXPERTISE
    }

    async fn execute(&self, _input: Payload) -> Result<Self::Output, AgentError> {
        unreachable!("DeploymentAgent uses execute_dynamic")
    }
}

#[async_trait::async_trait]
impl DynamicAgent for DeploymentAgent {
    fn name(&self) -> String {
        "DeploymentAgent".to_string()
    }

    fn description(&self) -> &str {
        "Handles production deployments with human approval"
    }

    async fn execute_dynamic(&self, input: Payload) -> Result<AgentOutput, AgentError> {
        // Prepare deployment plan
        let deployment_plan = json!({
            "target": "production",
            "service": "user-api",
            "version": "v2.1.0",
            "estimated_downtime": "30 seconds"
        });

        // Request human approval before proceeding
        Ok(AgentOutput::RequiresApproval {
            message_for_human: "Please review and approve deployment to production: user-api v2.1.0".to_string(),
            current_payload: deployment_plan,
        })
    }
}
```

#### Using HIL in Workflows

Here's a complete example showing how to handle the pause-approve-resume cycle:

```rust
use llm_toolkit::orchestrator::{
    ParallelOrchestrator, StrategyMap, StrategyStep, OrchestrationState,
    parallel::StepState
};
use std::sync::Arc;
use std::path::Path;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    // Define workflow with approval step
    let mut strategy = StrategyMap::new("Production Deployment");

    strategy.add_step(StrategyStep::new(
        "validate_changes",
        "Validate code changes",
        "ValidationAgent",
        "Validate changes for {{ service }}",
        "validation_result",
    ));

    strategy.add_step(StrategyStep::new(
        "deploy",
        "Deploy to production",
        "DeploymentAgent",
        "Deploy {{ service }} with validation: {{ validation_result }}",
        "deployment_result",
    ));

    // Create orchestrator and register agents
    let mut orchestrator = ParallelOrchestrator::new(strategy.clone());
    orchestrator.add_agent("ValidationAgent", Arc::new(ValidationAgent));
    orchestrator.add_agent("DeploymentAgent", Arc::new(DeploymentAgent));

    let state_file = Path::new("/tmp/deployment_state.json");

    // First execution: run until pause
    let result = orchestrator
        .execute(
            "Deploy user-api service",
            CancellationToken::new(),
            None,                    // No resume (fresh start)
            Some(state_file),        // Save state on pause
        )
        .await
        .unwrap();

    if result.paused {
        println!("Workflow paused for approval:");
        println!("Reason: {}", result.pause_reason.unwrap());
        println!("State saved to: {:?}", state_file);

        // ================================================================
        // Human intervention: Review and approve
        // ================================================================

        // Read the saved state
        let state_json = std::fs::read_to_string(state_file)
            .expect("Failed to read state file");
        let mut saved_state: OrchestrationState =
            serde_json::from_str(&state_json)
                .expect("Failed to deserialize state");

        // Find the paused step
        let step_state = saved_state
            .execution_manager
            .get_state("deploy")
            .expect("Deploy step not found");

        // Inspect the approval request
        if let StepState::PausedForApproval { message, payload } = step_state {
            println!("Approval message: {}", message);
            println!("Deployment plan: {}", serde_json::to_string_pretty(&payload).unwrap());

            // User reviews and approves...
            // Modify the state: mark step as completed
            saved_state
                .execution_manager
                .set_state("deploy", StepState::Completed);

            // Inject approved deployment result into context
            saved_state.context.insert(
                "deployment_result".to_string(),
                json!({
                    "status": "approved_and_deployed",
                    "approved_by": "user@example.com",
                    "timestamp": "2024-01-15T10:30:00Z"
                })
            );
        }

        // Write modified state back
        let modified_json = serde_json::to_string_pretty(&saved_state)
            .expect("Failed to serialize state");
        std::fs::write(state_file, modified_json)
            .expect("Failed to write state");

        println!("Approval granted. Resuming workflow...");

        // ================================================================
        // Resume execution with approved state
        // ================================================================

        let mut orchestrator_resumed = ParallelOrchestrator::new(strategy);
        orchestrator_resumed.add_agent("ValidationAgent", Arc::new(ValidationAgent));
        orchestrator_resumed.add_agent("DeploymentAgent", Arc::new(DeploymentAgent));

        let final_result = orchestrator_resumed
            .execute(
                "Deploy user-api service",
                CancellationToken::new(),
                Some(state_file),    // Resume from modified state
                None,                // No need to save again
            )
            .await
            .unwrap();

        assert!(final_result.success, "Workflow should complete successfully");
        assert!(!final_result.paused, "Workflow should not pause again");

        println!("Deployment completed successfully!");
        println!("Final result: {:?}", final_result.context.get("deployment_result"));
    }
}
```

#### Key Features

- **Explicit Approval Contract**: Agents use `AgentOutput::RequiresApproval` to clearly signal when human input is needed.
- **State Transparency**: The saved state file contains all information needed for the user to make an informed decision.
- **Flexible Approval Process**: Users can approve by simply editing the JSON state file, or build custom approval workflows (web UIs, CLI tools, etc.) that modify the state programmatically.
- **Seamless Resumption**: The orchestrator resumes exactly where it left off, with no duplicate work or lost context.
- **Audit Trail**: The state file serves as a complete record of what was requested, what was approved, and when.

#### Return Values

When an agent requests approval, the orchestrator returns a `ParallelOrchestrationResult` with:
- `paused = true`: Indicates execution was paused
- `success = true`: The pause is intentional and successful, not an error
- `pause_reason = Some(message)`: Contains the approval message from the agent
- `steps_executed = 0` (typically): No steps complete when pausing for approval
- The state is saved to the file specified in `save_state_to`

After resuming with an approved state:
- `paused = false`: Execution completed normally
- `success = true`: Workflow completed successfully
- `steps_executed`: Count of steps executed during resume (excludes already-completed steps)
- `context`: Contains all outputs, including injected approval data

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

## Future Directions

### Image Handling Abstraction
A planned feature is to introduce a unified interface for handling image inputs across different LLM providers. This would abstract away the complexities of dealing with various data formats (e.g., Base64, URLs, local file paths) and model-specific requirements, providing a simple and consistent API for multimodal applications.
