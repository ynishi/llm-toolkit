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
| `#[prompt(as_serialize)]` | Enables dot access (e.g., `{{ field.name }}`) in templates. |
| `#[prompt(as_prompt)]` | Forces use of `to_prompt()` (default behavior). |

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

#### Nested Types and Dot Access

**ToPrompt Philosophy**: By default, nested types use their `to_prompt()` implementation—each type controls its own prompt representation. This is similar to how React components control their own rendering.

```rust
use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
struct Profile {
    name: String,
    role: String,
}

impl ToPrompt for Profile {
    fn to_prompt(&self) -> String {
        format!("Profile: {} ({})", self.name, self.role)
    }
}

// Default: uses to_prompt() - the Profile controls its representation
#[derive(ToPrompt, Serialize)]
#[prompt(template = "{{ profile }}")]
struct DefaultExample {
    profile: Profile,
}

let example = DefaultExample {
    profile: Profile {
        name: "Alice".to_string(),
        role: "Admin".to_string(),
    },
};

assert_eq!(example.to_prompt(), "Profile: Alice (Admin)");
```

**When you need dot access** (e.g., `{{ profile.name }}`), use `#[prompt(as_serialize)]`:

```rust
// as_serialize: enables dot access for this field
#[derive(ToPrompt, Serialize)]
#[prompt(template = "User: {{ profile.name }}, Role: {{ profile.role }}")]
struct DotAccessExample {
    #[prompt(as_serialize)]
    profile: Profile,
}

let example = DotAccessExample {
    profile: Profile {
        name: "Bob".to_string(),
        role: "User".to_string(),
    },
};

assert_eq!(example.to_prompt(), "User: Bob, Role: User");
```

**Mixed usage** - some fields with dot access, others with `to_prompt()`:

```rust
#[derive(ToPrompt, Serialize)]
#[prompt(template = "Name: {{ data.name }}, Description: {{ description }}")]
struct MixedExample {
    #[prompt(as_serialize)]  // dot access
    data: Profile,
    description: Profile,    // uses to_prompt()
}
```

**Deep nesting** works with `as_serialize`:

```rust
#[derive(Debug, Clone, Serialize, ToPrompt)]
struct Company {
    name: String,
    ceo: Profile,
}

#[derive(ToPrompt, Serialize)]
#[prompt(template = "Company: {{ company.name }}, CEO: {{ company.ceo.name }}")]
struct DeepExample {
    #[prompt(as_serialize)]
    company: Company,
}
```

**Vec and Option** work with both approaches:

```rust
// With as_serialize: iterate and access fields
#[derive(ToPrompt, Serialize)]
#[prompt(template = "{% for p in profiles %}{{ p.name }}, {% endfor %}")]
struct VecDotAccess {
    #[prompt(as_serialize)]
    profiles: Vec<Profile>,
}

// Without as_serialize: each element uses to_prompt()
#[derive(ToPrompt, Serialize)]
#[prompt(template = "{% for p in profiles %}{{ p }}\n{% endfor %}")]
struct VecToPrompt {
    profiles: Vec<Profile>,  // Each Profile calls to_prompt()
}
```

**When to use which:**

| Use Case | Attribute | Behavior |
|----------|-----------|----------|
| Type has custom `to_prompt()` representation | (default) | Calls `to_prompt()` on the field |
| Need to access nested fields in template | `#[prompt(as_serialize)]` | Uses `Serialize` for dot access |
| Mixing both in same struct | Mix per-field | Apply attribute only where needed |

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

