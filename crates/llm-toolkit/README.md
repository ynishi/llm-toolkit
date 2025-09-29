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

For enums, the `ToPrompt` derive macro provides flexible ways to generate prompts that describe your enum variants for LLM consumption. You can use doc comments, custom descriptions, or exclude variants entirely.

#### Basic Usage with Doc Comments

By default, the macro extracts documentation from Rust doc comments (`///`) on both the enum and its variants:

```rust
use llm_toolkit::ToPrompt;

/// Represents different user intents for a chatbot
#[derive(ToPrompt)]
pub enum BasicIntent {
    /// User wants to greet or say hello
    Greeting,
    /// User is asking for help or assistance  
    Help,
}
```

#### Advanced Attribute Controls

The `ToPrompt` derive macro supports powerful attribute-based controls for fine-tuning the generated prompts:

- **`#[prompt("...")]`** - Provide a custom description that overrides the doc comment
- **`#[prompt(skip)]`** - Exclude a variant from the prompt entirely (useful for internal-only variants)
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
    Search { query: String },
    
    #[prompt("Custom: User is updating their profile settings and preferences")]
    UpdateProfile,
    
    #[prompt(skip)]
    InternalDebugAction,
    
    DeleteItem,
}

let action = UserAction::CreateDocument;
let p = action.to_prompt();
// The following would be printed:
// UserAction: Represents different actions a user can take in the system
//
// Possible values:
// - CreateDocument: User wants to create a new document
// - Search: User is searching for existing content
// - UpdateProfile: Custom: User is updating their profile settings and preferences
// - DeleteItem
```

Note how in the output:
- `CreateDocument` and `Search` use their doc comments
- `UpdateProfile` uses the custom description from `#[prompt("...")]`
- `InternalDebugAction` is completely excluded due to `#[prompt(skip)]`
- `DeleteItem` appears with just its name since it has no documentation

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

## Future Directions

### Image Handling Abstraction
A planned feature is to introduce a unified interface for handling image inputs across different LLM providers. This would abstract away the complexities of dealing with various data formats (e.g., Base64, URLs, local file paths) and model-specific requirements, providing a simple and consistent API for multimodal applications.