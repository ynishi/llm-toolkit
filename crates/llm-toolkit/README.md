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
| **Agent API** | Define reusable AI agents with expertise and structured outputs. | `Agent` trait, `#[derive(Agent)]` macro | Implemented |
| **Auto-JSON Enforcement** | Automatically add JSON schema instructions to agent prompts for better LLM compliance. | `#[derive(Agent)]` with `ToPrompt::prompt_schema()` integration | Implemented |
| **Built-in Retry** | Automatic retry on transient errors (ParseError, ProcessError, IoError) with configurable attempts. | `max_retries` attribute, `AgentError::is_retryable()` | Implemented |
| **Multi-Modal Payload** | Pass text and images to agents through a unified `Payload` interface with backward compatibility. | `Payload`, `PayloadContent` types | Implemented |
| **Multi-Agent Orchestration** | Coordinate multiple agents to execute complex workflows with adaptive error recovery. | `Orchestrator`, `BlueprintWorkflow`, `StrategyMap` | Implemented |
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

// Type-level: describe all possible variants
let schema = UserIntent::prompt_schema();
// Output:
// UserIntent: Represents different user intents for a chatbot
//
// Possible values:
// - Greeting: User wants to greet or say hello
// - Help: User is asking for help or assistance
```

**When to use which:**
- **`value.to_prompt()`** - When you need to describe a specific enum value to the LLM (e.g., "The user selected: Greeting")
- **`Enum::prompt_schema()`** - When you need to explain all possible options to the LLM (e.g., "Choose one of these intents...")

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

// Type-level schema
let schema = UserAction::prompt_schema();
// Output:
// UserAction: Represents different actions a user can take in the system
//
// Possible values:
// - CreateDocument: User wants to create a new document
// - Search: User is searching for existing content
// - UpdateProfile: Custom: User is updating their profile settings and preferences
// - DeleteItem
//
// Note: InternalDebugAction is excluded from schema due to #[prompt(skip)]
```

**Behavior of `#[prompt(skip)]`:**
- At **instance level** (`value.to_prompt()`): Shows only the variant name
- At **type level** (`Enum::prompt_schema()`): Completely excluded from the schema

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

## Agent API and Multi-Agent Orchestration

`llm-toolkit` provides a powerful agent framework for building multi-agent LLM systems with a clear separation of concerns.

### Agent API: Capability and Intent Separation

The Agent API follows the principle of **capability and intent separation**:
- **Capability**: An agent declares what it can do (`expertise`) and what it produces (`Output`)
- **Intent**: The orchestrator provides what needs to be done as a `Payload` (multi-modal content)

This separation enables maximum reusability and flexibility.

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
use std::path::PathBuf;

// Combine text and images
let payload = Payload::text("What's in this image?")
    .with_image(PathBuf::from("/path/to/image.png"));

let result = agent.execute(payload).await?;

// Or from raw image data
let image_bytes = std::fs::read("/path/to/image.png")?;
let payload = Payload::text("Describe this screenshot")
    .with_image_data(image_bytes);
```

**Backward Compatibility:**

All existing code using `String` continues to work thanks to automatic conversion:

```rust
// This still works unchanged
let result = agent.execute("Simple text intent".to_string().into()).await?;
```

**Note:** While the `Payload` type supports images, not all agent backends currently process them. `ClaudeCodeAgent` and `GeminiAgent` will log a warning if images are included but not yet supported by the CLI integration.

#### Defining Agents: Two Approaches

`llm-toolkit` provides two ways to define agents, each optimized for different use cases:

##### 1. Simple Agents with `#[derive(Agent)]` (Recommended for Prototyping)

For quick prototyping and simple use cases, use the derive macro:

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

**Features:**
- ✅ Simplest possible interface
- ✅ Minimal boilerplate
- ✅ Perfect for prototyping
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
// ### Schema for `ReviewResult`
// {
//   "quality_score": "number", // Overall quality score from 0 to 100,
//   "issues": "string[]", // List of identified issues,
//   "recommendations": "string[]" // Actionable recommendations for improvement
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

**Recommendation:** Always use `#[derive(ToPrompt)]` with `#[prompt(mode = "full")]` for structured outputs to get the best LLM compliance.

**Nested Schema Expansion:**

The schema generation automatically expands nested types that implement `ToPrompt`, including both `Vec<T>` and regular nested objects:

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

// Generated schema for ProducerOutput:
// ### Schema for `ProducerOutput`
// {
//   "evaluation_passed": "boolean", // Whether the evaluation passed all checks,
//   "results": [
//     {
//       "rule": "string", // The rule being checked,
//       "passed": "boolean", // Whether this specific rule passed
//     }
//   ], // List of evaluation results for each rule
// }
```

**How it works:**

- The macro detects `Vec<T>` fields at compile time
- At runtime (first call only), it calls `T::prompt_schema()` to get the nested schema
- The nested schema is inlined with proper indentation
- Result is cached with `OnceLock` for performance (zero cost after first call)

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

// Generated schema for EmblemResponse:
// ### Schema for `EmblemResponse`
// {
//   "obvious_emblem": {
//     "name": "string", // The name of the emblem,
//     "description": "string", // A description of the emblem
//   }, // An obvious, straightforward emblem,
//   "creative_emblem": {
//     "name": "string", // The name of the emblem,
//     "description": "string", // A description of the emblem
//   }, // A creative, unexpected emblem
// }
```

**How it works:**

- The macro detects non-primitive types at compile time
- For `Vec<T>`: expands as an array with T's schema inline
- For nested objects: expands the object's schema inline
- At runtime (first call only), it calls `T::prompt_schema()` to get the nested schema
- The nested schema is inlined with proper indentation
- Result is cached with `OnceLock` for performance (zero cost after first call)

**Limitations:**

- The nested type must implement `ToPrompt` with `#[prompt(mode = "full")]`
- Schema expansion happens at runtime on first call to `prompt_schema()` (then cached)
- Deep nesting is supported but may be harder for LLMs to parse
- Primitive types (String, i32, bool, Vec, Option, HashMap, etc.) are not expanded

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
// - ProcessError: Process communication issues
// - IoError: Temporary I/O failures
//
// Behavior:
// - Attempt 1 fails → wait 100ms → retry
// - Attempt 2 fails → wait 200ms → retry
// - Attempt 3 fails → wait 300ms → retry
// - All attempts exhausted → return error
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

**Design Philosophy:**

Agent-level retries are intentionally **simple and limited** (2-3 attempts, fixed delay):
- **Fail fast**: Quickly report errors to the orchestrator
- **Orchestrator is smarter**: Has broader context for complex error recovery
  - Try different agents
  - Redesign strategy
  - Escalate to human
- **System stability**: Simple local retries + complex orchestration at the top = robust system

This design aligns with the Orchestrator's 3-stage error recovery (Tactical → Full Redesign → Human Escalation).

##### 2. Advanced Agents with `#[agent(...)]` (Recommended for Production)

For production use, testing, and when you need agent injection:

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
        fn expertise(&self) -> &str { "mock" }
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
            fn expertise(&self) -> &str { "error mock" }
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
    fn expertise(&self) -> &str { "Olama agent" }
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

**When to use which:**
- **`#[derive(Agent)]`**: Quick scripts, prototyping, simple tools
- **`#[agent(...)]` with `backend`**: Production with Claude/Gemini
- **`#[agent(...)]` with `default_inner`**: Custom backends (Olama, local models, mocks)

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
- ✅ **3-Stage Error Recovery**:
  - **Retry**: For transient errors
  - **Tactical Redesign**: Modify failed steps and continue
  - **Full Regenerate**: Start over with a new strategy
- ✅ **Built-in Validation**: Automatic registration of `InnerValidatorAgent` as a fallback validator
- ✅ **Smart Context Management**: Automatic passing of outputs between steps with `ToPrompt` support
- ✅ **Configurable Error Recovery Limits**: Control retry behavior to prevent infinite loops

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

#### Semantic Variable Mapping: Smart Context Management

**Problem**: In traditional orchestrators, context grows linearly as each step adds its output. By step 10, you're sending ~5KB of previous outputs on every agent call, most of which are irrelevant.

**Solution**: The orchestrator uses **semantic variable mapping** to pass only the necessary context to each agent.

**How It Works:**

```rust
// Step 3's intent template (generated by strategy LLM)
"Create a character profile using:
- Concept: {{ concept_content }}
- Design: {{ emblem_design }}
- World: {{ world_setting }}"
```

When executing Step 3, the orchestrator:

1. **Extracts placeholders**: `["concept_content", "emblem_design", "world_setting"]`
2. **Matches to previous steps**:
   - `concept_content` → Step 1's output (keyword match: "concept")
   - `emblem_design` → Step 2's output (keyword match: "emblem", "design")
   - `world_setting` → Step 0's output (LLM semantic match if keyword fails)
3. **Builds minimal context**: Only includes matched step outputs (full version, no truncation)
4. **Passes to agent**: Agent receives only relevant information

**Matching Strategy (2-Stage):**

1. **Phase 1 - Keyword Matching**: Fast pattern matching on step descriptions and expected outputs
2. **Phase 2 - LLM Fallback**: If keywords fail, internal LLM performs semantic matching

**Benefits:**

- **Prevents Context Explosion**: O(k) where k = number of placeholders (typically 2-3), instead of O(n) where n = all previous steps
- **Full Information**: No truncation—sends complete output for matched steps
- **Semantic Clarity**: Variable names like `{{ concept_content }}` are meaningful to developers
- **Automatic**: Strategy generation LLM decides what context each step needs

**Example Output:**

```
Step 3 receives context:
{
  "concept_content": "<Full Step 1 output>",    // ~2KB
  "emblem_design": "<Full Step 2 output>",      // ~1.5KB
  "world_setting": "<Full Step 0 output>",      // ~1KB
  "previous_output": "<Step 2 output>"          // ~1.5KB
}
Total: ~6KB (vs ~25KB if all 10 previous steps were included)
```

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

## Future Directions

### Image Handling Abstraction
A planned feature is to introduce a unified interface for handling image inputs across different LLM providers. This would abstract away the complexities of dealing with various data formats (e.g., Base64, URLs, local file paths) and model-specific requirements, providing a simple and consistent API for multimodal applications.