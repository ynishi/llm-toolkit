# llm-toolkit-expertise

**Agent as Code**: Graph-based composition system for LLM agent capabilities.

[![Crates.io](https://img.shields.io/crates/v/llm-toolkit-expertise.svg)](https://crates.io/crates/llm-toolkit-expertise)
[![Documentation](https://docs.rs/llm-toolkit-expertise/badge.svg)](https://docs.rs/llm-toolkit-expertise)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

`llm-toolkit-expertise` provides a flexible, composition-based approach to defining LLM agent expertise through **weighted knowledge fragments**. Instead of rigid inheritance hierarchies, expertise is built by composing independent fragments with priorities and contextual activation rules.

### Core Concepts

- 🧩 **Composition over Inheritance**: Build agents like equipment sets
- ⚖️ **Weighted Fragments**: Knowledge with priority levels (Critical/High/Normal/Low)
- 🎯 **Context-Driven**: Enable dynamic behavior based on TaskHealth and runtime context
- 📊 **Visualization**: Generate Mermaid graphs and tree views
- 🔧 **JSON Schema**: Full schema support for validation and tooling

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
llm-toolkit-expertise = "0.1.0"
```

### Basic Example

```rust
use llm_toolkit_expertise::{
    Expertise, WeightedFragment, KnowledgeFragment, Priority, ContextProfile, TaskHealth,
};

// Create a code reviewer expertise
let expertise = Expertise::new("rust-reviewer", "1.0")
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
        .with_context(ContextProfile::Conditional {
            task_types: vec!["security-review".to_string()],
            user_states: vec![],
            task_health: Some(TaskHealth::AtRisk),
        })
    );

// Generate prompt
println!("{}", expertise.to_prompt());

// Generate visualizations
println!("{}", expertise.to_tree());
println!("{}", expertise.to_mermaid());
```

## Features

### 🎛️ Priority Levels

Control how strongly knowledge should be enforced:

- **Critical**: Absolute must-follow (violations = error)
- **High**: Recommended/emphasized (explicit instruction)
- **Normal**: Standard context (general guidance)
- **Low**: Reference information (background)

```rust
WeightedFragment::new(fragment)
    .with_priority(Priority::Critical)
```

### 🔄 Context-Aware Activation

Fragments can be conditionally activated based on:

- **Task Types**: `"debug"`, `"security-review"`, `"refactor"`, etc.
- **User States**: `"beginner"`, `"expert"`, `"confused"`, etc.
- **Task Health**: `OnTrack`, `AtRisk`, `OffTrack`

```rust
ContextProfile::Conditional {
    task_types: vec!["debug".to_string()],
    user_states: vec!["beginner".to_string()],
    task_health: Some(TaskHealth::AtRisk),
}
```

### 📚 Knowledge Fragment Types

Five types of knowledge representation:

1. **Logic**: Thinking procedures with Chain-of-Thought steps
2. **Guideline**: Behavioral rules with positive/negative examples (Anchors)
3. **QualityStandard**: Evaluation criteria and passing grades
4. **ToolDefinition**: Tool interfaces (JSON format)
5. **Text**: Free-form text knowledge

```rust
// Logic fragment
KnowledgeFragment::Logic {
    instruction: "Systematic debugging".to_string(),
    steps: vec!["Reproduce", "Isolate", "Fix", "Verify"].iter().map(|s| s.to_string()).collect(),
}

// Guideline with anchoring
KnowledgeFragment::Guideline {
    rule: "Prefer explicit error handling".to_string(),
    anchors: vec![Anchor {
        context: "Parsing user input".to_string(),
        positive: "parse().map_err(|e| Error::Parse(e))?".to_string(),
        negative: "parse().unwrap()".to_string(),
        reason: "Unwrap can panic on bad input".to_string(),
    }],
}
```

### 📊 Visualization

Generate multiple visualization formats:

**Tree View**:
```rust
let tree = expertise.to_tree();
// Output:
// Expertise: rust-reviewer (v1.0)
// ├─ Tags: lang:rust, role:reviewer
// └─ Content:
//    ├─ [CRITICAL] Text: Always run cargo check...
//    └─ [HIGH] Logic: Check for security issues
//       └─ Health: ⚠️ At Risk
```

**Mermaid Graph**:
```rust
let mermaid = expertise.to_mermaid();
// Generates Mermaid syntax with color-coded priority nodes
```

### 🔗 llm-toolkit Integration

Enable the `integration` feature to use `ToPrompt` trait:

```toml
[dependencies]
llm-toolkit-expertise = { version = "0.1.0", features = ["integration"] }
```

```rust
use llm_toolkit::ToPrompt;

let expertise = Expertise::new("test", "1.0")
    .with_fragment(/* ... */);

let prompt_part = expertise.to_prompt()?;
```

### 📋 JSON Schema Generation

Generate JSON Schema for validation and tooling:

```rust
use llm_toolkit_expertise::{dump_expertise_schema, save_expertise_schema};

// Get schema as JSON
let schema = dump_expertise_schema();
println!("{}", serde_json::to_string_pretty(&schema)?);

// Save to file
save_expertise_schema("expertise-schema.json")?;
```

## Examples

The crate includes several examples:

```bash
# Basic expertise creation and usage
cargo run --example basic_expertise

# Generate JSON Schema
cargo run --example generate_schema

# Context-aware prompt generation
cargo run --example prompt_generation
```

## Architecture

### Composition over Inheritance

Unlike traditional inheritance-based systems, `llm-toolkit-expertise` uses **graph composition**:

- **No fragile base class problem**: Parent changes don't break children
- **Flexible mixing**: Combine arbitrary fragments with tags
- **Conflict resolution**: Higher priority always wins
- **Dynamic assembly**: Runtime context determines active fragments

### TaskHealth: Adaptive Behavior

The `TaskHealth` enum enables "gear shifting" based on task status:

- **OnTrack**: Speed mode (concise, confident)
- **AtRisk**: Careful mode (verify, clarify)
- **OffTrack**: Stop mode (reassess, consult)

This mirrors how senior engineers adjust their approach based on project health.

## Roadmap

### Phase 2: Prompt Compiler
- Dynamic System Prompt generation from weighted fragments
- Priority-based ordering (Critical first, Low last)
- Context-aware fragment selection engine

### Phase 3: State Analyzer
- Conversation history analysis
- TaskHealth and user_state inference
- Lightweight classifier for context detection

### Phase 4: Registry System
- Expertise storage and versioning
- Tag-based search and discovery
- Composition recommendations

## Design Philosophy

1. **Independence**: Works standalone, integrates optionally
2. **Extensibility**: Future Prompt Compiler/State Analyzer ready
3. **Type Safety**: Rust types + JSON Schema validation
4. **Simplicity**: Start simple, grow as needed

## Contributing

Contributions welcome! This is an early-stage project exploring new patterns for agent capability composition.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [llm-toolkit](https://github.com/ynishi/llm-toolkit) - Core LLM utilities and agent framework
- [llm-toolkit-macros](https://crates.io/crates/llm-toolkit-macros) - Derive macros for LLM types
