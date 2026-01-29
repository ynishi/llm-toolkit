# llm-toolkit

A low-level, unopinionated Rust toolkit for the LLM last mile problem.

[![Crates.io](https://img.shields.io/crates/v/llm-toolkit.svg)](https://crates.io/crates/llm-toolkit)
[![Documentation](https://docs.rs/llm-toolkit/badge.svg)](https://docs.rs/llm-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

High-level LLM frameworks like LangChain can be problematic in Rust. Their heavy abstractions often conflict with Rust's strengths, imposing significant constraints on developers.

`llm-toolkit` takes a different approach: a **minimalist, unopinionated toolkit** that provides robust utilities for LLM integration, much like how `candle` provides core building blocks for ML without dictating application architecture.

## Core Design Principles

1. **Minimalist & Unopinionated**: No imposed architecture. Use what you need.
2. **Focused on the "Last Mile Problem"**: Solving the boundary between typed Rust and unstructured LLM responses.
3. **Minimal Dependencies**: Primarily `serde` and `minijinja`.

## Workspace Structure

| Crate | Description |
|-------|-------------|
| [`llm-toolkit`](./crates/llm-toolkit) | Core library with extraction, prompts, agents, and orchestration |
| [`llm-toolkit-macros`](./crates/llm-toolkit-macros) | Procedural macros (`#[derive(ToPrompt)]`, `#[agent(...)]`, etc.) |

## Quick Start

```toml
[dependencies]
# Essential Core (default) - prompt, extract, intent, models
# Minimal dependencies, no async runtime required
llm-toolkit = "0.60"

# With derive macros (ToPrompt, define_intent)
llm-toolkit = { version = "0.60", features = ["derive"] }

# With agent CLI support (claude, gemini, codex, llama-cpp)
llm-toolkit = { version = "0.60", features = ["agent"] }

# With direct API clients (no CLI dependency)
llm-toolkit = { version = "0.60", features = ["all-apis"] }
# Or individual: "anthropic-api", "gemini-api", "openai-api"
```

### Feature Tiers

| Tier | Features | Use Case |
|------|----------|----------|
| **Essential** | `default` | Prompt templates, JSON extraction, intent parsing |
| **Essential + Derive** | `derive` | Add `#[derive(ToPrompt)]`, `#[define_intent]` macros |
| **Agent** | `agent` | CLI-based LLM agents (typed, async) |
| **Full** | `all-apis` | Direct HTTP API calls without CLI |

### Extract JSON from LLM Response

```rust
use llm_toolkit::extract_json;

let response = r#"Here's the data: {"status": "ok", "count": 42}"#;
let json = extract_json(response).unwrap();
// => {"status": "ok", "count": 42}
```

### Structured Prompts

```rust
use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(ToPrompt, Serialize)]
#[prompt(template = "Analyze {{code}} for {{language}} best practices.")]
struct CodeReview {
    code: String,
    language: String,
}

let review = CodeReview {
    code: "fn main() {}".into(),
    language: "Rust".into(),
};
println!("{}", review.to_prompt());
```

### Define an Agent

```rust
use llm_toolkit::{agent, Agent};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ReviewResult {
    issues: Vec<String>,
    score: u8,
}

#[agent(
    expertise = "You are a code reviewer. Analyze code and return issues.",
    output = "ReviewResult"
)]
struct CodeReviewer;
```

## Features

| Feature | Description |
|---------|-------------|
| **Content Extraction** | Extract JSON/code blocks from unstructured LLM responses |
| **Fuzzy JSON Repair** | Auto-fix LLM JSON syntax errors and typos in tagged enums |
| **Prompt Generation** | `#[derive(ToPrompt)]`, `prompt!` macro, external templates |
| **Intent Extraction** | `#[define_intent]` for structured intent parsing |
| **Agent API** | `#[agent(...)]` macro with retry, profiles, and multi-modal support |
| **Orchestration** | `Orchestrator` for multi-agent workflows |
| **Context Detection** | Rule-based and LLM-based context inference |
| **Type-Safe Models** | `ClaudeModel`, `GeminiModel`, `OpenAIModel` enums with validation |
| **Direct API Clients** | HTTP API agents for Anthropic, Gemini, OpenAI (no CLI required) |

## Documentation

- [API Documentation (docs.rs)](https://docs.rs/llm-toolkit)
- [Detailed Guide](./crates/llm-toolkit/docs/)

## License

MIT License - see [LICENSE](LICENSE) for details.
