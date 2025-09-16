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

## Features

| Feature Area | Description | Key Components | Status |
|---|---|---|---|
| **Content Extraction** | Safely extracting structured data (like JSON) from unstructured LLM responses. | `extract` module (`FlexibleExtractor`, `extract_json`) | Implemented |
| **Prompt Generation** | Building complex and type-safe prompts from Rust data structures. | `prompt` module (`ToPrompt` trait) | Implemented |
| **Intent Extraction** | Extracting structured intents (e.g., enums) from LLM responses. | `intent` module (`IntentExtractor`, `PromptBasedExtractor`) | Implemented |
| **Resilient Deserialization** | Deserializing LLM responses into Rust types, handling schema variations. | (Planned) | Planned |

3.  **Minimal Dependencies:**
    The toolkit will have minimal dependencies (likely only `serde` and `serde_json`) to ensure it can be added to any Rust project with negligible overhead and maximum compatibility.
