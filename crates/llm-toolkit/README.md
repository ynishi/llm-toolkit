# llm-toolkit
Basic llm tools for rust

# Motivation & Philosophy

High-level LLM frameworks like LangChain, while powerful, can be problematic in Rust. Their heavy abstractions and complex type systems often conflict with Rust's strengths, imposing significant constraints and learning curves on developers.

There is a clear need for a different kind of tool: a low-level, unopinionated, and minimalist toolkit that provides robust "last mile" utilities for LLM integration, much like how `candle` provides core building blocks for ML without dictating the entire application architecture.

This document proposes the creation of `llm-toolkit`, a new library crate designed to be the professional's choice for building reliable, high-performance LLM-powered applications in Rust.

## Documentation

Detailed documentation:

- [Core Design Principles](./docs/01-core-design-principles.md)
- [Features](./docs/02-features.md)
- [Prompt Generation](./docs/03-prompt-generation.md)
- [Intent Extraction With Intentframe](./docs/04-intent-extraction-with-intentframe.md)
- [Type Safe Intents With Defineintent](./docs/05-type-safe-intents-with-defineintent.md)
- [Agent Api And Multi Agent Orchestration/00 Index](./docs/06-agent-api-and-multi-agent-orchestration/00-index.md)
- [Observability](./docs/07-observability.md)
- [Future Directions](./docs/08-future-directions.md)
