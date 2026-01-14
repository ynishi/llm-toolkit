## Features


| Feature Area | Description | Key Components | Status |
|---|---|---|---|
| **Content Extraction** | Safely extracting structured data (like JSON) from unstructured LLM responses. Includes automatic JSON sanitization (trailing commas, unclosed brackets). | `extract` module (`FlexibleExtractor`, `sanitize_json`) | Implemented |
| **Prompt Generation** | Building complex prompts from Rust data structures with a powerful templating engine. Supports dot access for nested types with `#[prompt(as_serialize)]`. | `prompt!` macro, `#[derive(ToPrompt)]`, `#[derive(ToPromptSet)]`, `#[prompt(as_serialize)]` | Implemented |
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
| **Type-Safe Model Identifiers** | Enum-based model identifiers with validation for Claude, Gemini, and OpenAI. Prevents typos, supports Custom variants with prefix validation, and provides both API IDs and CLI names. | `ClaudeModel`, `GeminiModel`, `OpenAIModel`, `Model` enums | Implemented (v0.59.0) |
| **Direct API Clients** | HTTP API clients for LLM providers without CLI dependency. Includes retry support, multi-modal payloads, and provider-specific features (Gemini thinking, Google Search). | `AnthropicApiAgent`, `GeminiApiAgent`, `OpenAIApiAgent` | Implemented (v0.58.0) |
| **Resilient Deserialization** | Fuzzy JSON repair for LLM outputs: syntax sanitization (trailing commas, unclosed brackets/strings) and schema-based typo correction for tagged enums using similarity algorithms (Jaro-Winkler, Levenshtein). | `sanitize_json`, `repair_tagged_enum_json`, `TaggedEnumSchema`, `FuzzyOptions` | Implemented (v0.61.0) |

