## Direct API Clients and Type-Safe Models

`llm-toolkit` provides direct HTTP API clients for major LLM providers, allowing you to call APIs without CLI dependencies. Combined with type-safe model identifiers, this enables robust, validated LLM integrations.

## Feature Flags

Enable API clients via Cargo features:

```toml
[dependencies]
# Individual providers
llm-toolkit = { version = "0.59", features = ["anthropic-api"] }
llm-toolkit = { version = "0.59", features = ["gemini-api"] }
llm-toolkit = { version = "0.59", features = ["openai-api"] }
llm-toolkit = { version = "0.59", features = ["ollama-api"] }
llm-toolkit = { version = "0.59", features = ["llama-cpp-server"] }

# All providers
llm-toolkit = { version = "0.59", features = ["all-apis"] }
```

## Type-Safe Model Identifiers

The `models` module provides enum-based model identifiers that prevent typos and ensure only valid model names are used.

### Design Philosophy

- **Type Safety**: Enums prevent invalid model names at compile time
- **Flexibility**: `Custom` variant allows new models without code changes
- **Validation**: Custom models are validated by prefix (e.g., `claude-`, `gemini-`, `gpt-`)
- **Dual Names**: Both API IDs and CLI shorthand names are supported

### ClaudeModel

```rust
use llm_toolkit::models::ClaudeModel;

// Use predefined models
let model = ClaudeModel::Opus46;
assert_eq!(model.as_api_id(), "claude-opus-4-6");
assert_eq!(model.as_cli_name(), "claude-opus-4.6");

// Parse from string (shorthand or full name)
let model: ClaudeModel = "opus".parse().unwrap();    // Opus46
let model: ClaudeModel = "sonnet-4.6".parse().unwrap();

// Custom model (validated - must start with "claude-")
let model: ClaudeModel = "claude-future-model-2027".parse().unwrap();

// Invalid prefix fails
let result: Result<ClaudeModel, _> = "gpt-5".parse();
assert!(result.is_err());
```

**Available Variants:**
| Variant | API ID | CLI Name |
|---------|--------|----------|
| `Opus46` | `claude-opus-4-6` | `claude-opus-4.6` |
| `Sonnet46` (default) | `claude-sonnet-4-6` | `claude-sonnet-4.6` |
| `Haiku45` | `claude-haiku-4-5-20251001` | `claude-haiku-4.5` |
| `Opus45` | `claude-opus-4-5-20251101` | `claude-opus-4.5` |
| `Sonnet45` | `claude-sonnet-4-5-20250929` | `claude-sonnet-4.5` |
| `Opus41` | `claude-opus-4-1-20250805` | `claude-opus-4.1` |
| `Opus4` | `claude-opus-4-20250514` | `claude-opus-4` |
| `Sonnet4` | `claude-sonnet-4-20250514` | `claude-sonnet-4` |

### GeminiModel

```rust
use llm_toolkit::models::GeminiModel;

let model = GeminiModel::Pro31;
assert_eq!(model.as_api_id(), "gemini-3.1-pro-preview");

// Parse shortcuts
let model: GeminiModel = "flash".parse().unwrap();    // Flash25 (default)
let model: GeminiModel = "flash-3".parse().unwrap();   // Flash3
let model: GeminiModel = "pro-3.1".parse().unwrap();   // Pro31
```

**Available Variants:**
| Variant | API ID | CLI Name |
|---------|--------|----------|
| `Pro31` | `gemini-3.1-pro-preview` | `pro-3.1` |
| `Flash3` | `gemini-3-flash-preview` | `flash-3` |
| `Pro3` | `gemini-3-pro-preview` | `pro-3` |
| `Flash25` (default) | `gemini-2.5-flash` | `flash` |
| `Pro25` | `gemini-2.5-pro` | `pro` |
| `FlashLite25` | `gemini-2.5-flash-lite` | `flash-lite` |
| `Flash20` | `gemini-2.0-flash` | `flash-2.0` |

### OpenAIModel

```rust
use llm_toolkit::models::OpenAIModel;

let model = OpenAIModel::Gpt52;
assert_eq!(model.as_api_id(), "gpt-5.2");

// Parse shortcuts
let model: OpenAIModel = "5".parse().unwrap();       // Gpt5 (default)
let model: OpenAIModel = "codex".parse().unwrap();   // Gpt52Codex
let model: OpenAIModel = "o3".parse().unwrap();
```

**Available Variants:**
| Category | Variant | API ID |
|----------|---------|--------|
| GPT-5 Series | `Gpt52`, `Gpt52Pro`, `Gpt51`, `Gpt5` (default), `Gpt5Mini` | `gpt-5.2`, `gpt-5.2-pro`, etc. |
| Codex Series | `Gpt52Codex`, `Gpt51Codex`, `Gpt51CodexMini`, `Gpt5Codex`, `Gpt5CodexMini` | `gpt-5.2-codex`, etc. |
| GPT-4 Series | `Gpt41`, `Gpt41Mini`, `Gpt4o`, `Gpt4oMini` | `gpt-4.1`, `gpt-4o`, etc. |
| O-Series | `O3Pro`, `O3`, `O3Mini`, `O1`, `O1Pro` | `o3-pro`, `o3`, etc. |

### Provider-Agnostic Model

```rust
use llm_toolkit::models::{Model, ClaudeModel, GeminiModel};

// Wrap any provider-specific model
let model: Model = ClaudeModel::Opus45.into();
let model: Model = GeminiModel::Flash3.into();

// Access common interface
println!("{}", model.as_api_id());
```

## Direct API Clients

### AnthropicApiAgent

```rust
use llm_toolkit::agent::impls::AnthropicApiAgent;
use llm_toolkit::agent::Agent;
use llm_toolkit::models::ClaudeModel;

// From environment variable (ANTHROPIC_API_KEY)
let agent = AnthropicApiAgent::try_from_env()?;

// Direct API key with model
let agent = AnthropicApiAgent::new("your-api-key", "claude-sonnet-4-6");

// Using typed model
let agent = AnthropicApiAgent::new("your-api-key", ClaudeModel::Sonnet46.as_api_id())
    .with_claude_model(ClaudeModel::Opus46)  // Override with typed model
    .with_system("You are a helpful assistant")
    .with_max_tokens(4096);

// Execute
let response = agent.execute("Explain Rust ownership".into()).await?;
```

### GeminiApiAgent

```rust
use llm_toolkit::agent::impls::GeminiApiAgent;
use llm_toolkit::agent::Agent;
use llm_toolkit::models::GeminiModel;

// From environment variable (GEMINI_API_KEY)
let agent = GeminiApiAgent::try_from_env()?;

// With Gemini 3.1 thinking capabilities
let agent = GeminiApiAgent::try_gemini_3_from_env(true)?  // enable Google Search
    .with_thinking_level("HIGH");

// Using typed model
let agent = GeminiApiAgent::new("your-api-key", GeminiModel::Pro31.as_api_id())
    .with_gemini_model(GeminiModel::Flash3)
    .with_system_instruction("You are a helpful assistant")
    .with_google_search(true);

let response = agent.execute("What happened today?".into()).await?;
```

**Gemini-specific features:**
- `with_thinking_level("LOW" | "MEDIUM" | "HIGH")` - Enable thinking for Gemini 3+
- `with_google_search(true)` - Enable Google Search tool

### OpenAIApiAgent

```rust
use llm_toolkit::agent::impls::OpenAIApiAgent;
use llm_toolkit::agent::Agent;
use llm_toolkit::models::OpenAIModel;

// From environment variable (OPENAI_API_KEY)
let agent = OpenAIApiAgent::try_from_env()?;

// Using typed model
let agent = OpenAIApiAgent::new("your-api-key", OpenAIModel::Gpt5.as_api_id())
    .with_openai_model(OpenAIModel::Gpt52)
    .with_max_tokens(4096);

let response = agent.execute("Explain quantum computing".into()).await?;
```

### OllamaApiAgent

For local LLM inference with Ollama:

```rust
use llm_toolkit::agent::impls::OllamaApiAgent;
use llm_toolkit::agent::Agent;

// Default configuration (localhost:11434, llama3)
let agent = OllamaApiAgent::new();

// Custom model and endpoint
let agent = OllamaApiAgent::new()
    .with_endpoint("http://192.168.1.100:11434")
    .with_model("qwen2.5-coder:1.5b")
    .with_system_prompt("You are a helpful assistant");

// From environment variables (OLLAMA_HOST, OLLAMA_MODEL)
let agent = OllamaApiAgent::from_env();

// Health check and model listing
if agent.is_healthy().await {
    let models = agent.list_models().await?;
    println!("Available models: {:?}", models);
}

let response = agent.execute("Hello, world!".into()).await?;
```

**Prerequisites:**
1. Install Ollama: https://ollama.ai/download
2. Pull a model: `ollama pull llama3`
3. Start the server: `ollama serve`

### LlamaCppServerAgent

For local LLM inference with llama-server (llama.cpp HTTP server):

```rust
use llm_toolkit::agent::impls::{LlamaCppServerAgent, ChatTemplate};
use llm_toolkit::agent::Agent;

// Default configuration (localhost:8080, Llama3 template)
let agent = LlamaCppServerAgent::new();

// Custom configuration
let agent = LlamaCppServerAgent::new()
    .with_endpoint("http://192.168.1.100:8080")
    .with_chat_template(ChatTemplate::Qwen)
    .with_max_tokens(256)
    .with_temperature(0.7)
    .with_system_prompt("You are a helpful assistant");

// From environment variables
let agent = LlamaCppServerAgent::from_env();

// Health check and slot info
if agent.is_healthy().await {
    let slots = agent.available_slots().await?;
    println!("Available slots: {}", slots);
}

let response = agent.execute("Hello, world!".into()).await?;
```

**Chat Templates:**
- `ChatTemplate::Llama3` - Llama 3 format (default)
- `ChatTemplate::Qwen` - Qwen/Qwen2/Qwen2.5 format (ChatML-based)
- `ChatTemplate::ChatMl` - Generic ChatML format (same as Qwen)
- `ChatTemplate::Mistral` - Mistral/Mixtral format
- `ChatTemplate::Lfm2` - Sakana LFM2 format
- `ChatTemplate::None` - Raw prompt (no template)
- `ChatTemplate::Custom { ... }` - Custom template

> **Note:** `Qwen` and `ChatMl` use identical formatting (`<|im_start|>/<|im_end|>` tokens). Use `Qwen` for Qwen models and `ChatMl` for other ChatML-compatible models.

**Prerequisites:**
1. Build llama.cpp: https://github.com/ggerganov/llama.cpp
2. Download a GGUF model
3. Start the server: `llama-server -m model.gguf --port 8080`

## Environment Variables

| Provider | API Key Variable | Model Variable |
|----------|------------------|----------------|
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| Gemini | `GEMINI_API_KEY` | `GEMINI_MODEL` |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` |
| Ollama | `OLLAMA_HOST` | `OLLAMA_MODEL` |
| llama-server | `LLAMA_SERVER_ENDPOINT` | `LLAMA_SERVER_MAX_TOKENS` |

## Multi-Modal Support

All API clients support multi-modal payloads:

```rust
use llm_toolkit::agent::Payload;
use llm_toolkit::attachment::Attachment;

let payload = Payload::new("What's in this image?")
    .with_attachment(Attachment::from_path("image.png")?);

let response = agent.execute(payload).await?;
```

**Note:** Remote URL attachments are supported by OpenAI but not by Anthropic/Gemini (they require base64-encoded data).

## Error Handling

All clients return `AgentError` with retry information:

```rust
use llm_toolkit::agent::AgentError;

match agent.execute(payload).await {
    Ok(response) => println!("{}", response),
    Err(AgentError::ProcessError {
        status_code,
        is_retryable,
        retry_after,
        message
    }) => {
        if is_retryable {
            if let Some(delay) = retry_after {
                tokio::time::sleep(delay).await;
                // retry...
            }
        }
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

Retryable status codes: `429`, `500`, `502`, `503`, `504`

## CLI Agents vs API Clients

| Aspect | CLI Agents | API Clients |
|--------|-----------|-------------|
| Dependency | Requires CLI installed | HTTP only |
| Features | Full CLI capabilities | Core API features |
| Setup | Install CLI, authenticate | API key only |
| Use case | Development, advanced features | Production, simple integration |

**CLI Agents:** `ClaudeCodeAgent`, `GeminiAgent`, `CodexAgent`
**API Clients:** `AnthropicApiAgent`, `GeminiApiAgent`, `OpenAIApiAgent`, `OllamaApiAgent`, `LlamaCppServerAgent`

## Future Direction

The models module will evolve to support capability-based model selection:

```rust
// Planned API
Model::query()
    .provider(Provider::Any)
    .tier(Tier::Fast)
    .with_capability(Cap::Vision)
    .max_budget_per_1k(0.01)
    .select()
```
