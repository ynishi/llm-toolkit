### Retrieval-Augmented Generation (RAG)

`llm-toolkit` provides first-class support for RAG through composable agents. Instead of separate `Retriever` traits, retrievers are implemented as regular `Agent`s, enabling seamless integration with the existing agent ecosystem.

#### Design Philosophy

- **Retriever = Agent with `Output = Vec<Document>`**
- **No special traits** - works with all existing Agent infrastructure
- **Composable** - combine with other agent wrappers (retry, history, etc.)

```text
┌─────────────────────────────────────┐
│   RetrievalAwareAgent               │
│                                     │
│  ┌──────────┐      ┌──────────┐   │
│  │Retriever │──────│  Inner   │   │
│  │  Agent   │ docs │  Agent   │   │
│  └──────────┘      └──────────┘   │
└─────────────────────────────────────┘
```

#### Document Type

```rust
use llm_toolkit::retrieval::Document;

let doc = Document::new("Rust is a systems programming language.")
    .with_source("rust_intro.md")
    .with_score(0.92);
```

Fields:
- `content: String` - The textual content
- `source: Option<String>` - Source identifier (file path, URL, etc.)
- `score: Option<f32>` - Relevance score (higher = more relevant)

#### Implementing a Retriever Agent

Retrievers are just agents that return `Vec<Document>`:

```rust
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::retrieval::Document;
use async_trait::async_trait;

struct VectorStoreAgent {
    // Your vector database client
}

#[async_trait]
impl Agent for VectorStoreAgent {
    type Output = Vec<Document>;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &str = "Semantic search over documentation";
        &EXPERTISE
    }

    async fn execute(&self, payload: Payload) -> Result<Vec<Document>, AgentError> {
        let query = payload.to_text();

        // Perform semantic search (Pinecone, Weaviate, Qdrant, etc.)
        let results = self.search(query).await?;

        Ok(results)
    }
}
```

#### Composing with RetrievalAwareAgent

`RetrievalAwareAgent` automatically retrieves documents and augments the payload:

```rust
use llm_toolkit::agent::retrieval::RetrievalAwareAgent;
use llm_toolkit::agent::Payload;

// Create retriever and base agent
let retriever = VectorStoreAgent::new();
let llm_agent = MyLLMAgent::new();

// Compose into RAG agent
let rag_agent = RetrievalAwareAgent::new(retriever, llm_agent);

// Execute - retrieval happens automatically
let response = rag_agent.execute(Payload::text("What is Rust?")).await?;
```

**Process:**
1. Execute retriever with input payload → get `Vec<Document>`
2. Augment payload with retrieved documents
3. Execute inner agent with augmented payload
4. Return inner agent's output

#### Accessing Documents in Inner Agent

The inner agent receives documents via `payload.documents()`:

```rust
#[async_trait]
impl Agent for MyLLMAgent {
    type Output = String;
    // ...

    async fn execute(&self, payload: Payload) -> Result<String, AgentError> {
        let query = payload.to_text();
        let documents = payload.documents();  // Retrieved docs

        if documents.is_empty() {
            return Ok("No context available.".to_string());
        }

        // Build context from documents
        let context: String = documents
            .iter()
            .map(|d| format!("- {}", d.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Use context in your prompt
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}",
            context, query
        );

        // Call LLM...
        Ok(response)
    }
}
```

#### Ingestion Pattern

For document ingestion, implement an agent that accepts `Attachment`s:

```rust
use llm_toolkit::attachment::Attachment;
use llm_toolkit::agent::{Agent, Payload};

struct IngestAgent {
    client: VectorDBClient,
}

#[async_trait]
impl Agent for IngestAgent {
    type Output = IngestResult;
    // ...

    async fn execute(&self, payload: Payload) -> Result<IngestResult, AgentError> {
        let attachments = payload.attachments();

        for attachment in attachments {
            // 1. Extract text from file
            // 2. Chunk and embed
            // 3. Store in vector database
        }

        Ok(IngestResult { count: attachments.len() })
    }
}

// Usage
let payload = Payload::attachment(Attachment::local("document.pdf"));
let result = ingest_agent.execute(payload).await?;
```

#### Example: Complete RAG Setup

```rust
use llm_toolkit::agent::retrieval::RetrievalAwareAgent;
use llm_toolkit::agent::{Agent, Payload};
use llm_toolkit::retrieval::Document;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create and seed vector store
    let vector_store = VectorStoreAgent::new();
    vector_store.seed(vec![
        Document::new("Rust has ownership and borrowing.")
            .with_source("rust_memory.md")
            .with_score(0.92),
        Document::new("Rust supports async/await syntax.")
            .with_source("rust_async.md")
            .with_score(0.88),
    ]).await;

    // 2. Create RAG agent
    let llm = ClaudeCodeAgent::new();
    let rag_agent = RetrievalAwareAgent::new(vector_store, llm);

    // 3. Query
    let response = rag_agent
        .execute(Payload::text("How does Rust handle memory?"))
        .await?;

    println!("{}", response);
    Ok(())
}
```

Run the complete example:
```bash
cargo run --example agent_retrieval_aware --features agent
```

#### Benefits

- ✅ **No special traits** - Retriever is just an Agent
- ✅ **Full composability** - Works with RetryAgent, PersonaAgent, etc.
- ✅ **Type-safe** - Compile-time verification
- ✅ **Easy testing** - Mock retrievers are just mock agents
- ✅ **Tracing support** - Automatic instrumentation via `tracing`
