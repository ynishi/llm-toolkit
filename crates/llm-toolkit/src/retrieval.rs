//! Data structures for Retrieval-Augmented Generation (RAG).
//!
//! This module provides core data structures for RAG integration.
//! Retrieval and ingestion logic should be implemented as `Agent`s,
//! not as separate traits, allowing for better composability and
//! integration with the agent ecosystem.
//!
//! # Design Philosophy
//!
//! Instead of defining `Retriever` and `Ingestor` traits, implement
//! retrieval and ingestion as regular agents:
//!
//! ```ignore
//! // Retriever as Agent
//! impl Agent for MyVectorStore {
//!     type Output = Vec<Document>;
//!     async fn execute(&self, payload: Payload) -> Result<Vec<Document>, AgentError> {
//!         let query = payload.to_text();
//!         // Perform retrieval...
//!         Ok(documents)
//!     }
//! }
//!
//! // Ingestor as Agent
//! impl Agent for MyIngestor {
//!     type Output = ();
//!     async fn execute(&self, payload: Payload) -> Result<(), AgentError> {
//!         // Extract documents from payload and ingest...
//!         Ok(())
//!     }
//! }
//! ```
//!
//! Then use `RetrievalAwareAgent` to compose them:
//!
//! ```ignore
//! let retriever = MyVectorStore::new();
//! let base_agent = MyLLMAgent::new();
//! let rag_agent = RetrievalAwareAgent::new(retriever, base_agent);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a piece of retrieved content from a knowledge source.
///
/// This is typically returned by retriever agents (agents with `Output = Vec<Document>`).
/// Documents can be attached to payloads and will be formatted by `PersonaAgent`
/// into a "Retrieved Context" section in the prompt.
///
/// # Examples
///
/// ```rust
/// use llm_toolkit::retrieval::Document;
///
/// let doc = Document {
///     content: "Rust is a systems programming language.".to_string(),
///     source: Some("rust_intro.md".to_string()),
///     score: Some(0.92),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    /// The textual content of the document
    pub content: String,

    /// Optional source identifier (e.g., file path, URL, document ID)
    pub source: Option<String>,

    /// Optional relevance or similarity score (higher = more relevant)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

impl Document {
    /// Creates a new document with the given content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            source: None,
            score: None,
        }
    }

    /// Sets the source identifier for this document.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Sets the relevance score for this document.
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }
}

/// Represents content to be ingested into a knowledge base.
///
/// This is typically consumed by ingestor agents (agents with `Output = ()`).
/// Ingestor agents should extract these from payloads and persist them
/// to the underlying storage (vector database, search engine, etc.).
///
/// # Examples
///
/// ```rust
/// use llm_toolkit::retrieval::IngestibleDocument;
/// use std::collections::HashMap;
///
/// let doc = IngestibleDocument::new(
///     "New content to store",
///     "doc_123",
/// ).with_metadata("author", "Alice");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestibleDocument {
    /// The textual content to be ingested
    pub content: String,

    /// A unique identifier for this content (used for updates/deduplication)
    pub source_id: String,

    /// Optional metadata to be stored alongside the content
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl IngestibleDocument {
    /// Creates a new ingestible document.
    pub fn new(content: impl Into<String>, source_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            source_id: source_id.into(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Sets all metadata at once.
    pub fn with_metadata_map(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}
