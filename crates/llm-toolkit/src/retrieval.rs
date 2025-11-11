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
//! ## Retrieval Pattern
//!
//! Retrievers return `Vec<Document>` and can be composed with `RetrievalAwareAgent`:
//!
//! ```ignore
//! // Retriever as Agent
//! impl Agent for MyVectorStore {
//!     type Output = Vec<Document>;
//!     async fn execute(&self, payload: Payload) -> Result<Vec<Document>, AgentError> {
//!         let query = payload.to_text();
//!         // Perform semantic search...
//!         Ok(documents)
//!     }
//! }
//!
//! // Compose with RetrievalAwareAgent
//! let retriever = MyVectorStore::new();
//! let base_agent = MyLLMAgent::new();
//! let rag_agent = RetrievalAwareAgent::new(retriever, base_agent);
//! ```
//!
//! ## Ingestion Pattern
//!
//! Ingest agent accept `Attachment`s from payload and handle all implementation details
//! (upload, store creation, metadata management) internally:
//!
//! ```ignore
//! use llm_toolkit::attachment::Attachment;
//!
//! // Gemini Files API style
//! struct GeminiIngestAgent {
//!     client: GeminiClient,
//!     store_name: String,  // Internal state
//! }
//!
//! impl Agent for GeminiIngestAgent {
//!     type Output = IngestResult;  // Can be any type
//!
//!     async fn execute(&self, payload: Payload) -> Result<IngestResult, AgentError> {
//!         let attachments = payload.attachments();
//!         let mut file_names = Vec::new();
//!
//!         for attachment in attachments {
//!             // 1. Upload file
//!             let file = self.client.files.upload(attachment).await?;
//!
//!             // 2. Import into store (internal detail)
//!             self.client.stores.import_file(&self.store_name, &file.name).await?;
//!
//!             file_names.push(file.name);
//!         }
//!
//!         Ok(IngestResult { file_names })
//!     }
//! }
//!
//! // Usage - just pass files
//! let geminiIngestAgent = GeminiIngestAgent::new(client, "my-store");
//! let payload = Payload::attachment(Attachment::local("document.pdf"));
//! let result = geminiIngestAgent.execute(payload).await?;
//! ```
use serde::{Deserialize, Serialize};

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
