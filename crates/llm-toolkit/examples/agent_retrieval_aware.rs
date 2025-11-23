//! Example: RetrievalAwareAgent for RAG (Retrieval-Augmented Generation)
//!
//! This example demonstrates the new Agent-based RAG design where:
//! - Retrievers are implemented as regular `Agent`s with `Output = Vec<Document>`
//! - `RetrievalAwareAgent` composes a retriever and inner agent
//! - No special `Retriever` trait needed - everything is just `Agent`
//!
//! Run this example with:
//! ```sh
//! cargo run --example agent_retrieval_aware --features agent
//! ```

use async_trait::async_trait;
use llm_toolkit::agent::retrieval::RetrievalAwareAgent;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::retrieval::Document;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// VectorStoreAgent: Retriever implemented as an Agent
// ============================================================================

/// A simple in-memory vector store implemented as an Agent.
///
/// This demonstrates how retrievers are just regular agents that return Vec<Document>.
/// In production, this would connect to a real vector database like Pinecone, Weaviate, etc.
#[derive(Clone)]
struct VectorStoreAgent {
    documents: Arc<Mutex<Vec<Document>>>,
}

impl VectorStoreAgent {
    fn new() -> Self {
        Self {
            documents: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Seed the store with initial documents
    async fn seed(&self, docs: Vec<Document>) {
        let mut store = self.documents.lock().await;
        store.extend(docs);
    }
}

#[async_trait]
impl Agent for VectorStoreAgent {
    // KEY: Retriever agents return Vec<Document>
    type Output = Vec<Document>;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Semantic search over programming language documentation";
        &EXPERTISE
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let query = payload.to_text();
        let docs = self.documents.lock().await;

        // Simple substring search (in production: use embeddings + similarity search)
        let mut results: Vec<_> = docs
            .iter()
            .filter(|doc| doc.content.to_lowercase().contains(&query.to_lowercase()))
            .cloned()
            .collect();

        // Sort by score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top 3
        results.truncate(3);

        println!(
            "\n[VectorStoreAgent] Retrieved {} documents for: \"{}\"",
            results.len(),
            query
        );
        for (i, doc) in results.iter().enumerate() {
            println!(
                "  {}. {} (score: {:.2})",
                i + 1,
                doc.source.as_ref().unwrap_or(&"unknown".to_string()),
                doc.score.unwrap_or(0.0)
            );
        }

        Ok(results)
    }
}

// ============================================================================
// MockLLMAgent: Simulates an LLM for demonstration
// ============================================================================

#[derive(Clone)]
struct MockLLMAgent;

#[async_trait]
impl Agent for MockLLMAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Answering questions about programming languages";
        &EXPERTISE
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        let query = payload.to_text();
        let documents = payload.documents();

        println!("\n[MockLLMAgent] Processing query: \"{}\"", query);
        println!(
            "[MockLLMAgent] Received {} documents in context",
            documents.len()
        );

        // Simulate LLM response based on context
        let response = if documents.is_empty() {
            "I don't have enough context to answer this question accurately.".to_string()
        } else {
            // Extract key information from documents
            let context_summary: Vec<_> = documents
                .iter()
                .map(|d| d.content.split('.').next().unwrap_or(&d.content))
                .collect();

            format!(
                "Based on the retrieved documentation:\n\n{}\n\nThese documents provide relevant information to answer your question.",
                context_summary.join("\n- ")
            )
        };

        println!("\n[MockLLMAgent] Generated response:");
        println!("{}", response);

        Ok(response)
    }
}

// ============================================================================
// Main Example
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("RetrievalAwareAgent Example: Agent-based RAG");
    println!("{}\n", "=".repeat(70));

    // ========================================================================
    // Setup: Create and seed the vector store
    // ========================================================================

    println!("Setting up vector store with programming language docs...\n");

    let vector_store = VectorStoreAgent::new();

    vector_store
        .seed(vec![
            Document::new(
                "Rust is a systems programming language focused on safety and performance.",
            )
            .with_source("rust_intro.md")
            .with_score(0.95),
            Document::new(
                "Rust's ownership system ensures memory safety without garbage collection.",
            )
            .with_source("rust_memory.md")
            .with_score(0.92),
            Document::new(
                "Rust supports async programming through futures and the async/await syntax.",
            )
            .with_source("rust_async.md")
            .with_score(0.88),
            Document::new("Python is a high-level interpreted language known for its simplicity.")
                .with_source("python_intro.md")
                .with_score(0.90),
            Document::new("Python has dynamic typing which allows for rapid prototyping.")
                .with_source("python_typing.md")
                .with_score(0.85),
            Document::new("Go is a compiled language designed for building scalable systems.")
                .with_source("go_intro.md")
                .with_score(0.87),
        ])
        .await;

    // ========================================================================
    // Scenario 1: Direct retrieval (without LLM)
    // ========================================================================

    println!("{}", "=".repeat(70));
    println!("Scenario 1: Direct Retrieval");
    println!("{}\n", "=".repeat(70));

    let query1 = Payload::text("async programming");
    let docs = vector_store.execute(query1).await?;

    println!("\nDirect retrieval returned {} documents\n", docs.len());

    // ========================================================================
    // Scenario 2: RAG with RetrievalAwareAgent
    // ========================================================================

    println!("{}", "=".repeat(70));
    println!("Scenario 2: RAG with RetrievalAwareAgent");
    println!("{}\n", "=".repeat(70));

    // Compose: VectorStoreAgent + MockLLMAgent = RAG Agent
    let llm_agent = MockLLMAgent;
    let rag_agent = RetrievalAwareAgent::new(vector_store.clone(), llm_agent);

    let query2 = Payload::text("How does Rust handle memory safety?");
    println!("Query: \"{}\"", query2.to_text());

    let response = rag_agent.execute(query2).await?;

    println!("\n{}", "─".repeat(70));
    println!("Final Answer:");
    println!("{}", "─".repeat(70));
    println!("{}\n", response);

    // ========================================================================
    // Scenario 3: Query with no matching documents
    // ========================================================================

    println!("{}", "=".repeat(70));
    println!("Scenario 3: Query with No Matching Documents");
    println!("{}\n", "=".repeat(70));

    let query3 = Payload::text("What is Java?");
    println!("Query: \"{}\"", query3.to_text());

    let response = rag_agent.execute(query3).await?;

    println!("\n{}", "─".repeat(70));
    println!("Final Answer:");
    println!("{}", "─".repeat(70));
    println!("{}\n", response);

    // ========================================================================
    // Summary
    // ========================================================================

    println!("{}", "=".repeat(70));
    println!("Summary: Key Takeaways");
    println!("{}\n", "=".repeat(70));

    println!("1. Retriever = Regular Agent with Output = Vec<Document>");
    println!("   - No special Retriever trait needed");
    println!("   - Works with all existing Agent infrastructure\n");

    println!("2. RetrievalAwareAgent composes retriever + inner agent");
    println!("   - Automatically retrieves documents");
    println!("   - Augments payload with documents");
    println!("   - Passes augmented payload to inner agent\n");

    println!("3. Clean separation of concerns");
    println!("   - Retrieval logic in VectorStoreAgent");
    println!("   - LLM logic in MockLLMAgent");
    println!("   - Composition in RetrievalAwareAgent\n");

    println!("4. Easy to test and extend");
    println!("   - Mock retrievers are just mock agents");
    println!("   - Can compose with other agent wrappers (history, retry, etc.)");
    println!("   - Type-safe and composable\n");

    println!("{}\n", "=".repeat(70));

    Ok(())
}
