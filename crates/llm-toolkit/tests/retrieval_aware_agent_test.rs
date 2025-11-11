//! Integration tests for RetrievalAwareAgent.
//!
//! These tests verify that the new Agent-based RAG design works correctly.

#![cfg(feature = "agent")]

use async_trait::async_trait;
use llm_toolkit::agent::retrieval::RetrievalAwareAgent;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::retrieval::Document;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Mock retriever agent that returns predefined documents
#[derive(Clone)]
struct MockRetrieverAgent {
    documents: Vec<Document>,
    calls: Arc<Mutex<Vec<Payload>>>,
}

impl MockRetrieverAgent {
    fn new(documents: Vec<Document>) -> Self {
        Self {
            documents,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn get_calls(&self) -> Vec<Payload> {
        self.calls.lock().await.clone()
    }
}

#[async_trait]
impl Agent for MockRetrieverAgent {
    type Output = Vec<Document>;

    fn expertise(&self) -> &str {
        "Mock document retriever"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        self.calls.lock().await.push(payload);
        Ok(self.documents.clone())
    }
}

/// Mock inner agent for testing
#[derive(Clone)]
struct MockInnerAgent<T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static> {
    response: T,
    calls: Arc<Mutex<Vec<Payload>>>,
}

impl<T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static> MockInnerAgent<T> {
    fn new(response: T) -> Self {
        Self {
            response,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn get_calls(&self) -> Vec<Payload> {
        self.calls.lock().await.clone()
    }
}

#[async_trait]
impl<T> Agent for MockInnerAgent<T>
where
    T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    type Output = T;

    fn expertise(&self) -> &str {
        "Mock inner agent"
    }

    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        self.calls.lock().await.push(payload);
        Ok(self.response.clone())
    }
}

#[tokio::test]
async fn test_retrieval_aware_agent_basic_flow() {
    // Setup: Create retriever and inner agent
    let documents = vec![
        Document::new("Rust is a systems programming language.")
            .with_source("rust_intro.md")
            .with_score(0.92),
        Document::new("Rust has ownership and borrowing.")
            .with_source("rust_memory.md")
            .with_score(0.88),
    ];

    let retriever = MockRetrieverAgent::new(documents.clone());
    let inner_agent = MockInnerAgent::new("Generated response".to_string());

    // Create RetrievalAwareAgent
    let rag_agent = RetrievalAwareAgent::new(retriever.clone(), inner_agent.clone());

    // Execute
    let payload = Payload::text("What is Rust?");
    let result = rag_agent.execute(payload.clone()).await.unwrap();

    // Verify response
    assert_eq!(result, "Generated response");

    // Verify retriever was called
    let retriever_calls = retriever.get_calls().await;
    assert_eq!(retriever_calls.len(), 1);
    assert_eq!(retriever_calls[0].to_text(), "What is Rust?");

    // Verify inner agent received augmented payload with documents
    let inner_calls = inner_agent.get_calls().await;
    assert_eq!(inner_calls.len(), 1);

    let received_docs = inner_calls[0].documents();
    assert_eq!(received_docs.len(), 2);
    assert_eq!(
        received_docs[0].content,
        "Rust is a systems programming language."
    );
    assert_eq!(
        received_docs[1].content,
        "Rust has ownership and borrowing."
    );
}

#[tokio::test]
async fn test_retrieval_aware_agent_with_empty_results() {
    // Retriever that returns no documents
    let retriever = MockRetrieverAgent::new(vec![]);
    let inner_agent = MockInnerAgent::new("No context available".to_string());

    let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent.clone());

    // Execute
    let result = rag_agent.execute(Payload::text("Query")).await.unwrap();
    assert_eq!(result, "No context available");

    // Inner agent should still receive payload (just with no documents)
    let inner_calls = inner_agent.get_calls().await;
    assert_eq!(inner_calls.len(), 1);
    assert_eq!(inner_calls[0].documents().len(), 0);
}

#[tokio::test]
async fn test_retrieval_aware_agent_propagates_retriever_error() {
    // Create a failing retriever
    #[derive(Clone)]
    struct FailingRetriever;

    #[async_trait]
    impl Agent for FailingRetriever {
        type Output = Vec<Document>;

        fn expertise(&self) -> &str {
            "Failing retriever"
        }

        async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
            Err(AgentError::ExecutionFailed("Retrieval failed".to_string()))
        }
    }

    let retriever = FailingRetriever;
    let inner_agent = MockInnerAgent::new("Should not be reached".to_string());
    let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent.clone());

    // Execute and verify error is propagated
    let result = rag_agent.execute(Payload::text("Query")).await;
    assert!(result.is_err());

    // Inner agent should never be called
    let inner_calls = inner_agent.get_calls().await;
    assert_eq!(inner_calls.len(), 0);
}

#[tokio::test]
async fn test_retrieval_aware_agent_preserves_attachments() {
    use llm_toolkit::attachment::Attachment;

    let retriever = MockRetrieverAgent::new(vec![Document::new("Doc content")]);
    let inner_agent = MockInnerAgent::new("ok".to_string());
    let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent.clone());

    // Execute with attachment
    let attachment = Attachment::in_memory(vec![1, 2, 3]);
    let payload = Payload::text("Query").with_attachment(attachment.clone());

    let _ = rag_agent.execute(payload).await.unwrap();

    // Verify attachment was preserved in inner agent's payload
    let inner_calls = inner_agent.get_calls().await;
    assert!(inner_calls[0].has_attachments());
    assert_eq!(inner_calls[0].attachments().len(), 1);
    assert_eq!(inner_calls[0].attachments()[0], &attachment);

    // Verify document was also added
    assert_eq!(inner_calls[0].documents().len(), 1);
}

#[tokio::test]
async fn test_retrieval_aware_agent_expertise_delegation() {
    let retriever = MockRetrieverAgent::new(vec![]);
    let inner_agent = MockInnerAgent::new("Response".to_string());
    let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent);

    // Expertise should be inherited from inner agent
    assert_eq!(rag_agent.expertise(), "Mock inner agent");
}

#[tokio::test]
async fn test_document_builder_pattern() {
    // Test the builder pattern for Document
    let doc = Document::new("Content")
        .with_source("source.txt")
        .with_score(0.95);

    assert_eq!(doc.content, "Content");
    assert_eq!(doc.source, Some("source.txt".to_string()));
    assert_eq!(doc.score, Some(0.95));
}

#[tokio::test]
async fn test_payload_with_documents() {
    // Test that Payload correctly handles documents
    let doc1 = Document::new("Doc 1").with_score(0.9);
    let doc2 = Document::new("Doc 2").with_score(0.8);

    let payload = Payload::text("Query")
        .with_document(doc1.clone())
        .with_documents(vec![doc2.clone()]);

    let docs = payload.documents();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].content, "Doc 1");
    assert_eq!(docs[1].content, "Doc 2");
}
