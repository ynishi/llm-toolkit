//! Retrieval-aware agent wrapper for RAG (Retrieval-Augmented Generation).
//!
//! This module provides `RetrievalAwareAgent`, which wraps any agent and
//! automatically augments its input with retrieved documents.

use super::{Agent, AgentError, Payload};
use crate::retrieval::Document;
use async_trait::async_trait;

/// An agent wrapper that automatically retrieves relevant documents
/// and augments the payload before passing it to the inner agent.
///
/// This agent follows the same pattern as `HistoryAwareAgent`, composing
/// two agents: a retriever agent and an inner agent. The retriever agent
/// must return `Vec<Document>` as its output.
///
/// # Design
///
/// ```text
/// ┌─────────────────────────────────────┐
/// │   RetrievalAwareAgent               │
/// │                                     │
/// │  ┌──────────┐      ┌──────────┐   │
/// │  │Retriever │──────│  Inner   │   │
/// │  │  Agent   │ docs │  Agent   │   │
/// │  └──────────┘      └──────────┘   │
/// └─────────────────────────────────────┘
/// ```
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::agent::retrieval::RetrievalAwareAgent;
/// use llm_toolkit::agent::{Agent, Payload};
///
/// // Create a retriever agent (returns Vec<Document>)
/// let retriever = VectorStoreAgent::new();
///
/// // Create your main agent
/// let base_agent = MyLLMAgent::new();
///
/// // Wrap with retrieval capability
/// let rag_agent = RetrievalAwareAgent::new(retriever, base_agent);
///
/// // Execute - retrieval happens automatically
/// let response = rag_agent.execute(Payload::text("What is Rust?")).await?;
/// ```
pub struct RetrievalAwareAgent<R, I>
where
    R: Agent<Output = Vec<Document>>,
    I: Agent,
{
    /// The retriever agent that fetches relevant documents
    retriever: R,

    /// The inner agent that processes the augmented payload
    inner_agent: I,
}

impl<R, I> RetrievalAwareAgent<R, I>
where
    R: Agent<Output = Vec<Document>>,
    I: Agent,
{
    /// Creates a new retrieval-aware agent.
    ///
    /// # Arguments
    ///
    /// * `retriever` - An agent that returns `Vec<Document>` when executed
    /// * `inner_agent` - The agent to wrap with retrieval capabilities
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let rag_agent = RetrievalAwareAgent::new(
    ///     VectorStoreAgent::new(),
    ///     MyLLMAgent::new(),
    /// );
    /// ```
    pub fn new(retriever: R, inner_agent: I) -> Self {
        Self {
            retriever,
            inner_agent,
        }
    }

    /// Returns a reference to the retriever agent.
    pub fn retriever(&self) -> &R {
        &self.retriever
    }

    /// Returns a reference to the inner agent.
    pub fn inner_agent(&self) -> &I {
        &self.inner_agent
    }
}

#[async_trait]
impl<R, I> Agent for RetrievalAwareAgent<R, I>
where
    R: Agent<Output = Vec<Document>> + Send + Sync,
    I: Agent + Send + Sync,
    I::Output: Send,
{
    type Output = I::Output;

    fn expertise(&self) -> &str {
        // Inherit expertise from the inner agent
        self.inner_agent.expertise()
    }

    fn capabilities(&self) -> Option<Vec<super::Capability>> {
        // Inherit capabilities from the inner agent
        self.inner_agent.capabilities()
    }

    /// Executes the agent with automatic document retrieval.
    ///
    /// # Process
    ///
    /// 1. Execute the retriever agent with the input payload to get relevant documents
    /// 2. Augment the payload with the retrieved documents
    /// 3. Execute the inner agent with the augmented payload
    /// 4. Return the inner agent's output
    ///
    /// # Error Handling
    ///
    /// If the retriever fails, the error is propagated immediately.
    /// If the inner agent fails, its error is propagated.
    #[crate::tracing::instrument(
        name = "retrieval_aware_agent.execute",
        skip(self, payload),
        fields(
            retriever.expertise = self.retriever.expertise(),
            inner_agent.expertise = self.inner_agent.expertise(),
        )
    )]
    async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
        // Step 1: Retrieve relevant documents
        crate::tracing::debug!(
            target: "llm_toolkit::agent::retrieval",
            "Executing retriever agent"
        );

        let documents = self.retriever.execute(payload.clone()).await?;

        crate::tracing::debug!(
            target: "llm_toolkit::agent::retrieval",
            document_count = documents.len(),
            "Retrieved documents from retriever agent"
        );

        // Step 2: Augment payload with documents
        let augmented_payload = payload.with_documents(documents);

        crate::tracing::trace!(
            target: "llm_toolkit::agent::retrieval",
            "Augmented payload with retrieved documents"
        );

        // Step 3: Execute inner agent with augmented payload
        crate::tracing::debug!(
            target: "llm_toolkit::agent::retrieval",
            "Executing inner agent with augmented payload"
        );

        self.inner_agent.execute(augmented_payload).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentError, Payload};
    use crate::retrieval::Document;
    use async_trait::async_trait;
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Mock retriever agent for testing
    #[derive(Clone)]
    struct MockRetriever {
        documents: Vec<Document>,
        calls: Arc<Mutex<Vec<Payload>>>,
    }

    impl MockRetriever {
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
    impl Agent for MockRetriever {
        type Output = Vec<Document>;

        fn expertise(&self) -> &str {
            "Mock retriever for testing"
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
            "Mock inner agent for testing"
        }

        async fn execute(&self, payload: Payload) -> Result<Self::Output, AgentError> {
            self.calls.lock().await.push(payload);
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_retrieval_aware_agent_augments_payload() {
        // Setup: Create mock retriever with documents
        let documents = vec![
            Document::new("Rust is a systems programming language.")
                .with_source("rust_intro.md")
                .with_score(0.92),
            Document::new("Rust has ownership and borrowing.")
                .with_source("rust_memory.md")
                .with_score(0.88),
        ];

        let retriever = MockRetriever::new(documents.clone());
        let inner_agent = MockInnerAgent::new("Response".to_string());

        // Create RetrievalAwareAgent
        let rag_agent = RetrievalAwareAgent::new(retriever.clone(), inner_agent.clone());

        // Execute
        let payload = Payload::text("What is Rust?");
        let result = rag_agent.execute(payload.clone()).await.unwrap();

        // Verify response
        assert_eq!(result, "Response");

        // Verify retriever was called
        let retriever_calls = retriever.get_calls().await;
        assert_eq!(retriever_calls.len(), 1);
        assert_eq!(retriever_calls[0].to_text(), "What is Rust?");

        // Verify inner agent received augmented payload
        let inner_calls = inner_agent.get_calls().await;
        assert_eq!(inner_calls.len(), 1);

        let received_docs = inner_calls[0].documents();
        assert_eq!(received_docs.len(), 2);
        assert_eq!(received_docs[0].content, "Rust is a systems programming language.");
        assert_eq!(received_docs[1].content, "Rust has ownership and borrowing.");
    }

    #[tokio::test]
    async fn test_retrieval_aware_agent_propagates_retriever_error() {
        // Setup: Create a retriever that always fails
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

        // Verify inner agent was never called
        let inner_calls = inner_agent.get_calls().await;
        assert_eq!(inner_calls.len(), 0);
    }

    #[tokio::test]
    async fn test_retrieval_aware_agent_with_empty_results() {
        // Setup: Retriever that returns no documents
        let retriever = MockRetriever::new(vec![]);
        let inner_agent = MockInnerAgent::new("No context".to_string());
        let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent.clone());

        // Execute
        let result = rag_agent.execute(Payload::text("Query")).await.unwrap();
        assert_eq!(result, "No context");

        // Verify inner agent still received payload (just with no documents)
        let inner_calls = inner_agent.get_calls().await;
        assert_eq!(inner_calls.len(), 1);
        assert_eq!(inner_calls[0].documents().len(), 0);
    }

    #[tokio::test]
    async fn test_expertise_delegation() {
        let retriever = MockRetriever::new(vec![]);
        let inner_agent = MockInnerAgent::new("Response".to_string());
        let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent);

        // Expertise should be inherited from inner agent
        assert_eq!(rag_agent.expertise(), "Mock inner agent for testing");
    }

    #[tokio::test]
    async fn test_retrieval_aware_agent_preserves_attachments() {
        use crate::attachment::Attachment;

        let retriever = MockRetriever::new(vec![Document::new("Doc content")]);
        let inner_agent = MockInnerAgent::new("ok".to_string());
        let rag_agent = RetrievalAwareAgent::new(retriever, inner_agent.clone());

        // Execute with attachment
        let attachment = Attachment::in_memory(vec![1, 2, 3]);
        let payload = Payload::text("Query").with_attachment(attachment.clone());

        let _ = rag_agent.execute(payload).await.unwrap();

        // Verify attachment was preserved
        let inner_calls = inner_agent.get_calls().await;
        assert!(inner_calls[0].has_attachments());
        assert_eq!(inner_calls[0].attachments().len(), 1);
        assert_eq!(inner_calls[0].attachments()[0], &attachment);
    }
}
