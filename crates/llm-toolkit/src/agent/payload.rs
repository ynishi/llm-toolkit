//! Payload types for multi-modal agent communication.
//!
//! This module provides abstractions for passing different types of content
//! (text, images, etc.) to agents in a unified way.

use super::dialogue::message::MessageMetadata;
use super::payload_message::PayloadMessage;
use crate::attachment::Attachment;
use crate::retrieval::Document;
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "agent")]
use super::expertise::RenderContext;

#[cfg(feature = "agent")]
use super::execution_context::ExecutionContext;

#[cfg(feature = "agent")]
use super::detected_context::DetectedContext;

/// Content that can be included in a payload.
///
/// This enum allows agents to receive different types of input,
/// including text, attachments, and potentially other media types in the future.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadContent {
    /// Plain text content (treated as User input in dialogue context)
    Text(String),

    /// Arbitrary file attachment (local, remote, or in-memory)
    Attachment(Attachment),

    /// Dialogue message with explicit speaker information
    Message {
        speaker: crate::agent::dialogue::Speaker,
        content: String,
        #[allow(dead_code)]
        metadata: MessageMetadata,
    },

    /// Dialogue participants information
    ///
    /// Contains metadata about all participants in a dialogue, including their
    /// names, roles, and background information. This allows agents to understand
    /// the context of who they're interacting with.
    Participants(Vec<crate::agent::dialogue::ParticipantInfo>),

    /// A retrieved document from a knowledge source (RAG)
    ///
    /// Documents are typically added by retriever agents and formatted
    /// by PersonaAgent into a "Retrieved Context" section.
    Document(Document),

    /// Context information (e.g., DialogueContext, environment info)
    ///
    /// This is used for information that should remain visible even in long
    /// conversations without being buried in history. PersonaAgent handles
    /// strategic placement of this context based on conversation length.
    Context(String),
}

/// Inner payload data, wrapped in Arc for efficient cloning.
#[derive(Debug, Clone, PartialEq)]
struct PayloadInner {
    contents: Vec<PayloadContent>,

    /// Runtime context for expertise rendering (not serialized)
    ///
    /// This is used by ExpertiseAgent to determine which knowledge fragments
    /// should be included and how they should be prioritized during prompt generation.
    ///
    /// Separate from `PayloadContent::Context` which is for LLM-visible natural language context.
    #[cfg(feature = "agent")]
    render_context: Option<RenderContext>,

    /// Raw execution context from orchestrator layer (not serialized)
    ///
    /// Contains factual runtime information like step info, journal summary,
    /// and redesign count. Used by context detectors to infer higher-level context.
    #[cfg(feature = "agent")]
    execution_context: Option<ExecutionContext>,

    /// Detected context from analysis layers (not serialized)
    ///
    /// Contains inferred information like task_type, task_health, and user_states.
    /// Progressively enriched by multiple detector layers (rule-based → LLM-based).
    #[cfg(feature = "agent")]
    detected_context: Option<DetectedContext>,
}

/// A multi-modal payload that can contain multiple content items.
///
/// This structure uses `Arc` internally to make cloning efficient,
/// which is especially important when retrying agent operations.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::payload::{Payload, PayloadContent};
/// use std::path::PathBuf;
///
/// // Simple text payload
/// let payload: Payload = "Analyze this text".to_string().into();
///
/// // Multi-modal payload with text and attachment
/// let payload = Payload::text("What's in this image?")
///     .with_attachment(Attachment::local("/path/to/image.png"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Payload {
    /// The contents of this payload, wrapped in Arc for efficient cloning
    inner: Arc<PayloadInner>,
}

impl Payload {
    /// Creates a new empty payload.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(PayloadInner {
                contents: Vec::new(),
                #[cfg(feature = "agent")]
                render_context: None,
                #[cfg(feature = "agent")]
                execution_context: None,
                #[cfg(feature = "agent")]
                detected_context: None,
            }),
        }
    }

    /// Creates a payload with a single text content.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(PayloadInner {
                contents: vec![PayloadContent::Text(text.into())],
                #[cfg(feature = "agent")]
                render_context: None,
                #[cfg(feature = "agent")]
                execution_context: None,
                #[cfg(feature = "agent")]
                detected_context: None,
            }),
        }
    }

    /// Creates a payload with a single attachment.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let attachment = Attachment::in_memory(vec![1, 2, 3]);
    /// let payload = Payload::attachment(attachment);
    /// ```
    pub fn attachment(attachment: Attachment) -> Self {
        Self {
            inner: Arc::new(PayloadInner {
                contents: vec![PayloadContent::Attachment(attachment)],
                #[cfg(feature = "agent")]
                render_context: None,
                #[cfg(feature = "agent")]
                execution_context: None,
                #[cfg(feature = "agent")]
                detected_context: None,
            }),
        }
    }

    /// Creates a payload from any type that implements ToPrompt.
    ///
    /// This is a convenience method that converts structured data (DTOs, structs, etc.)
    /// directly into a Payload without needing to call `.to_prompt()` explicitly.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Serialize, Deserialize, ToPrompt)]
    /// struct AnalysisRequest {
    ///     code: String,
    ///     language: String,
    /// }
    ///
    /// let request = AnalysisRequest {
    ///     code: "fn main() {}".to_string(),
    ///     language: "rust".to_string(),
    /// };
    ///
    /// // Before: request.to_prompt().into()
    /// // After:  Payload::from_prompt(request)
    /// let payload = Payload::from_prompt(request);
    /// ```
    pub fn from_prompt<T: crate::prompt::ToPrompt>(value: T) -> Self {
        Self::text(value.to_prompt())
    }

    /// Returns a reference to the contents of this payload.
    pub fn contents(&self) -> &[PayloadContent] {
        &self.inner.contents
    }

    /// Returns the count of total content's text length in this payload.
    pub fn total_content_count(&self) -> usize {
        self.inner.contents.iter().fold(0, |acc, content| {
            match content {
                PayloadContent::Text(text) => acc + text.len(),
                PayloadContent::Message { content, .. } => acc + content.len(),
                PayloadContent::Attachment(_) => acc + 1, // Count each attachment as 1
                PayloadContent::Document(doc) => acc + doc.content.len(),
                PayloadContent::Participants(participants) => acc + participants.len(),
                PayloadContent::Context(ctx) => acc + ctx.len(),
            }
        })
    }

    /// Helper: Creates a new PayloadInner from contents while preserving all contexts
    fn create_inner(&self, contents: Vec<PayloadContent>) -> PayloadInner {
        PayloadInner {
            contents,
            #[cfg(feature = "agent")]
            render_context: self.inner.render_context.clone(),
            #[cfg(feature = "agent")]
            execution_context: self.inner.execution_context.clone(),
            #[cfg(feature = "agent")]
            detected_context: self.inner.detected_context.clone(),
        }
    }

    /// Adds text content to this payload.
    pub fn with_text(self, text: impl Into<String>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Text(text.into()));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Set text content to this payload, replacing existing text contents.
    pub fn set_text(self, text: impl Into<String>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.retain(|c| !matches!(c, PayloadContent::Text(_)));
        new_contents.push(PayloadContent::Text(text.into()));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Adds an attachment to this payload.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let payload = Payload::text("Question")
    ///     .with_attachment(Attachment::in_memory(vec![1, 2, 3]));
    /// ```
    pub fn with_attachment(self, attachment: Attachment) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Attachment(attachment));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Adds a retrieved document to this payload.
    ///
    /// Documents are typically added by retriever agents and will be
    /// formatted by PersonaAgent into a "Retrieved Context" section.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::{Document, Payload};
    ///
    /// let doc = Document::new("Rust is a systems programming language.")
    ///     .with_source("rust_intro.md")
    ///     .with_score(0.92);
    ///
    /// let payload = Payload::text("What is Rust?")
    ///     .with_document(doc);
    /// ```
    pub fn with_document(self, document: Document) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Document(document));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Adds multiple retrieved documents to this payload.
    ///
    /// This is a convenience method for adding multiple documents at once,
    /// typically used by RetrievalAwareAgent after retrieval.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::{Document, Payload};
    ///
    /// let docs = vec![
    ///     Document::new("Content 1").with_score(0.9),
    ///     Document::new("Content 2").with_score(0.8),
    /// ];
    ///
    /// let payload = Payload::text("Query")
    ///     .with_documents(docs);
    /// ```
    pub fn with_documents(self, documents: impl IntoIterator<Item = Document>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.extend(documents.into_iter().map(PayloadContent::Document));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Prepends text content to the beginning of this payload.
    ///
    /// This is useful for adding system instructions or context
    /// before the user's original content.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("User question")
    ///     .prepend_text("System instructions: ");
    ///
    /// assert_eq!(payload.to_text(), "System instructions: \nUser question");
    /// ```
    pub fn prepend_text(self, text: impl Into<String>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.insert(0, PayloadContent::Text(text.into()));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Prepends a dialogue message to the beginning of this payload.
    ///
    /// This is useful for adding system instructions or context messages
    /// with explicit speaker attribution before the existing content.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::dialogue::Speaker;
    ///
    /// let payload = Payload::text("User question")
    ///     .prepend_message(Speaker::System, "Keep responses concise.");
    /// ```
    pub fn prepend_message(
        self,
        speaker: crate::agent::dialogue::Speaker,
        content: impl Into<String>,
    ) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.insert(
            0,
            PayloadContent::Message {
                speaker,
                content: content.into(),
                metadata: MessageMetadata::default(),
            },
        );
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Prepends a dialogue messages to the beginning of this payload.
    ///
    /// This is useful for adding system instructions or context messages
    /// with explicit speaker attribution before the existing content.
    pub fn prepend_messages(self, messages: Vec<PayloadContent>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        let mut new_messages = messages
            .into_iter()
            .filter(|p| matches!(p, PayloadContent::Message { .. }))
            .collect::<Vec<PayloadContent>>();
        new_messages.append(&mut new_contents);
        Self {
            inner: Arc::new(self.create_inner(new_messages)),
        }
    }

    /// Adds a dialogue message with metadata to this payload.
    ///
    /// This allows fine-grained control over message reaction behavior.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::dialogue::{Speaker, MessageMetadata, MessageType};
    ///
    /// let payload = Payload::new()
    ///     .add_message_with_metadata(
    ///         Speaker::System,
    ///         "Command completed successfully",
    ///         MessageMetadata::new().with_type(MessageType::CommandResult),
    ///     );
    /// ```
    pub fn add_message_with_metadata(
        self,
        speaker: crate::agent::dialogue::Speaker,
        content: impl Into<String>,
        metadata: MessageMetadata,
    ) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Message {
            speaker,
            content: content.into(),
            metadata,
        });
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Prepends a system message to the beginning of this payload.
    ///
    /// This is a convenience method equivalent to `prepend_message(Speaker::System, content)`.
    /// Useful for adding system-level instructions or constraints.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("User question")
    ///     .prepend_system("IMPORTANT: Keep responses under 300 characters.");
    /// ```
    pub fn prepend_system(self, instruction: impl Into<String>) -> Self {
        self.prepend_message(crate::agent::dialogue::Speaker::System, instruction)
    }

    /// Merges another payload's contents into this one.
    ///
    /// This appends all content items from the other payload to the end of this payload's
    /// contents. This is useful for combining context from multiple sources.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let context = Payload::text("Previous conversation context");
    /// let new_input = Payload::text("New question")
    ///     .with_attachment(Attachment::local("image.png"));
    ///
    /// let combined = context.merge(new_input);
    /// // Contains: "Previous conversation context", "New question", and the attachment
    /// ```
    pub fn merge(self, other: Payload) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.extend(other.contents().iter().cloned());
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Adds a dialogue message to this payload.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::dialogue::Speaker;
    ///
    /// let payload = Payload::new()
    ///     .with_message(Speaker::System, "You are an AI assistant")
    ///     .with_message(Speaker::user("Alice"), "Hello!");
    /// ```
    pub fn with_message(
        self,
        speaker: crate::agent::dialogue::Speaker,
        content: impl Into<String>,
    ) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Message {
            speaker,
            content: content.into(),
            metadata: MessageMetadata::default(),
        });
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Creates a payload from multiple dialogue messages.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::dialogue::Speaker;
    ///
    /// let payload = Payload::from_messages(vec![
    ///     PayloadMessage::system("You are a helpful assistant"),
    ///     PayloadMessage::user("user1", "User", "What is Rust?"),
    /// ]);
    /// ```
    pub fn from_messages(messages: Vec<PayloadMessage>) -> Self {
        let contents = messages
            .into_iter()
            .map(|message| PayloadContent::Message {
                speaker: message.speaker,
                content: message.content,
                metadata: message.metadata,
            })
            .collect();

        Self {
            inner: Arc::new(PayloadInner {
                contents,
                #[cfg(feature = "agent")]
                render_context: None,
                #[cfg(feature = "agent")]
                execution_context: None,
                #[cfg(feature = "agent")]
                detected_context: None,
            }),
        }
    }

    /// Adds participants information to this payload.
    ///
    /// This provides context about all participants in a dialogue, which can be
    /// used by agents to understand the conversation context and relationships.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::dialogue::ParticipantInfo;
    ///
    /// let participants = vec![
    ///     ParticipantInfo::new("Alice".to_string(), "PM".to_string(), "Product manager".to_string()),
    ///     ParticipantInfo::new("Bob".to_string(), "Engineer".to_string(), "Backend developer".to_string()),
    /// ];
    ///
    /// let payload = Payload::text("Current task")
    ///     .with_participants(participants);
    /// ```
    pub fn with_participants(
        self,
        participants: Vec<crate::agent::dialogue::ParticipantInfo>,
    ) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Participants(participants));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Returns the participants information if present.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("Task")
    ///     .with_participants(participants);
    ///
    /// if let Some(participants) = payload.participants() {
    ///     println!("Found {} participants", participants.len());
    /// }
    /// ```
    pub fn participants(&self) -> Option<&Vec<crate::agent::dialogue::ParticipantInfo>> {
        self.inner.contents.iter().find_map(|c| match c {
            PayloadContent::Participants(p) => Some(p),
            _ => None,
        })
    }

    /// Returns all text contents concatenated with newlines.
    ///
    /// This is useful for agents that only support text input.
    /// Note: This only returns Text variants, not Message or Context variants.
    /// For Message-aware processing, use `to_messages()` instead.
    pub fn to_text(&self) -> String {
        self.inner
            .contents
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns all structured messages (both Text and Message variants) as `PayloadMessage`.
    ///
    /// This preserves the structure of dialogue messages with speaker information.
    /// - `Text` variants are converted to `PayloadMessage::system`
    /// - `Message` variants preserve their original speaker and content
    /// - `Attachment` variants are not included
    ///
    /// This is useful for agents that need to understand the message structure
    /// and maintain proper conversation history.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::dialogue::Speaker;
    ///
    /// let payload = Payload::from_messages(vec![
    ///     (Speaker::System, "System instruction".to_string()),
    ///     (Speaker::user("Alice", "PM"), "What should we build?".to_string()),
    /// ]);
    ///
    /// let messages = payload.to_messages();
    /// assert_eq!(messages.len(), 2);
    /// assert_eq!(messages[0].speaker, Speaker::System);
    /// assert_eq!(messages[1].speaker, Speaker::user("Alice", "PM"));
    /// ```
    pub fn to_messages(&self) -> Vec<PayloadMessage> {
        self.inner
            .contents
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Message {
                    speaker,
                    content,
                    metadata,
                } => Some(PayloadMessage {
                    speaker: speaker.clone(),
                    content: content.clone(),
                    metadata: metadata.clone(),
                }),
                PayloadContent::Text(_)
                | PayloadContent::Attachment(_)
                | PayloadContent::Document(_)
                | PayloadContent::Participants(_)
                | PayloadContent::Context(_) => None,
            })
            .collect()
    }

    /// Returns true if this payload contains only text.
    pub fn is_text_only(&self) -> bool {
        self.inner
            .contents
            .iter()
            .all(|c| matches!(c, PayloadContent::Text(_)))
    }

    /// Returns true if this payload contains any attachments.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let payload = Payload::text("Text")
    ///     .with_attachment(Attachment::in_memory(vec![1, 2, 3]));
    /// assert!(payload.has_attachments());
    /// ```
    pub fn has_attachments(&self) -> bool {
        self.inner
            .contents
            .iter()
            .any(|c| matches!(c, PayloadContent::Attachment(_)))
    }

    /// Returns a vector of references to all attachments in this payload.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let payload = Payload::text("Text")
    ///     .with_attachment(Attachment::in_memory(vec![1, 2, 3]));
    /// let attachments = payload.attachments();
    /// assert_eq!(attachments.len(), 1);
    /// ```
    pub fn attachments(&self) -> Vec<&Attachment> {
        self.inner
            .contents
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Attachment(a) => Some(a),
                _ => None,
            })
            .collect()
    }

    /// Returns a vector of references to all documents in this payload.
    ///
    /// Documents are typically added by retriever agents for RAG use cases.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::{Document, Payload};
    ///
    /// let doc = Document::new("Content").with_score(0.9);
    /// let payload = Payload::text("Query").with_document(doc);
    ///
    /// let documents = payload.documents();
    /// assert_eq!(documents.len(), 1);
    /// ```
    pub fn documents(&self) -> Vec<&Document> {
        self.inner
            .contents
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Document(d) => Some(d),
                _ => None,
            })
            .collect()
    }

    /// Adds context information to this payload.
    ///
    /// Context is used for information that should remain visible even in long
    /// conversations without being buried in history (e.g., DialogueContext, environment info).
    ///
    /// PersonaAgent will place this context strategically based on conversation length.
    ///
    /// **Note**: This is for LLM-visible natural language context. For structured context
    /// that controls expertise rendering, use `with_render_context()` or the builder methods
    /// `with_task_type()`, `with_user_state()`, `with_task_health()`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("User question")
    ///     .with_context("Environment: Production\nFocus: Security");
    /// ```
    pub fn with_context(self, context: impl Into<String>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Context(context.into()));
        Self {
            inner: Arc::new(self.create_inner(new_contents)),
        }
    }

    /// Returns all context strings from this payload.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("Task")
    ///     .with_context("Environment: Test");
    ///
    /// let contexts = payload.contexts();
    /// assert_eq!(contexts.len(), 1);
    /// ```
    pub fn contexts(&self) -> Vec<&str> {
        self.inner
            .contents
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Context(ctx) => Some(ctx.as_str()),
                _ => None,
            })
            .collect()
    }

    // ============================================================================
    // RenderContext builder methods (for ExpertiseAgent)
    // ============================================================================

    /// Sets the render context explicitly.
    ///
    /// This is used by ExpertiseAgent to determine which knowledge fragments
    /// should be included during prompt generation.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::agent::expertise::RenderContext;
    /// use llm_toolkit::TaskHealth;
    ///
    /// let context = RenderContext::new()
    ///     .with_task_type("security-review")
    ///     .with_task_health(TaskHealth::AtRisk);
    ///
    /// let payload = Payload::text("Review this code")
    ///     .with_render_context(context);
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_render_context(self, context: RenderContext) -> Self {
        let mut inner = (*self.inner).clone();
        inner.render_context = Some(context);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Returns the render context if present.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("Task")
    ///     .with_task_type("security-review");
    ///
    /// if let Some(context) = payload.render_context() {
    ///     println!("Task type: {:?}", context.task_type);
    /// }
    /// ```
    #[cfg(feature = "agent")]
    pub fn render_context(&self) -> Option<&RenderContext> {
        self.inner.render_context.as_ref()
    }

    /// Adds a task type to the render context.
    ///
    /// This is a convenience method for building render context incrementally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("Review this code")
    ///     .with_task_type("security-review");
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_task_type(self, task_type: impl Into<String>) -> Self {
        let context = self
            .inner
            .render_context
            .clone()
            .unwrap_or_default()
            .with_task_type(task_type);
        self.with_render_context(context)
    }

    /// Adds a user state to the render context.
    ///
    /// This is a convenience method for building render context incrementally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// let payload = Payload::text("Explain this concept")
    ///     .with_user_state("beginner");
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_user_state(self, state: impl Into<String>) -> Self {
        let context = self
            .inner
            .render_context
            .clone()
            .unwrap_or_default()
            .with_user_state(state);
        self.with_render_context(context)
    }

    /// Sets the task health in the render context.
    ///
    /// This is a convenience method for building render context incrementally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    /// use llm_toolkit::TaskHealth;
    ///
    /// let payload = Payload::text("Debug this issue")
    ///     .with_task_health(TaskHealth::AtRisk);
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_task_health(self, health: crate::context::TaskHealth) -> Self {
        let context = self
            .inner
            .render_context
            .clone()
            .unwrap_or_default()
            .with_task_health(health);
        self.with_render_context(context)
    }

    // ============================================================================
    // ExecutionContext methods (for Orchestrator injection)
    // ============================================================================

    /// Sets the execution context from orchestrator.
    ///
    /// This contains raw runtime information like step info, journal summary,
    /// and redesign count. Used by context detectors to infer higher-level context.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::{Payload, ExecutionContext, StepInfo};
    ///
    /// let exec_ctx = ExecutionContext::new()
    ///     .with_step_info(StepInfo::new("step_1", "Analyze code", "AnalyzerAgent"))
    ///     .with_redesign_count(2);
    ///
    /// let payload = Payload::text("Analyze this code")
    ///     .with_execution_context(exec_ctx);
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_execution_context(self, context: ExecutionContext) -> Self {
        let mut inner = (*self.inner).clone();
        inner.execution_context = Some(context);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Returns the execution context if present.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// if let Some(exec_ctx) = payload.execution_context() {
    ///     println!("Redesign count: {}", exec_ctx.redesign_count);
    /// }
    /// ```
    #[cfg(feature = "agent")]
    pub fn execution_context(&self) -> Option<&ExecutionContext> {
        self.inner.execution_context.as_ref()
    }

    // ============================================================================
    // DetectedContext methods (for layered detection)
    // ============================================================================

    /// Sets the detected context from detector analysis.
    ///
    /// For layered detection, use the `merge()` pattern to progressively
    /// enrich the detected context:
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::{Payload, DetectedContext};
    /// use llm_toolkit::TaskHealth;
    ///
    /// // Layer 1: Rule-based detection
    /// let detected1 = DetectedContext::new()
    ///     .with_task_health(TaskHealth::AtRisk)
    ///     .detected_by("RuleBasedDetector");
    ///
    /// let payload = Payload::text("Review code")
    ///     .with_detected_context(detected1);
    ///
    /// // Layer 2: LLM-based enrichment
    /// let detected2 = DetectedContext::new()
    ///     .with_user_state("confused")
    ///     .detected_by("LLMDetector");
    ///
    /// // Merge with existing
    /// let merged = payload.detected_context()
    ///     .cloned()
    ///     .unwrap_or_default()
    ///     .merge(detected2);
    ///
    /// let payload = payload.with_detected_context(merged);
    /// ```
    #[cfg(feature = "agent")]
    pub fn with_detected_context(self, context: DetectedContext) -> Self {
        let mut inner = (*self.inner).clone();
        inner.detected_context = Some(context);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Returns the detected context if present.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::Payload;
    ///
    /// if let Some(detected) = payload.detected_context() {
    ///     println!("Task health: {:?}", detected.task_health);
    ///     println!("Detected by: {:?}", detected.detected_by);
    /// }
    /// ```
    #[cfg(feature = "agent")]
    pub fn detected_context(&self) -> Option<&DetectedContext> {
        self.inner.detected_context.as_ref()
    }

    /// Merges additional detected context into existing detected context.
    ///
    /// This is a convenience method for layered detection. If no detected context
    /// exists yet, the provided context becomes the initial detected context.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::{Payload, DetectedContext};
    ///
    /// let payload = Payload::text("Review code");
    ///
    /// // Layer 1
    /// let layer1 = DetectedContext::new()
    ///     .with_task_type("security-review")
    ///     .detected_by("Layer1");
    /// let payload = payload.merge_detected_context(layer1);
    ///
    /// // Layer 2
    /// let layer2 = DetectedContext::new()
    ///     .with_user_state("beginner")
    ///     .detected_by("Layer2");
    /// let payload = payload.merge_detected_context(layer2);
    ///
    /// // Result: contains both task_type and user_state
    /// ```
    #[cfg(feature = "agent")]
    pub fn merge_detected_context(self, context: DetectedContext) -> Self {
        let merged = self
            .inner
            .detected_context
            .clone()
            .unwrap_or_default()
            .merge(context);
        self.with_detected_context(merged)
    }
}

impl Default for Payload {
    fn default() -> Self {
        Self::new()
    }
}

// Conversion from String for backward compatibility
impl From<String> for Payload {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for Payload {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

// Display implementation (automatically provides ToString)
impl fmt::Display for Payload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_from_string() {
        let payload: Payload = "Hello world".to_string().into();
        assert_eq!(payload.contents().len(), 1);
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Text(s) if s == "Hello world"
        ));
    }

    #[test]
    fn test_payload_to_text() {
        use crate::attachment::Attachment;

        let payload = Payload::text("First line")
            .with_text("Second line")
            .with_attachment(Attachment::local("/test.png"));

        assert_eq!(payload.to_text(), "First line\nSecond line");
    }

    #[test]
    fn test_payload_set_text() {
        let payload = Payload::text("Old text")
            .with_text("Another old text")
            .with_attachment(Attachment::local("/test.png"))
            .set_text("New text");

        assert_eq!(payload.contents().len(), 2);
        let texts: Vec<&str> = payload
            .contents()
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&"New text"));
        assert!(!texts.contains(&"Another old text"));
        assert!(!texts.contains(&"Old text"));
        let attachments: Vec<&Attachment> = payload
            .contents()
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Attachment(a) => Some(a),
                _ => None,
            })
            .collect();
        assert!(!attachments.is_empty());
    }

    #[test]
    fn test_payload_is_text_only() {
        use crate::attachment::Attachment;

        let text_only = Payload::text("Only text");
        assert!(text_only.is_text_only());

        let with_attachment = Payload::text("Text").with_attachment(Attachment::local("/test.png"));
        assert!(!with_attachment.is_text_only());
    }

    #[test]
    fn test_payload_builder_pattern() {
        use crate::attachment::Attachment;

        let payload = Payload::new()
            .with_text("Question: What's in this attachment?")
            .with_attachment(Attachment::local("/path/to/file.png"))
            .with_text("Additional context");

        assert_eq!(payload.contents().len(), 3);
        assert!(payload.has_attachments());
        assert!(!payload.is_text_only());
    }

    #[test]
    fn test_prepend_text() {
        let payload = Payload::text("User question").prepend_text("System instructions");

        assert_eq!(payload.contents().len(), 2);
        assert_eq!(payload.to_text(), "System instructions\nUser question");
    }

    #[test]
    fn test_prepend_text_with_multimodal() {
        use crate::attachment::Attachment;

        let payload = Payload::text("User question")
            .with_attachment(Attachment::local("/test.png"))
            .prepend_text("System instructions");

        assert_eq!(payload.contents().len(), 3);
        assert!(payload.has_attachments());
        // First element should be the prepended text
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Text(s) if s == "System instructions"
        ));
    }

    #[test]
    fn test_prepend_message() {
        use crate::agent::dialogue::Speaker;

        let payload = Payload::text("User question")
            .prepend_message(Speaker::System, "Keep responses concise.");

        assert_eq!(payload.contents().len(), 2);
        // First element should be the prepended message
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Message { speaker, content, .. }
                if matches!(speaker, Speaker::System) && content == "Keep responses concise."
        ));
        // Second element should be the original text
        assert!(matches!(
            &payload.contents()[1],
            PayloadContent::Text(s) if s == "User question"
        ));
    }

    #[test]
    fn test_prepend_system() {
        let payload = Payload::text("User question").prepend_system("IMPORTANT: Be concise.");

        assert_eq!(payload.contents().len(), 2);
        // First element should be a System message
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Message { speaker, content, .. }
                if matches!(speaker, crate::agent::dialogue::Speaker::System)
                && content == "IMPORTANT: Be concise."
        ));
    }

    #[test]
    fn test_prepend_message_with_attachments() {
        use crate::agent::dialogue::Speaker;
        use crate::attachment::Attachment;

        let payload = Payload::text("User question")
            .with_attachment(Attachment::local("/test.png"))
            .prepend_message(Speaker::System, "Analyze this image concisely.");

        assert_eq!(payload.contents().len(), 3);
        assert!(payload.has_attachments());
        // First element should be the prepended message
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Message { speaker, .. } if matches!(speaker, Speaker::System)
        ));
    }

    #[test]
    fn test_prepend_system_chain() {
        let payload = Payload::text("User question")
            .prepend_system("Second instruction")
            .prepend_system("First instruction");

        assert_eq!(payload.contents().len(), 3);
        // Should be in FIFO order: First, Second, User question
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Message { content, .. } if content == "First instruction"
        ));
        assert!(matches!(
            &payload.contents()[1],
            PayloadContent::Message { content, .. } if content == "Second instruction"
        ));
    }

    #[test]
    fn test_payload_clone_is_cheap() {
        // Test that cloning is efficient (Arc-based)
        let payload = Payload::text("Large content")
            .with_text("More content")
            .with_text("Even more content");

        let cloned = payload.clone();

        // Arc ensures both point to same data
        assert_eq!(payload.to_text(), cloned.to_text());
        assert_eq!(payload.contents().len(), cloned.contents().len());
    }

    // === Tests for Attachment support ===

    #[test]
    fn test_payload_with_attachment() {
        use crate::attachment::Attachment;

        let attachment = Attachment::in_memory(vec![1, 2, 3, 4]);
        let payload = Payload::text("Analyze this data").with_attachment(attachment.clone());

        assert_eq!(payload.contents().len(), 2);
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Text(s) if s == "Analyze this data"
        ));
        assert!(matches!(
            &payload.contents()[1],
            PayloadContent::Attachment(_)
        ));
    }

    #[test]
    fn test_payload_attachment_constructor() {
        use crate::attachment::Attachment;

        let attachment = Attachment::remote("https://example.com/file.pdf");
        let payload = Payload::attachment(attachment);

        assert_eq!(payload.contents().len(), 1);
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Attachment(_)
        ));
    }

    #[test]
    fn test_payload_has_attachments() {
        use crate::attachment::Attachment;

        let text_only = Payload::text("Just text");
        assert!(!text_only.has_attachments());

        let with_attachment =
            Payload::text("Text").with_attachment(Attachment::in_memory(vec![1, 2, 3]));
        assert!(with_attachment.has_attachments());
    }

    #[test]
    fn test_payload_attachments_iterator() {
        use crate::attachment::Attachment;

        let attachment1 = Attachment::in_memory(vec![1, 2, 3]);
        let attachment2 = Attachment::local("/test/file.png");

        let payload = Payload::text("Text")
            .with_attachment(attachment1.clone())
            .with_text("More text")
            .with_attachment(attachment2.clone());

        let attachments = payload.attachments();
        assert_eq!(attachments.len(), 2);
    }

    #[test]
    fn test_payload_multimodal_with_all_types() {
        use crate::attachment::Attachment;

        let payload = Payload::text("Question")
            .with_attachment(Attachment::local("/image.jpg"))
            .with_attachment(Attachment::in_memory(vec![1, 2, 3]))
            .with_attachment(Attachment::in_memory(vec![4, 5, 6]))
            .with_text("Additional context");

        assert_eq!(payload.contents().len(), 5);
        assert!(payload.has_attachments());
        assert!(!payload.is_text_only());
    }

    #[test]
    fn test_payload_attachment_with_builder_pattern() {
        use crate::attachment::Attachment;

        let payload = Payload::new()
            .with_text("System prompt")
            .with_attachment(Attachment::remote("https://example.com/data.json"))
            .with_attachment(Attachment::local("/local/file.txt"));

        assert_eq!(payload.contents().len(), 3);
        assert_eq!(payload.attachments().len(), 2);
    }

    #[test]
    fn test_payload_merge() {
        use crate::attachment::Attachment;

        let payload1 = Payload::text("First text").with_attachment(Attachment::local("/file1.txt"));
        let payload2 =
            Payload::text("Second text").with_attachment(Attachment::local("/file2.txt"));

        let merged = payload1.merge(payload2);

        assert_eq!(merged.contents().len(), 4);
        assert_eq!(merged.attachments().len(), 2);
        assert_eq!(merged.to_text(), "First text\nSecond text");
    }

    #[test]
    fn test_payload_merge_text_only() {
        let payload1 = Payload::text("Hello");
        let payload2 = Payload::text("World");

        let merged = payload1.merge(payload2);

        assert_eq!(merged.contents().len(), 2);
        assert_eq!(merged.to_text(), "Hello\nWorld");
        assert!(!merged.has_attachments());
    }

    #[test]
    fn test_payload_merge_empty() {
        let payload1 = Payload::text("Content");
        let payload2 = Payload::new();

        let merged = payload1.merge(payload2);

        assert_eq!(merged.contents().len(), 1);
        assert_eq!(merged.to_text(), "Content");
    }

    #[test]
    fn test_to_messages_with_text_variants() {
        let payload = Payload::text("First").with_text("Second");

        // Text variants are NOT included in to_messages()
        let messages = payload.to_messages();
        assert_eq!(messages.len(), 0);

        // Text should be retrievable via to_text()
        assert_eq!(payload.to_text(), "First\nSecond");
    }

    #[test]
    fn test_to_messages_with_message_variants() {
        use crate::agent::dialogue::Speaker;

        let payload = Payload::from_messages(vec![
            PayloadMessage::system("System message"),
            PayloadMessage::user("Alice", "PM", "User message"),
        ]);

        let messages = payload.to_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].speaker, Speaker::System);
        assert_eq!(messages[0].content, "System message");
        assert_eq!(messages[1].speaker, Speaker::user("Alice", "PM"));
        assert_eq!(messages[1].content, "User message");
    }

    #[test]
    fn test_to_messages_mixed_content() {
        use crate::agent::dialogue::Speaker;
        use crate::attachment::Attachment;

        let payload = Payload::text("Plain text")
            .with_message(Speaker::user("Bob", "Dev"), "User input")
            .with_attachment(Attachment::local("/test.png"))
            .with_text("More text");

        // to_messages() returns only Message variants (Text and Attachment excluded)
        let messages = payload.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].speaker, Speaker::user("Bob", "Dev"));
        assert_eq!(messages[0].content, "User input");

        // to_text() returns only Text variants
        assert_eq!(payload.to_text(), "Plain text\nMore text");
    }

    #[test]
    fn test_to_text_does_not_include_messages() {
        let payload = Payload::from_messages(vec![
            PayloadMessage::system("System message"),
            PayloadMessage::user("Alice", "PM", "User message"),
        ]);

        // to_text() should return empty string (only Text variants)
        assert_eq!(payload.to_text(), "");
    }

    #[test]
    fn test_to_text_vs_to_messages_separation() {
        use crate::agent::dialogue::Speaker;

        let payload = Payload::text("Plain text 1")
            .with_text("Plain text 2")
            .with_message(Speaker::System, "Message 1");

        // to_text() returns only Text variants
        assert_eq!(payload.to_text(), "Plain text 1\nPlain text 2");

        // to_messages() returns only Message variants (Text excluded)
        let messages = payload.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].speaker, Speaker::System);
        assert_eq!(messages[0].content, "Message 1");
    }

    #[test]
    fn test_payload_from_to_prompt() {
        use llm_toolkit_macros::ToPrompt;
        use serde::{Deserialize, Serialize};

        // Define a simple DTO with ToPrompt
        #[derive(Serialize, Deserialize, ToPrompt)]
        struct TestDto {
            name: String,
            value: i32,
        }

        let dto = TestDto {
            name: "test".to_string(),
            value: 42,
        };

        // Before: had to write dto.to_prompt().into()
        // Now: can use Payload::from_prompt(dto)
        let payload = Payload::from_prompt(dto);

        // Verify it contains the structured data in ToPrompt format
        let text = payload.to_text();
        assert!(text.contains("name: test"));
        assert!(text.contains("value: 42"));
    }

    #[test]
    fn test_payload_from_string_still_works() {
        // Verify that String conversion still works (more specific impl takes precedence)
        let payload: Payload = "test string".to_string().into();
        assert_eq!(payload.to_text(), "test string");

        let payload: Payload = "test &str".into();
        assert_eq!(payload.to_text(), "test &str");
    }

    // === Tests for Context support ===

    #[test]
    fn test_payload_with_context() {
        let payload =
            Payload::text("User question").with_context("Environment: Production\nFocus: Security");

        assert_eq!(payload.contents().len(), 2);

        // First should be Text
        assert!(matches!(
            &payload.contents()[0],
            PayloadContent::Text(s) if s == "User question"
        ));

        // Second should be Context
        assert!(matches!(
            &payload.contents()[1],
            PayloadContent::Context(s) if s == "Environment: Production\nFocus: Security"
        ));
    }

    #[test]
    fn test_payload_contexts() {
        let payload = Payload::text("Question")
            .with_context("Context 1")
            .with_context("Context 2");

        let contexts = payload.contexts();
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0], "Context 1");
        assert_eq!(contexts[1], "Context 2");
    }

    #[test]
    fn test_payload_contexts_empty() {
        let payload = Payload::text("Just text");
        let contexts = payload.contexts();
        assert_eq!(contexts.len(), 0);
    }

    #[test]
    fn test_payload_context_with_messages() {
        let payload = Payload::from_messages(vec![
            PayloadMessage::system("System instruction"),
            PayloadMessage::user("Alice", "PM", "User question"),
        ])
        .with_context("Important context");

        // Should have 2 messages + 1 context
        assert_eq!(payload.contents().len(), 3);

        let contexts = payload.contexts();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0], "Important context");
    }

    #[test]
    fn test_payload_context_not_in_to_text() {
        let payload = Payload::text("Text content").with_context("Context content");

        // Context should not appear in to_text() output
        assert_eq!(payload.to_text(), "Text content");
    }

    #[test]
    fn test_payload_context_not_in_to_messages() {
        let payload = Payload::text("Text").with_context("Context");

        // Context should not appear in to_messages() output
        let messages = payload.to_messages();
        assert_eq!(messages.len(), 0); // Text also doesn't appear in to_messages
    }

    #[test]
    fn test_payload_merge_with_context() {
        let payload1 = Payload::text("First").with_context("Context 1");
        let payload2 = Payload::text("Second").with_context("Context 2");

        let merged = payload1.merge(payload2);

        assert_eq!(merged.contents().len(), 4);
        let contexts = merged.contexts();
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0], "Context 1");
        assert_eq!(contexts[1], "Context 2");
    }

    #[test]
    fn test_payload_total_content_count_includes_context() {
        let payload = Payload::text("Hello").with_context("World");

        // "Hello" (5) + "World" (5) = 10
        assert_eq!(payload.total_content_count(), 10);
    }
}
