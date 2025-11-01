//! Payload types for multi-modal agent communication.
//!
//! This module provides abstractions for passing different types of content
//! (text, images, etc.) to agents in a unified way.

use crate::attachment::Attachment;
use std::fmt;
use std::sync::Arc;

/// Content that can be included in a payload.
///
/// This enum allows agents to receive different types of input,
/// including text, attachments, and potentially other media types in the future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadContent {
    /// Plain text content (treated as User input in dialogue context)
    Text(String),

    /// Arbitrary file attachment (local, remote, or in-memory)
    Attachment(Attachment),

    /// Dialogue message with explicit speaker information
    Message {
        speaker: crate::agent::dialogue::Speaker,
        content: String,
    },
}

/// Inner payload data, wrapped in Arc for efficient cloning.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PayloadInner {
    contents: Vec<PayloadContent>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
            }),
        }
    }

    /// Creates a payload with a single text content.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(PayloadInner {
                contents: vec![PayloadContent::Text(text.into())],
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
            }),
        }
    }

    /// Returns a reference to the contents of this payload.
    pub fn contents(&self) -> &[PayloadContent] {
        &self.inner.contents
    }

    /// Adds text content to this payload.
    pub fn with_text(self, text: impl Into<String>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Text(text.into()));
        Self {
            inner: Arc::new(PayloadInner {
                contents: new_contents,
            }),
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
            inner: Arc::new(PayloadInner {
                contents: new_contents,
            }),
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
            inner: Arc::new(PayloadInner {
                contents: new_contents,
            }),
        }
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
            inner: Arc::new(PayloadInner {
                contents: new_contents,
            }),
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
        });
        Self {
            inner: Arc::new(PayloadInner {
                contents: new_contents,
            }),
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
    ///     (Speaker::System, "You are a helpful assistant".to_string()),
    ///     (Speaker::user("user1"), "What is Rust?".to_string()),
    /// ]);
    /// ```
    pub fn from_messages(messages: Vec<(crate::agent::dialogue::Speaker, String)>) -> Self {
        let contents = messages
            .into_iter()
            .map(|(speaker, content)| PayloadContent::Message { speaker, content })
            .collect();

        Self {
            inner: Arc::new(PayloadInner { contents }),
        }
    }

    /// Returns all text contents concatenated with newlines.
    ///
    /// This is useful for agents that only support text input.
    /// Note: This only returns Text variants, not Message variants.
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
}
