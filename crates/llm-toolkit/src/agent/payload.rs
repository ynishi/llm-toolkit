//! Payload types for multi-modal agent communication.
//!
//! This module provides abstractions for passing different types of content
//! (text, images, etc.) to agents in a unified way.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// Content that can be included in a payload.
///
/// This enum allows agents to receive different types of input,
/// including text, images, and potentially other media types in the future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadContent {
    /// Plain text content
    Text(String),
    /// Image from a file path
    Image(PathBuf),
    /// Image from raw bytes (e.g., PNG, JPEG)
    ImageData(Vec<u8>),
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
/// // Multi-modal payload with text and image
/// let payload = Payload::text("What's in this image?")
///     .with_image(PathBuf::from("/path/to/image.png"));
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

    /// Creates a payload with a single image from a path.
    pub fn image(path: PathBuf) -> Self {
        Self {
            inner: Arc::new(PayloadInner {
                contents: vec![PayloadContent::Image(path)],
            }),
        }
    }

    /// Creates a payload with a single image from raw bytes.
    pub fn image_data(data: Vec<u8>) -> Self {
        Self {
            inner: Arc::new(PayloadInner {
                contents: vec![PayloadContent::ImageData(data)],
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

    /// Adds an image from a path to this payload.
    pub fn with_image(self, path: PathBuf) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::Image(path));
        Self {
            inner: Arc::new(PayloadInner {
                contents: new_contents,
            }),
        }
    }

    /// Adds an image from raw bytes to this payload.
    pub fn with_image_data(self, data: Vec<u8>) -> Self {
        let mut new_contents = self.inner.contents.clone();
        new_contents.push(PayloadContent::ImageData(data));
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

    /// Returns all text contents concatenated with newlines.
    ///
    /// This is useful for agents that only support text input.
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

    /// Returns true if this payload contains any images.
    pub fn has_images(&self) -> bool {
        self.inner
            .contents
            .iter()
            .any(|c| matches!(c, PayloadContent::Image(_) | PayloadContent::ImageData(_)))
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
        let payload = Payload::text("First line")
            .with_text("Second line")
            .with_image(PathBuf::from("/test.png"));

        assert_eq!(payload.to_text(), "First line\nSecond line");
    }

    #[test]
    fn test_payload_is_text_only() {
        let text_only = Payload::text("Only text");
        assert!(text_only.is_text_only());

        let with_image = Payload::text("Text").with_image(PathBuf::from("/test.png"));
        assert!(!with_image.is_text_only());
    }

    #[test]
    fn test_payload_has_images() {
        let text_only = Payload::text("Only text");
        assert!(!text_only.has_images());

        let with_image = Payload::text("Text").with_image(PathBuf::from("/test.png"));
        assert!(with_image.has_images());

        let with_image_data = Payload::text("Text").with_image_data(vec![1, 2, 3]);
        assert!(with_image_data.has_images());
    }

    #[test]
    fn test_payload_builder_pattern() {
        let payload = Payload::new()
            .with_text("Question: What's in this image?")
            .with_image(PathBuf::from("/path/to/image.png"))
            .with_text("Additional context");

        assert_eq!(payload.contents().len(), 3);
        assert!(payload.has_images());
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
        let payload = Payload::text("User question")
            .with_image(PathBuf::from("/test.png"))
            .prepend_text("System instructions");

        assert_eq!(payload.contents().len(), 3);
        assert!(payload.has_images());
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
}
