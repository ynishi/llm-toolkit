//! Payload types for multi-modal agent communication.
//!
//! This module provides abstractions for passing different types of content
//! (text, images, etc.) to agents in a unified way.

use std::fmt;
use std::path::PathBuf;

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

/// A multi-modal payload that can contain multiple content items.
///
/// This allows passing text, images, and other content types to agents
/// in a flexible and extensible way.
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
/// let payload = Payload {
///     contents: vec![
///         PayloadContent::Text("What's in this image?".to_string()),
///         PayloadContent::Image(PathBuf::from("/path/to/image.png")),
///     ],
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payload {
    /// The contents of this payload
    pub contents: Vec<PayloadContent>,
}

impl Payload {
    /// Creates a new empty payload.
    pub fn new() -> Self {
        Self {
            contents: Vec::new(),
        }
    }

    /// Creates a payload with a single text content.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            contents: vec![PayloadContent::Text(text.into())],
        }
    }

    /// Creates a payload with a single image from a path.
    pub fn image(path: PathBuf) -> Self {
        Self {
            contents: vec![PayloadContent::Image(path)],
        }
    }

    /// Creates a payload with a single image from raw bytes.
    pub fn image_data(data: Vec<u8>) -> Self {
        Self {
            contents: vec![PayloadContent::ImageData(data)],
        }
    }

    /// Adds text content to this payload.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.contents.push(PayloadContent::Text(text.into()));
        self
    }

    /// Adds an image from a path to this payload.
    pub fn with_image(mut self, path: PathBuf) -> Self {
        self.contents.push(PayloadContent::Image(path));
        self
    }

    /// Adds an image from raw bytes to this payload.
    pub fn with_image_data(mut self, data: Vec<u8>) -> Self {
        self.contents.push(PayloadContent::ImageData(data));
        self
    }

    /// Returns all text contents concatenated with newlines.
    ///
    /// This is useful for agents that only support text input.
    pub fn to_text(&self) -> String {
        self.contents
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
        self.contents
            .iter()
            .all(|c| matches!(c, PayloadContent::Text(_)))
    }

    /// Returns true if this payload contains any images.
    pub fn has_images(&self) -> bool {
        self.contents
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
        assert_eq!(payload.contents.len(), 1);
        assert!(matches!(
            &payload.contents[0],
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

        assert_eq!(payload.contents.len(), 3);
        assert!(payload.has_images());
        assert!(!payload.is_text_only());
    }
}
