//! Attachment types for multimodal workflows.
//!
//! This module provides the foundation for handling file-based outputs from agents
//! that can be consumed by subsequent agents in a workflow.

use std::path::PathBuf;

/// Represents a resource that can be attached to a payload or produced by an agent.
///
/// Attachments provide a flexible way to handle various types of data sources:
/// - Local files on the filesystem
/// - Remote resources accessible via URLs
/// - In-memory data with optional metadata
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attachment {
    /// A file on the local filesystem.
    Local(PathBuf),

    /// A resource accessible via a URL (e.g., http://, https://, s3://).
    ///
    /// Note: Remote fetching is not yet implemented. This variant is reserved
    /// for future functionality.
    Remote(String),

    /// In-memory data with optional name and MIME type.
    InMemory {
        /// The raw bytes of the attachment.
        bytes: Vec<u8>,
        /// Optional file name for identification.
        file_name: Option<String>,
        /// Optional MIME type (e.g., "image/png", "application/pdf").
        mime_type: Option<String>,
    },
}

impl Attachment {
    /// Creates a new local file attachment.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::attachment::Attachment;
    /// use std::path::PathBuf;
    ///
    /// let attachment = Attachment::local(PathBuf::from("/path/to/file.png"));
    /// ```
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }

    /// Creates a new remote URL attachment.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let attachment = Attachment::remote("https://example.com/image.png");
    /// ```
    pub fn remote(url: impl Into<String>) -> Self {
        Self::Remote(url.into())
    }

    /// Creates a new in-memory attachment from raw bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let data = vec![0x89, 0x50, 0x4E, 0x47]; // PNG header
    /// let attachment = Attachment::in_memory(data);
    /// ```
    pub fn in_memory(bytes: Vec<u8>) -> Self {
        Self::InMemory {
            bytes,
            file_name: None,
            mime_type: None,
        }
    }

    /// Creates a new in-memory attachment with metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// let data = vec![0x89, 0x50, 0x4E, 0x47];
    /// let attachment = Attachment::in_memory_with_meta(
    ///     data,
    ///     Some("chart.png".to_string()),
    ///     Some("image/png".to_string()),
    /// );
    /// ```
    pub fn in_memory_with_meta(
        bytes: Vec<u8>,
        file_name: Option<String>,
        mime_type: Option<String>,
    ) -> Self {
        Self::InMemory {
            bytes,
            file_name,
            mime_type,
        }
    }

    /// Returns the file name if available.
    ///
    /// For local files, extracts the file name from the path.
    /// For remote URLs, extracts the last path segment.
    /// For in-memory attachments, returns the stored file name.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::attachment::Attachment;
    /// use std::path::PathBuf;
    ///
    /// let attachment = Attachment::local(PathBuf::from("/path/to/file.png"));
    /// assert_eq!(attachment.file_name(), Some("file.png".to_string()));
    /// ```
    pub fn file_name(&self) -> Option<String> {
        match self {
            Self::Local(path) => path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string()),
            Self::Remote(url) => url
                .split('/')
                .next_back()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string()),
            Self::InMemory { file_name, .. } => file_name.clone(),
        }
    }

    /// Returns the MIME type if available or can be inferred.
    ///
    /// For local files, attempts to infer the MIME type from the file extension.
    /// For in-memory attachments, returns the stored MIME type.
    /// For remote URLs, returns None.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_toolkit::attachment::Attachment;
    /// use std::path::PathBuf;
    ///
    /// let attachment = Attachment::local(PathBuf::from("/path/to/file.png"));
    /// assert_eq!(attachment.mime_type(), Some("image/png".to_string()));
    /// ```
    pub fn mime_type(&self) -> Option<String> {
        match self {
            Self::InMemory { mime_type, .. } => mime_type.clone(),
            Self::Local(path) => Self::infer_mime_type_from_path(path),
            Self::Remote(_) => None,
        }
    }

    /// Infers MIME type from file extension.
    fn infer_mime_type_from_path(path: &std::path::Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "jpg" | "jpeg" => Some("image/jpeg".to_string()),
                "png" => Some("image/png".to_string()),
                "gif" => Some("image/gif".to_string()),
                "webp" => Some("image/webp".to_string()),
                "bmp" => Some("image/bmp".to_string()),
                "svg" => Some("image/svg+xml".to_string()),
                "pdf" => Some("application/pdf".to_string()),
                "json" => Some("application/json".to_string()),
                "xml" => Some("application/xml".to_string()),
                "txt" => Some("text/plain".to_string()),
                "html" | "htm" => Some("text/html".to_string()),
                "csv" => Some("text/csv".to_string()),
                "md" => Some("text/markdown".to_string()),
                _ => None,
            })
    }

    /// Loads the attachment data as bytes.
    ///
    /// For local files, reads the file from the filesystem.
    /// For in-memory attachments, returns a clone of the stored bytes.
    /// For remote URLs, returns an error (not yet implemented).
    ///
    /// This method is only available when the `agent` feature is enabled,
    /// as it requires async runtime support.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read (for local attachments)
    /// - Remote fetching is attempted (not yet supported)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llm_toolkit::attachment::Attachment;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let attachment = Attachment::in_memory(vec![1, 2, 3]);
    /// let bytes = attachment.load_bytes().await?;
    /// assert_eq!(bytes, vec![1, 2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "agent")]
    pub async fn load_bytes(&self) -> Result<Vec<u8>, std::io::Error> {
        match self {
            Self::Local(path) => tokio::fs::read(path).await,
            Self::InMemory { bytes, .. } => Ok(bytes.clone()),
            Self::Remote(_url) => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Remote attachment loading not yet implemented",
            )),
        }
    }
}

/// Trait for types that can produce named attachments.
///
/// This trait is typically derived using `#[derive(ToAttachments)]`.
/// Types implementing this trait can be used as agent outputs that produce
/// file-based data that can be consumed by subsequent agents in a workflow.
///
/// # Examples
///
/// ```
/// use llm_toolkit::attachment::{Attachment, ToAttachments};
/// use std::path::PathBuf;
///
/// // Manual implementation
/// struct MyOutput {
///     data: Vec<u8>,
/// }
///
/// impl ToAttachments for MyOutput {
///     fn to_attachments(&self) -> Vec<(String, Attachment)> {
///         vec![("data".to_string(), Attachment::in_memory(self.data.clone()))]
///     }
/// }
///
/// let output = MyOutput { data: vec![1, 2, 3] };
/// let attachments = output.to_attachments();
/// assert_eq!(attachments.len(), 1);
/// assert_eq!(attachments[0].0, "data");
/// ```
pub trait ToAttachments {
    /// Converts this type into a list of named attachments.
    ///
    /// Returns `Vec<(key, Attachment)>` where key identifies the attachment.
    /// The key is used by the orchestrator to reference this attachment in
    /// subsequent steps.
    fn to_attachments(&self) -> Vec<(String, Attachment)>;
}

/// Trait for types that can declare their attachment schema at compile-time.
///
/// This trait is automatically implemented when deriving `ToAttachments`.
/// It provides metadata about what attachment keys a type will produce,
/// which is used by the Agent derive macro to augment the agent's expertise.
///
/// # Examples
///
/// ```
/// use llm_toolkit::attachment::AttachmentSchema;
///
/// struct MyOutput;
///
/// impl AttachmentSchema for MyOutput {
///     fn attachment_keys() -> &'static [&'static str] {
///         &["chart", "thumbnail"]
///     }
/// }
///
/// assert_eq!(MyOutput::attachment_keys(), &["chart", "thumbnail"]);
/// ```
pub trait AttachmentSchema {
    /// Returns a static slice of attachment keys this type produces.
    fn attachment_keys() -> &'static [&'static str];

    /// Optional: Returns descriptions for each attachment key.
    ///
    /// The tuple format is `(key, description)`.
    fn attachment_descriptions() -> Option<&'static [(&'static str, &'static str)]> {
        None
    }
}

// === Blanket implementations for common types ===

impl ToAttachments for Vec<u8> {
    fn to_attachments(&self) -> Vec<(String, Attachment)> {
        vec![("data".to_string(), Attachment::in_memory(self.clone()))]
    }
}

impl ToAttachments for PathBuf {
    fn to_attachments(&self) -> Vec<(String, Attachment)> {
        vec![("file".to_string(), Attachment::local(self.clone()))]
    }
}

impl ToAttachments for Attachment {
    fn to_attachments(&self) -> Vec<(String, Attachment)> {
        vec![("attachment".to_string(), self.clone())]
    }
}

impl<T: ToAttachments> ToAttachments for Option<T> {
    fn to_attachments(&self) -> Vec<(String, Attachment)> {
        match self {
            Some(inner) => inner.to_attachments(),
            None => Vec::new(),
        }
    }
}

impl<T: ToAttachments> ToAttachments for Vec<T> {
    fn to_attachments(&self) -> Vec<(String, Attachment)> {
        self.iter()
            .enumerate()
            .flat_map(|(i, item)| {
                item.to_attachments()
                    .into_iter()
                    .map(move |(key, attachment)| (format!("{}_{}", key, i), attachment))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Tests for Attachment ===

    #[test]
    fn test_local_attachment_creation() {
        let path = PathBuf::from("/path/to/file.png");
        let attachment = Attachment::local(path.clone());

        match attachment {
            Attachment::Local(p) => assert_eq!(p, path),
            _ => panic!("Expected Local variant"),
        }
    }

    #[test]
    fn test_remote_attachment_creation() {
        let url = "https://example.com/image.png";
        let attachment = Attachment::remote(url);

        match attachment {
            Attachment::Remote(u) => assert_eq!(u, url),
            _ => panic!("Expected Remote variant"),
        }
    }

    #[test]
    fn test_in_memory_attachment_creation() {
        let data = vec![1, 2, 3, 4];
        let attachment = Attachment::in_memory(data.clone());

        match attachment {
            Attachment::InMemory {
                bytes,
                file_name,
                mime_type,
            } => {
                assert_eq!(bytes, data);
                assert_eq!(file_name, None);
                assert_eq!(mime_type, None);
            }
            _ => panic!("Expected InMemory variant"),
        }
    }

    #[test]
    fn test_in_memory_attachment_with_metadata() {
        let data = vec![1, 2, 3, 4];
        let name = Some("test.png".to_string());
        let mime = Some("image/png".to_string());

        let attachment = Attachment::in_memory_with_meta(data.clone(), name.clone(), mime.clone());

        match attachment {
            Attachment::InMemory {
                bytes,
                file_name,
                mime_type,
            } => {
                assert_eq!(bytes, data);
                assert_eq!(file_name, name);
                assert_eq!(mime_type, mime);
            }
            _ => panic!("Expected InMemory variant"),
        }
    }

    #[test]
    fn test_file_name_extraction_local() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file.png"));
        assert_eq!(attachment.file_name(), Some("file.png".to_string()));
    }

    #[test]
    fn test_file_name_extraction_local_no_extension() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file"));
        assert_eq!(attachment.file_name(), Some("file".to_string()));
    }

    #[test]
    fn test_file_name_extraction_remote() {
        let attachment = Attachment::remote("https://example.com/path/to/image.jpg");
        assert_eq!(attachment.file_name(), Some("image.jpg".to_string()));
    }

    #[test]
    fn test_file_name_extraction_remote_trailing_slash() {
        // Trailing slash indicates a directory, so no file name
        let attachment = Attachment::remote("https://example.com/path/to/");
        assert_eq!(attachment.file_name(), None);
    }

    #[test]
    fn test_file_name_extraction_in_memory() {
        let attachment =
            Attachment::in_memory_with_meta(vec![1, 2, 3], Some("chart.png".to_string()), None);
        assert_eq!(attachment.file_name(), Some("chart.png".to_string()));
    }

    #[test]
    fn test_file_name_extraction_in_memory_none() {
        let attachment = Attachment::in_memory(vec![1, 2, 3]);
        assert_eq!(attachment.file_name(), None);
    }

    #[test]
    fn test_mime_type_inference_png() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file.png"));
        assert_eq!(attachment.mime_type(), Some("image/png".to_string()));
    }

    #[test]
    fn test_mime_type_inference_jpg() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file.jpg"));
        assert_eq!(attachment.mime_type(), Some("image/jpeg".to_string()));
    }

    #[test]
    fn test_mime_type_inference_jpeg() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file.jpeg"));
        assert_eq!(attachment.mime_type(), Some("image/jpeg".to_string()));
    }

    #[test]
    fn test_mime_type_inference_pdf() {
        let attachment = Attachment::local(PathBuf::from("/path/to/document.pdf"));
        assert_eq!(attachment.mime_type(), Some("application/pdf".to_string()));
    }

    #[test]
    fn test_mime_type_inference_json() {
        let attachment = Attachment::local(PathBuf::from("/path/to/data.json"));
        assert_eq!(attachment.mime_type(), Some("application/json".to_string()));
    }

    #[test]
    fn test_mime_type_inference_unknown_extension() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file.unknown"));
        assert_eq!(attachment.mime_type(), None);
    }

    #[test]
    fn test_mime_type_inference_no_extension() {
        let attachment = Attachment::local(PathBuf::from("/path/to/file"));
        assert_eq!(attachment.mime_type(), None);
    }

    #[test]
    fn test_mime_type_in_memory_with_type() {
        let attachment = Attachment::in_memory_with_meta(
            vec![1, 2, 3],
            None,
            Some("application/octet-stream".to_string()),
        );
        assert_eq!(
            attachment.mime_type(),
            Some("application/octet-stream".to_string())
        );
    }

    #[test]
    fn test_mime_type_in_memory_without_type() {
        let attachment = Attachment::in_memory(vec![1, 2, 3]);
        assert_eq!(attachment.mime_type(), None);
    }

    #[test]
    fn test_mime_type_remote() {
        let attachment = Attachment::remote("https://example.com/file.png");
        assert_eq!(attachment.mime_type(), None);
    }

    #[cfg(feature = "agent")]
    #[tokio::test]
    async fn test_load_bytes_in_memory() {
        let data = vec![1, 2, 3, 4, 5];
        let attachment = Attachment::in_memory(data.clone());

        let loaded = attachment.load_bytes().await.unwrap();
        assert_eq!(loaded, data);
    }

    #[cfg(feature = "agent")]
    #[tokio::test]
    async fn test_load_bytes_remote_unsupported() {
        let attachment = Attachment::remote("https://example.com/file.png");

        let result = attachment.load_bytes().await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn test_attachment_clone() {
        let attachment = Attachment::in_memory_with_meta(
            vec![1, 2, 3],
            Some("test.bin".to_string()),
            Some("application/octet-stream".to_string()),
        );

        let cloned = attachment.clone();
        assert_eq!(attachment, cloned);
    }

    #[test]
    fn test_attachment_debug() {
        let attachment = Attachment::local(PathBuf::from("/test/path.txt"));
        let debug_str = format!("{:?}", attachment);
        assert!(debug_str.contains("Local"));
        assert!(debug_str.contains("path.txt"));
    }

    // === Tests for ToAttachments trait ===

    #[test]
    fn test_to_attachments_vec_u8() {
        let data = vec![1, 2, 3, 4, 5];
        let attachments = data.to_attachments();

        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].0, "data");
        match &attachments[0].1 {
            Attachment::InMemory { bytes, .. } => assert_eq!(bytes, &data),
            _ => panic!("Expected InMemory attachment"),
        }
    }

    #[test]
    fn test_to_attachments_pathbuf() {
        let path = PathBuf::from("/test/file.txt");
        let attachments = path.to_attachments();

        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].0, "file");
        match &attachments[0].1 {
            Attachment::Local(p) => assert_eq!(p, &path),
            _ => panic!("Expected Local attachment"),
        }
    }

    #[test]
    fn test_to_attachments_attachment() {
        let attachment = Attachment::remote("https://example.com/file.pdf");
        let attachments = attachment.to_attachments();

        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].0, "attachment");
    }

    #[test]
    fn test_to_attachments_option_some() {
        let data = Some(vec![1, 2, 3]);
        let attachments = data.to_attachments();

        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].0, "data");
    }

    #[test]
    fn test_to_attachments_option_none() {
        let data: Option<Vec<u8>> = None;
        let attachments = data.to_attachments();

        assert_eq!(attachments.len(), 0);
    }

    #[test]
    fn test_to_attachments_vec() {
        let items = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let attachments = items.to_attachments();

        assert_eq!(attachments.len(), 2);
        assert_eq!(attachments[0].0, "data_0");
        assert_eq!(attachments[1].0, "data_1");
    }

    #[test]
    fn test_to_attachments_custom_implementation() {
        struct MyOutput {
            chart: Vec<u8>,
            thumbnail: Vec<u8>,
        }

        impl ToAttachments for MyOutput {
            fn to_attachments(&self) -> Vec<(String, Attachment)> {
                vec![
                    (
                        "chart".to_string(),
                        Attachment::in_memory(self.chart.clone()),
                    ),
                    (
                        "thumbnail".to_string(),
                        Attachment::in_memory(self.thumbnail.clone()),
                    ),
                ]
            }
        }

        let output = MyOutput {
            chart: vec![1, 2, 3],
            thumbnail: vec![4, 5, 6],
        };

        let attachments = output.to_attachments();
        assert_eq!(attachments.len(), 2);
        assert_eq!(attachments[0].0, "chart");
        assert_eq!(attachments[1].0, "thumbnail");
    }

    // === Tests for AttachmentSchema trait ===

    #[test]
    fn test_attachment_schema_keys() {
        struct TestOutput;

        impl AttachmentSchema for TestOutput {
            fn attachment_keys() -> &'static [&'static str] {
                &["image", "data"]
            }
        }

        let keys = TestOutput::attachment_keys();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], "image");
        assert_eq!(keys[1], "data");
    }

    #[test]
    fn test_attachment_schema_with_descriptions() {
        struct TestOutput;

        impl AttachmentSchema for TestOutput {
            fn attachment_keys() -> &'static [&'static str] {
                &["chart", "report"]
            }

            fn attachment_descriptions() -> Option<&'static [(&'static str, &'static str)]> {
                Some(&[
                    ("chart", "Visual chart of the data"),
                    ("report", "Detailed text report"),
                ])
            }
        }

        let keys = TestOutput::attachment_keys();
        assert_eq!(keys, &["chart", "report"]);

        let descriptions = TestOutput::attachment_descriptions().unwrap();
        assert_eq!(descriptions.len(), 2);
        assert_eq!(descriptions[0].0, "chart");
        assert_eq!(descriptions[0].1, "Visual chart of the data");
    }

    #[test]
    fn test_attachment_schema_empty_keys() {
        struct EmptyOutput;

        impl AttachmentSchema for EmptyOutput {
            fn attachment_keys() -> &'static [&'static str] {
                &[]
            }
        }

        let keys = EmptyOutput::attachment_keys();
        assert_eq!(keys.len(), 0);
    }
}
