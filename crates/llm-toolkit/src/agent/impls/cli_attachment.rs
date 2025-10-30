//! Attachment handling utilities for CLI-based agents.
//!
//! Provides functionality to write attachments to temporary files and format
//! prompts with attachment paths for CLI tools like `gemini` and `claude`.

use crate::agent::AgentError;
use crate::attachment::Attachment;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// RAII guard for temporary attachment directory.
///
/// Automatically cleans up the directory and all its contents when dropped,
/// unless `keep` is set to true.
pub(crate) struct TempAttachmentDir {
    path: PathBuf,
    keep: bool,
}

impl TempAttachmentDir {
    /// Creates a new temporary directory for attachments.
    ///
    /// # Arguments
    /// * `base_dir` - Parent directory where temp dir will be created
    /// * `keep` - If true, directory won't be deleted on drop
    pub fn new(base_dir: &Path, keep: bool) -> std::io::Result<Self> {
        let session_id = generate_session_id();
        let dir_name = format!("llm-toolkit-attachments-{}", session_id);
        let path = base_dir.join(dir_name);
        std::fs::create_dir_all(&path)?;

        debug!(
            target: "llm_toolkit::agent::cli_attachment",
            "Created temp attachment directory: {}", path.display()
        );

        Ok(Self { path, keep })
    }

    /// Returns the path to the temporary directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempAttachmentDir {
    fn drop(&mut self) {
        if !self.keep {
            if let Err(e) = std::fs::remove_dir_all(&self.path) {
                warn!(
                    target: "llm_toolkit::agent::cli_attachment",
                    "Failed to clean up temp attachment dir {}: {}",
                    self.path.display(),
                    e
                );
            } else {
                debug!(
                    target: "llm_toolkit::agent::cli_attachment",
                    "Cleaned up temp attachment directory: {}", self.path.display()
                );
            }
        } else {
            debug!(
                target: "llm_toolkit::agent::cli_attachment",
                "Keeping temp attachment directory: {}", self.path.display()
            );
        }
    }
}

/// Generates a unique session ID for the temporary directory.
///
/// Format: `{timestamp_hex}_{random_hex}`
fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // Generate random component using timestamp-based seed
    // This avoids needing the `rand` crate dependency
    let random = (timestamp % 0xFFFFFFFF) as u32;

    format!("{:x}_{:x}", timestamp, random)
}

/// Generates a unique filename for an attachment.
///
/// Preserves original filename and extension when available,
/// adds UUID to prevent collisions.
fn generate_temp_filename(attachment: &Attachment, index: usize) -> String {
    let base_name = attachment
        .file_name()
        .as_ref()
        .and_then(|name| {
            // Remove extension
            Path::new(name.as_str())
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| format!("attachment_{}", index));

    let extension = attachment
        .file_name()
        .as_ref()
        .and_then(|name| {
            Path::new(name.as_str())
                .extension()
                .map(|s| format!(".{}", s.to_string_lossy()))
        })
        .unwrap_or_else(|| {
            // Infer extension from mime type
            match attachment.mime_type().as_deref() {
                Some("image/png") => ".png",
                Some("image/jpeg") | Some("image/jpg") => ".jpg",
                Some("application/pdf") => ".pdf",
                Some("application/json") => ".json",
                Some("text/plain") => ".txt",
                _ => "",
            }
            .to_string()
        });

    // Generate short unique ID (8 chars from timestamp)
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let unique_id = format!("{:x}", timestamp % 0xFFFFFFFF);
    let short_id = &unique_id[..unique_id.len().min(8)];

    format!("{}_{}{}", base_name, short_id, extension)
}

/// Processes all attachments and writes them to temporary files.
///
/// Returns a vector of paths to the created files.
///
/// # Attachment Handling
/// - **InMemory**: Writes bytes to temp file
/// - **Local**: Copies file to temp directory
/// - **Remote**: Logs warning and skips (URLs not downloaded)
pub async fn process_attachments(
    attachments: &[&Attachment],
    temp_dir: &Path,
) -> Result<Vec<PathBuf>, AgentError> {
    let mut paths = Vec::new();

    for (index, attachment) in attachments.iter().enumerate() {
        match **attachment {
            Attachment::InMemory { ref bytes, .. } => {
                let filename = generate_temp_filename(attachment, index);
                let path = temp_dir.join(&filename);

                debug!(
                    target: "llm_toolkit::agent::cli_attachment",
                    "Writing InMemory attachment to: {}", path.display()
                );

                tokio::fs::write(&path, bytes).await.map_err(|e| {
                    AgentError::Other(format!("Failed to write attachment {}: {}", filename, e))
                })?;

                paths.push(path);
            }
            Attachment::Local(ref source) => {
                let filename = generate_temp_filename(attachment, index);
                let dest = temp_dir.join(&filename);

                debug!(
                    target: "llm_toolkit::agent::cli_attachment",
                    "Copying Local attachment from {} to {}", source.display(), dest.display()
                );

                tokio::fs::copy(source, &dest).await.map_err(|e| {
                    AgentError::Other(format!(
                        "Failed to copy attachment from {}: {}",
                        source.display(),
                        e
                    ))
                })?;

                paths.push(dest);
            }
            Attachment::Remote(ref url) => {
                warn!(
                    target: "llm_toolkit::agent::cli_attachment",
                    "Remote attachments are not yet supported, skipping: {}", url
                );
                // Skip remote attachments for now
                // Future: could download to temp file
            }
        }
    }

    Ok(paths)
}

/// Formats a prompt with attachment paths appended.
///
/// If there are no attachments, returns the original prompt unchanged.
///
/// # Format
/// ```text
/// <prompt>
///
/// Attachments:
/// - /path/to/file1.png (image/png)
/// - /path/to/file2.pdf (application/pdf)
/// ```
pub fn format_prompt_with_attachments(prompt: &str, paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return prompt.to_string();
    }

    let mut result = prompt.to_string();
    result.push_str("\n\nAttachments:\n");

    for path in paths {
        // Try to get MIME type from extension
        let mime = path
            .extension()
            .and_then(|ext| match ext.to_string_lossy().as_ref() {
                "png" => Some("image/png"),
                "jpg" | "jpeg" => Some("image/jpeg"),
                "pdf" => Some("application/pdf"),
                "json" => Some("application/json"),
                "txt" => Some("text/plain"),
                "md" => Some("text/markdown"),
                "csv" => Some("text/csv"),
                "xml" => Some("application/xml"),
                "html" | "htm" => Some("text/html"),
                _ => None,
            })
            .unwrap_or("application/octet-stream");

        result.push_str(&format!("- {} ({})\n", path.display(), mime));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_session_id() {
        let id1 = generate_session_id();
        let id2 = generate_session_id();

        // Should be non-empty
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());

        // Should contain underscore separator
        assert!(id1.contains('_'));
        assert!(id2.contains('_'));

        // Should be hex format
        let parts: Vec<&str> = id1.split('_').collect();
        assert_eq!(parts.len(), 2);
        assert!(u128::from_str_radix(parts[0], 16).is_ok());
        assert!(u32::from_str_radix(parts[1], 16).is_ok());
    }

    #[test]
    fn test_generate_temp_filename_with_name() {
        let attachment = Attachment::local("test_image.png");
        let filename = generate_temp_filename(&attachment, 0);

        assert!(filename.starts_with("test_image_"));
        assert!(filename.ends_with(".png"));
    }

    #[test]
    fn test_generate_temp_filename_without_name() {
        let attachment =
            Attachment::in_memory_with_meta(vec![1, 2, 3], None, Some("image/png".to_string()));
        let filename = generate_temp_filename(&attachment, 5);

        assert!(filename.starts_with("attachment_5_"));
        assert!(filename.ends_with(".png"));
    }

    #[test]
    fn test_generate_temp_filename_mime_type_fallback() {
        let attachment = Attachment::in_memory_with_meta(
            vec![1, 2, 3],
            None,
            Some("application/pdf".to_string()),
        );
        let filename = generate_temp_filename(&attachment, 0);

        assert!(filename.ends_with(".pdf"));
    }

    #[test]
    fn test_format_prompt_with_attachments_empty() {
        let prompt = "Test prompt";
        let paths = vec![];
        let result = format_prompt_with_attachments(prompt, &paths);

        assert_eq!(result, "Test prompt");
    }

    #[test]
    fn test_format_prompt_with_attachments_single() {
        let prompt = "Test prompt";
        let paths = vec![PathBuf::from("/tmp/test.png")];
        let result = format_prompt_with_attachments(prompt, &paths);

        assert!(result.contains("Test prompt"));
        assert!(result.contains("Attachments:"));
        assert!(result.contains("/tmp/test.png"));
        assert!(result.contains("image/png"));
    }

    #[test]
    fn test_format_prompt_with_attachments_multiple() {
        let prompt = "Test prompt";
        let paths = vec![
            PathBuf::from("/tmp/test.png"),
            PathBuf::from("/tmp/doc.pdf"),
        ];
        let result = format_prompt_with_attachments(prompt, &paths);

        assert!(result.contains("Test prompt"));
        assert!(result.contains("Attachments:"));
        assert!(result.contains("/tmp/test.png"));
        assert!(result.contains("image/png"));
        assert!(result.contains("/tmp/doc.pdf"));
        assert!(result.contains("application/pdf"));
    }

    #[test]
    fn test_format_prompt_with_attachments_unknown_mime() {
        let prompt = "Test prompt";
        let paths = vec![PathBuf::from("/tmp/test.xyz")];
        let result = format_prompt_with_attachments(prompt, &paths);

        assert!(result.contains("application/octet-stream"));
    }

    #[tokio::test]
    async fn test_temp_attachment_dir_creation() {
        let temp = std::env::temp_dir();
        let dir = TempAttachmentDir::new(&temp, false).unwrap();

        // Directory should exist
        assert!(dir.path().exists());
        assert!(dir.path().is_dir());

        // Path should contain "llm-toolkit-attachments"
        assert!(
            dir.path()
                .to_string_lossy()
                .contains("llm-toolkit-attachments")
        );

        let path = dir.path().to_path_buf();
        drop(dir);

        // Directory should be cleaned up after drop
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn test_temp_attachment_dir_keep() {
        let temp = std::env::temp_dir();
        let path = {
            let dir = TempAttachmentDir::new(&temp, true).unwrap();
            let path = dir.path().to_path_buf();
            assert!(path.exists());
            path
            // dir is dropped here
        };

        // Directory should still exist after drop
        assert!(path.exists());

        // Clean up manually
        std::fs::remove_dir_all(&path).ok();
    }

    #[tokio::test]
    async fn test_process_attachments_in_memory() {
        let temp = std::env::temp_dir();
        let dir = TempAttachmentDir::new(&temp, false).unwrap();

        let attachments = [
            Attachment::in_memory(b"test data 1".to_vec()),
            Attachment::in_memory(b"test data 2".to_vec()),
        ];
        let attachment_refs: Vec<&Attachment> = attachments.iter().collect();

        let paths = process_attachments(&attachment_refs, dir.path())
            .await
            .unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths[0].exists());
        assert!(paths[1].exists());

        // Verify content
        let content1 = tokio::fs::read(&paths[0]).await.unwrap();
        assert_eq!(content1, b"test data 1");

        let content2 = tokio::fs::read(&paths[1]).await.unwrap();
        assert_eq!(content2, b"test data 2");
    }
}
