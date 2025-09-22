//! Multimodal support for prompts, including image data handling.

use crate::prompt::{PromptPart, ToPrompt};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Helper structure for handling image data in prompts.
///
/// This struct provides a convenient way to represent images
/// that can be included in multimodal prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// The MIME media type (e.g., "image/jpeg", "image/png").
    pub media_type: String,
    /// The raw image data.
    pub data: Vec<u8>,
}

impl ImageData {
    /// Creates a new `ImageData` instance with the given media type and data.
    pub fn new(media_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            media_type: media_type.into(),
            data,
        }
    }

    /// Creates an `ImageData` instance from a file path.
    ///
    /// The media type is inferred from the file extension.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the media type
    /// cannot be determined from the file extension.
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;

        let media_type = match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        }
        .to_string();

        Ok(Self { media_type, data })
    }

    /// Creates an `ImageData` instance from a base64-encoded string.
    ///
    /// # Arguments
    ///
    /// * `base64_str` - The base64-encoded image data
    /// * `media_type` - The MIME media type of the image
    ///
    /// # Errors
    ///
    /// Returns an error if the base64 string cannot be decoded.
    pub fn from_base64(
        base64_str: &str,
        media_type: impl Into<String>,
    ) -> Result<Self, base64::DecodeError> {
        let data = STANDARD.decode(base64_str)?;
        Ok(Self {
            media_type: media_type.into(),
            data,
        })
    }

    /// Converts the image data to a base64-encoded string.
    pub fn to_base64(&self) -> String {
        STANDARD.encode(&self.data)
    }
}

impl ToPrompt for ImageData {
    fn to_prompt_parts(&self) -> Vec<PromptPart> {
        vec![PromptPart::Image {
            media_type: self.media_type.clone(),
            data: self.data.clone(),
        }]
    }
}

// Optional: From implementations for common image library types
// These would be behind feature flags in a real implementation
// Commented out for now as the `image` feature is not defined

// #[cfg(feature = "image")]
// impl From<image::DynamicImage> for ImageData {
//     fn from(img: image::DynamicImage) -> Self {
//         use std::io::Cursor;
//
//         let mut buffer = Vec::new();
//         let mut cursor = Cursor::new(&mut buffer);
//
//         // Default to PNG format
//         img.write_to(&mut cursor, image::ImageFormat::Png)
//             .expect("Failed to encode image");
//
//         Self {
//             media_type: "image/png".to_string(),
//             data: buffer,
//         }
//     }
// }

// From implementation for data URL strings (e.g., "data:image/png;base64,...")
impl TryFrom<&str> for ImageData {
    type Error = String;

    fn try_from(data_url: &str) -> Result<Self, Self::Error> {
        if !data_url.starts_with("data:") {
            return Err("Not a data URL".to_string());
        }

        let content = data_url.strip_prefix("data:").ok_or("Invalid data URL")?;

        let parts: Vec<&str> = content.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err("Invalid data URL format".to_string());
        }

        let media_type = parts[0]
            .split(';')
            .next()
            .unwrap_or("application/octet-stream")
            .to_string();

        let is_base64 = parts[0].contains("base64");

        let data = if is_base64 {
            STANDARD
                .decode(parts[1])
                .map_err(|e| format!("Failed to decode base64: {}", e))?
        } else {
            // URL-encoded data
            parts[1].as_bytes().to_vec()
        };

        Ok(Self { media_type, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_data_creation() {
        let data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes
        let img = ImageData::new("image/jpeg", data.clone());

        assert_eq!(img.media_type, "image/jpeg");
        assert_eq!(img.data, data);
    }

    #[test]
    fn test_image_data_to_prompt_parts() {
        let data = vec![1, 2, 3, 4];
        let img = ImageData::new("image/png", data.clone());
        let parts = img.to_prompt_parts();

        assert_eq!(parts.len(), 1);
        match &parts[0] {
            PromptPart::Image {
                media_type,
                data: img_data,
            } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(img_data, &data);
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_base64_conversion() {
        let original_data = vec![72, 101, 108, 108, 111]; // "Hello" in ASCII
        let img = ImageData::new("image/test", original_data.clone());

        let base64 = img.to_base64();
        let decoded = ImageData::from_base64(&base64, "image/test").unwrap();

        assert_eq!(decoded.data, original_data);
        assert_eq!(decoded.media_type, "image/test");
    }

    #[test]
    fn test_data_url_parsing() {
        let data_url = "data:image/png;base64,SGVsbG8="; // "Hello" in base64
        let img = ImageData::try_from(data_url).unwrap();

        assert_eq!(img.media_type, "image/png");
        assert_eq!(img.data, b"Hello");
    }
}
