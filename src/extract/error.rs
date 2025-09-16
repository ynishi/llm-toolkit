/// Response parsing errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    #[error("Failed to extract tags from response: {0}")]
    TagExtractionFailed(String),

    #[error("Failed to extract metadata from response: {0}")]
    JsonParsingFailed(String),

    #[error("Failed to extract content from response: {0:?}")]
    AllStrategiesFailed(Vec<String>),

    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Invalid format in response: {0}")]
    InvalidFormat(String),

    #[error("Failed to process response: {0}")]
    ProcessingFailed(String),
}
