use super::dialogue::Speaker;

/// Lightweight message record used when constructing payloads and tracking history.
///
/// This replaces the ad-hoc `(Speaker, String)` tuple usage to provide a named
/// type for structured dialogue messages that do not require the additional
/// metadata stored in `DialogueMessage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadMessage {
    /// Speaker for this message.
    pub speaker: Speaker,
    /// Message content.
    pub content: String,
}

impl PayloadMessage {
    /// Creates a new payload message.
    pub fn new(speaker: Speaker, content: impl Into<String>) -> Self {
        Self {
            speaker,
            content: content.into(),
        }
    }

    /// Creates a payload message for a user speaker.
    pub fn user(
        name: impl Into<String>,
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            speaker: Speaker::user(name, role),
            content: content.into(),
        }
    }

    /// Creates a payload message for a system speaker.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            speaker: Speaker::System,
            content: content.into(),
        }
    }

    /// Creates a payload message for an agent speaker.
    pub fn agent(
        name: impl Into<String>,
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            speaker: Speaker::agent(name, role),
            content: content.into(),
        }
    }

    /// Formats the message as `[Speaker]: content` for history text.
    pub fn format(&self) -> String {
        format!("[{}]: {}", self.speaker.name(), self.content)
    }
}
