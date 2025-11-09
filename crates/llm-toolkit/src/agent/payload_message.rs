use serde::{Deserialize, Serialize};

use super::dialogue::message::{DialogueMessage, MessageMetadata};
use super::dialogue::{ParticipantInfo, Speaker};

/// Lightweight message record used when constructing payloads and tracking history.
///
/// This replaces the ad-hoc `(Speaker, String)` tuple usage to provide a named
/// type for structured dialogue messages that do not require the additional
/// metadata stored in `DialogueMessage`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PayloadMessage {
    /// Speaker for this message.
    pub speaker: Speaker,
    /// Message content.
    pub content: String,
    /// Optional metadata for controlling reaction behavior.
    #[serde(default, skip_serializing_if = "is_metadata_default")]
    pub metadata: MessageMetadata,
}

/// Helper function to skip serializing default metadata.
fn is_metadata_default(metadata: &MessageMetadata) -> bool {
    metadata.token_count.is_none()
        && !metadata.has_attachments
        && metadata.message_type.is_none()
        && metadata.custom.is_empty()
}

impl PayloadMessage {
    /// Creates a new payload message.
    pub fn new(speaker: Speaker, content: impl Into<String>) -> Self {
        Self {
            speaker,
            content: content.into(),
            metadata: MessageMetadata::default(),
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
            metadata: MessageMetadata::default(),
        }
    }

    /// Creates a payload message for a system speaker.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            speaker: Speaker::System,
            content: content.into(),
            metadata: MessageMetadata::default(),
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
            metadata: MessageMetadata::default(),
        }
    }

    /// Attaches metadata to this message.
    pub fn with_metadata(mut self, metadata: MessageMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns the relation of this message's speaker to the provided name.
    pub fn relation_to(&self, self_name: &str) -> SpeakerRelation {
        if self.speaker.name() == self_name {
            SpeakerRelation::Self_
        } else {
            SpeakerRelation::Other
        }
    }

    /// Attaches a relation flag to this message, producing a related message view.
    pub fn with_relation(self, relation: SpeakerRelation) -> RelatedPayloadMessage {
        RelatedPayloadMessage::new(self, relation)
    }
}

impl From<&DialogueMessage> for PayloadMessage {
    fn from(msg: &DialogueMessage) -> Self {
        PayloadMessage {
            speaker: msg.speaker.clone(),
            content: msg.content.clone(),
            metadata: msg.metadata.clone(),
        }
    }
}

/// Speaker relationship from the perspective of a viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerRelation {
    /// Message authored by the viewing agent.
    Self_,
    /// Message authored by an allied participant.
    Teammate,
    /// Message authored by an external participant.
    Other,
}

impl SpeakerRelation {
    /// Optional suffix for display purposes.
    pub fn suffix(self) -> Option<&'static str> {
        match self {
            SpeakerRelation::Self_ => Some("YOU"),
            SpeakerRelation::Teammate => Some("ALLY"),
            SpeakerRelation::Other => None,
        }
    }
}

/// Payload message annotated with speaker relation metadata.
#[derive(Debug, Clone)]
pub struct RelatedPayloadMessage {
    pub message: PayloadMessage,
    pub relation: SpeakerRelation,
}

impl RelatedPayloadMessage {
    pub fn new(message: PayloadMessage, relation: SpeakerRelation) -> Self {
        Self { message, relation }
    }

    /// Returns a human-readable label including relation suffix if available.
    pub fn display_label(&self) -> String {
        match self.relation.suffix() {
            Some(suffix) => format!("{} ({})", self.message.speaker.name(), suffix),
            None => self.message.speaker.name().to_string(),
        }
    }

    fn is_suitable_for_banner(&self, content_count: usize) -> bool {
        content_count > 1000
    }

    pub fn select_format(&self, content_count: usize) -> String {
        if self.is_suitable_for_banner(content_count) {
            self.format_banner()
        } else {
            self.format_line()
        }
    }

    /// Formats the message as a single line `[Speaker]: content`.
    pub fn format_line(&self) -> String {
        format!("[{}]: {}", self.display_label(), self.message.content)
    }

    /// Formats a banner style header useful for long-form sections.
    pub fn format_banner(&self) -> String {
        format!(
            "======= {} =======\n{}\n==================\n",
            self.display_label(),
            self.message.content
        )
    }
}

/// Participant annotated with relation metadata.
#[derive(Debug, Clone)]
pub struct RelatedParticipant {
    pub participant: ParticipantInfo,
    pub relation: SpeakerRelation,
}

impl RelatedParticipant {
    pub fn new(participant: ParticipantInfo, relation: SpeakerRelation) -> Self {
        Self {
            participant,
            relation,
        }
    }

    /// Returns a display label similar to `display_label` for messages.
    pub fn display_label(&self) -> String {
        match self.relation.suffix() {
            Some(suffix) => format!("{} ({})", self.participant.name, suffix),
            None => self.participant.name.clone(),
        }
    }

    /// Formats participant for bullet-list presentation.
    pub fn format_line(&self) -> String {
        format!(
            "- **{}** - {}: {}",
            self.display_label(),
            self.participant.role,
            self.participant.description
        )
    }
}

/// Derives relation for a participant relative to `self_name`.
pub fn participant_relation(participant: &ParticipantInfo, self_name: &str) -> SpeakerRelation {
    if participant.name == self_name {
        SpeakerRelation::Self_
    } else {
        SpeakerRelation::Teammate
    }
}

/// Formats messages with speaker information and YOU/ME marking.
///
/// Each message is formatted as "[Speaker]: content" or "[Speaker (YOU)]: content"
/// based on whether the speaker is the current persona.
fn relate_messages<'a>(
    messages: impl IntoIterator<Item = &'a PayloadMessage>,
    self_name: &str,
) -> Vec<RelatedPayloadMessage> {
    messages
        .into_iter()
        .cloned()
        .map(|message| {
            let relation = message.relation_to(self_name);
            message.with_relation(relation)
        })
        .collect()
}

pub fn format_messages_with_relation(
    messages: &[PayloadMessage],
    self_name: &str,
    total_content_count: usize,
) -> String {
    relate_messages(messages.iter(), self_name)
        .into_iter()
        .map(|related| related.select_format(total_content_count))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_to_detects_self() {
        let message = PayloadMessage::agent("Alice", "Engineer", "Hello");
        assert_eq!(message.relation_to("Alice"), SpeakerRelation::Self_);
        assert_eq!(message.relation_to("Bob"), SpeakerRelation::Other);
    }

    #[test]
    fn related_payload_message_formats() {
        let message = PayloadMessage::user("Carol", "PM", "Status update");
        let related = message.with_relation(SpeakerRelation::Teammate);
        assert_eq!(related.display_label(), "Carol (ALLY)");
        assert_eq!(related.format_line(), "[Carol (ALLY)]: Status update");
        let actual_banner = related.format_banner();
        assert!(actual_banner.contains("===== Carol (ALLY) ====="));
        assert!(actual_banner.contains("Status update"));
    }

    #[test]
    fn select_format_prefers_banner_for_large_content() {
        let long_content = "lorem ipsum dolor sit amet ".repeat(60); // > 1000 chars
        let messages = vec![PayloadMessage::system(long_content.clone())];
        let formatted = format_messages_with_relation(&messages, "Observer", long_content.len());
        assert!(
            formatted.contains("======="),
            "expected banner formatting with separators"
        );
        assert!(
            formatted.contains(&long_content),
            "expected banner to include original content"
        );
    }

    #[test]
    fn select_format_uses_line_for_short_content() {
        let messages = vec![PayloadMessage::user("Quinn", "PM", "Short update")];
        let formatted = format_messages_with_relation(&messages, "Observer", 42);
        assert_eq!(formatted, "[Quinn]: Short update");
    }

    #[test]
    fn related_participant_formats() {
        let participant = ParticipantInfo::new("Dana", "Researcher", "Focuses on UX studies");
        let related = RelatedParticipant::new(
            participant.clone(),
            participant_relation(&participant, "Dana"),
        );
        assert_eq!(related.display_label(), "Dana (YOU)");
        assert!(related.format_line().contains("Dana (YOU)"));

        let teammate = RelatedParticipant::new(
            participant.clone(),
            participant_relation(&participant, "Eli"),
        );
        assert_eq!(teammate.display_label(), "Dana (ALLY)");
        assert!(teammate.format_line().contains("Dana (ALLY)"));
    }
}
