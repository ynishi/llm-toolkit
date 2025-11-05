//! Capability definitions for agents.
//!
//! This module provides types for declaring and managing agent capabilities
//! (tools/actions) in a structured, extensible way.

use crate::prompt::ToPrompt;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a single capability (tool/action) that an agent can perform.
///
/// Capabilities are used to explicitly declare what concrete actions an agent
/// can execute, beyond its general expertise description. This enables:
///
/// - **Orchestrator precision**: Strategy generation can select agents based on
///   concrete capabilities rather than just natural language expertise.
/// - **Dialogue coordination**: Agents can discover what other participants can do.
/// - **Dynamic policy enforcement**: Dialogues can restrict which capabilities
///   are allowed in a given session.
///
/// # Examples
///
/// ```rust
/// use llm_toolkit::agent::Capability;
///
/// // Simple capability
/// let cap = Capability::new("file:write");
///
/// // With description for LLM clarity
/// let cap = Capability::new("file:write")
///     .with_description("Write content to a file on disk");
///
/// // From string slice (convenience)
/// let cap: Capability = "api:weather".into();
///
/// // From tuple (name, description)
/// let cap: Capability = ("db:query", "Query the database").into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct Capability {
    /// The capability identifier (e.g., "file:write", "api:weather", "db:query")
    ///
    /// Conventionally uses colon-separated namespacing: `category:action`
    pub name: String,

    /// Optional description for LLM understanding and human readability
    ///
    /// When present, this helps LLMs understand what the capability does
    /// and when it should be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Capability {
    /// Creates a new capability with the given name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm_toolkit::agent::Capability;
    ///
    /// let cap = Capability::new("file:read");
    /// assert_eq!(cap.name, "file:read");
    /// assert_eq!(cap.description, None);
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
        }
    }

    /// Sets the description for this capability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm_toolkit::agent::Capability;
    ///
    /// let cap = Capability::new("file:write")
    ///     .with_description("Write content to a file");
    ///
    /// assert_eq!(cap.description, Some("Write content to a file".to_string()));
    /// ```
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(desc) = &self.description {
            write!(f, "{}: {}", self.name, desc)
        } else {
            write!(f, "{}", self.name)
        }
    }
}

impl ToPrompt for Capability {
    fn to_prompt(&self) -> String {
        self.to_string()
    }

    fn prompt_schema() -> String {
        r#"Capability (tool/action identifier):
- Format: "category:action" (e.g., "file:read", "api:weather", "db:query")
- Optional description: "capability_name: human-readable description"
Examples:
  - file:read
  - file:write: Write content to a file
  - api:weather: Get current weather data"#
            .to_string()
    }
}

// Convenience conversions for ergonomic API

impl From<&str> for Capability {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl From<String> for Capability {
    fn from(name: String) -> Self {
        Self::new(name)
    }
}

impl From<(&str, &str)> for Capability {
    fn from((name, desc): (&str, &str)) -> Self {
        Self::new(name).with_description(desc)
    }
}

impl From<(String, String)> for Capability {
    fn from((name, desc): (String, String)) -> Self {
        Self::new(name).with_description(desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_new() {
        let cap = Capability::new("file:read");
        assert_eq!(cap.name, "file:read");
        assert_eq!(cap.description, None);
    }

    #[test]
    fn test_capability_with_description() {
        let cap = Capability::new("file:write").with_description("Write to file");
        assert_eq!(cap.name, "file:write");
        assert_eq!(cap.description, Some("Write to file".to_string()));
    }

    #[test]
    fn test_capability_display() {
        let cap1 = Capability::new("api:weather");
        assert_eq!(cap1.to_string(), "api:weather");

        let cap2 = Capability::new("api:weather").with_description("Get weather data");
        assert_eq!(cap2.to_string(), "api:weather: Get weather data");
    }

    #[test]
    fn test_capability_from_str() {
        let cap: Capability = "db:query".into();
        assert_eq!(cap.name, "db:query");
        assert_eq!(cap.description, None);
    }

    #[test]
    fn test_capability_from_tuple() {
        let cap: Capability = ("db:insert", "Insert records into database").into();
        assert_eq!(cap.name, "db:insert");
        assert_eq!(
            cap.description,
            Some("Insert records into database".to_string())
        );
    }

    #[test]
    fn test_capability_serialization() {
        let cap = Capability::new("file:read").with_description("Read file content");
        let json = serde_json::to_string(&cap).unwrap();
        let deserialized: Capability = serde_json::from_str(&json).unwrap();
        assert_eq!(cap, deserialized);
    }

    #[test]
    fn test_capability_eq_hash() {
        use std::collections::HashSet;

        let cap1 = Capability::new("api:call");
        let cap2 = Capability::new("api:call");
        let cap3 = Capability::new("api:call").with_description("Different desc");

        assert_eq!(cap1, cap2);
        assert_ne!(cap1, cap3); // Description matters for equality

        let mut set = HashSet::new();
        set.insert(cap1.clone());
        assert!(set.contains(&cap2));
        assert!(!set.contains(&cap3));
    }

    #[test]
    fn test_capability_to_prompt() {
        use crate::prompt::ToPrompt;

        let cap1 = Capability::new("file:read");
        assert_eq!(cap1.to_prompt(), "file:read");

        let cap2 = Capability::new("file:write").with_description("Write to file");
        assert_eq!(cap2.to_prompt(), "file:write: Write to file");
    }

    #[test]
    fn test_capability_prompt_schema() {
        use crate::prompt::ToPrompt;

        let schema = Capability::prompt_schema();
        assert!(schema.contains("Capability"));
        assert!(schema.contains("category:action"));
        assert!(schema.contains("file:read"));
        assert!(schema.contains("api:weather"));
    }
}
