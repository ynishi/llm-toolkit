//! Expandable and Selectable traits for dynamic prompt expansion.
//!
//! This module provides traits and utilities for building ReAct-style agents
//! that can select actions from a set of options and expand them into prompts.

use crate::agent::Payload;
use std::fmt;

/// Trait for types that can expand into prompts dynamically.
///
/// This enables ReAct-style agent loops where actions selected by the LLM
/// can be expanded into new prompts for further execution.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::intent::Expandable;
/// use llm_toolkit::agent::Payload;
///
/// enum Action {
///     Search { query: String },
///     Calculate { expression: String },
/// }
///
/// impl Expandable for Action {
///     fn expand(&self) -> Payload {
///         match self {
///             Action::Search { query } => {
///                 Payload::from(format!("Search the web for: {}", query))
///             }
///             Action::Calculate { expression } => {
///                 Payload::from(format!("Calculate: {}", expression))
///             }
///         }
///     }
/// }
/// ```
pub trait Expandable {
    /// Expand this item into a Payload for LLM execution.
    ///
    /// The returned Payload can contain text, images, or any other content
    /// that the agent needs to process.
    fn expand(&self) -> Payload;
}

/// Trait for selectable items that can be chosen by an LLM.
///
/// Types implementing this trait can be registered in a `SelectionRegistry`
/// and presented to the LLM as available options. When selected, they can
/// be expanded into prompts using the `Expandable` trait.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::intent::{Selectable, Expandable};
/// use llm_toolkit::agent::Payload;
///
/// enum Tool {
///     WebSearch,
///     Calculator,
/// }
///
/// impl Selectable for Tool {
///     fn selection_id(&self) -> &str {
///         match self {
///             Tool::WebSearch => "web_search",
///             Tool::Calculator => "calculator",
///         }
///     }
///
///     fn description(&self) -> &str {
///         match self {
///             Tool::WebSearch => "Search the web for information",
///             Tool::Calculator => "Perform mathematical calculations",
///         }
///     }
/// }
///
/// impl Expandable for Tool {
///     fn expand(&self) -> Payload {
///         match self {
///             Tool::WebSearch => Payload::from("Searching the web..."),
///             Tool::Calculator => Payload::from("Calculating..."),
///         }
///     }
/// }
/// ```
pub trait Selectable: Expandable {
    /// Get the unique identifier for this selectable item.
    ///
    /// This ID is used by the LLM to select the item and by the registry
    /// to look up the item when selected.
    fn selection_id(&self) -> &str;

    /// Get a human-readable description of what this item does.
    ///
    /// This description is presented to the LLM to help it understand
    /// when to select this item.
    fn description(&self) -> &str;
}

/// Registry for managing selectable items.
///
/// The registry maintains a collection of items that implement `Selectable`
/// and provides utilities for presenting them to LLMs and looking them up
/// by their selection ID.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::intent::{SelectionRegistry, Selectable, Expandable};
///
/// let mut registry = SelectionRegistry::new();
/// registry.register(Tool::WebSearch);
/// registry.register(Tool::Calculator);
///
/// // Generate prompt section for LLM
/// let prompt_section = registry.to_prompt_section();
///
/// // Look up selected item
/// if let Some(tool) = registry.get("web_search") {
///     let expanded = tool.expand();
///     // ... use expanded payload
/// }
/// ```
pub struct SelectionRegistry<T: Selectable> {
    items: Vec<T>,
}

impl<T: Selectable> SelectionRegistry<T> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Register a new selectable item.
    ///
    /// # Panics
    ///
    /// Panics if an item with the same selection_id is already registered.
    pub fn register(&mut self, item: T) -> &mut Self {
        let id = item.selection_id();
        if self.items.iter().any(|i| i.selection_id() == id) {
            panic!("Item with id '{}' is already registered", id);
        }
        self.items.push(item);
        self
    }

    /// Try to register a new selectable item.
    ///
    /// Returns `Err` if an item with the same selection_id is already registered.
    pub fn try_register(&mut self, item: T) -> Result<&mut Self, RegistryError> {
        let id = item.selection_id().to_string();
        if self.items.iter().any(|i| i.selection_id() == id) {
            return Err(RegistryError::DuplicateId { id });
        }
        self.items.push(item);
        Ok(self)
    }

    /// Get a reference to an item by its selection ID.
    pub fn get(&self, id: &str) -> Option<&T> {
        self.items.iter().find(|item| item.selection_id() == id)
    }

    /// Get all registered items.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Generate a prompt section listing all selectable items.
    ///
    /// This section can be included in prompts to inform the LLM about
    /// available options.
    ///
    /// # Format
    ///
    /// The output is formatted as a Markdown list:
    /// ```text
    /// ## Available Actions
    ///
    /// - `action_id`: Description of the action
    /// - `another_action`: Description of another action
    /// ```
    pub fn to_prompt_section(&self) -> String {
        self.to_prompt_section_with_title("Available Actions")
    }

    /// Generate a prompt section with a custom title.
    pub fn to_prompt_section_with_title(&self, title: &str) -> String {
        let mut output = format!("## {}\n\n", title);
        for item in &self.items {
            output.push_str(&format!(
                "- `{}`: {}\n",
                item.selection_id(),
                item.description()
            ));
        }
        output
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the number of registered items.
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T: Selectable> Default for SelectionRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Selectable> fmt::Debug for SelectionRegistry<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectionRegistry")
            .field("items", &self.items)
            .finish()
    }
}

/// Errors that can occur when working with SelectionRegistry.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Item with id '{id}' is already registered")]
    DuplicateId { id: String },

    #[error("Item with id '{id}' not found in registry")]
    NotFound { id: String },
}

/// Errors that can occur during ReAct loop execution.
#[derive(Debug, thiserror::Error)]
pub enum ReActError {
    #[error("Agent error: {0}")]
    Agent(#[from] crate::agent::AgentError),

    #[error("Selection not found: {0}")]
    SelectionNotFound(String),

    #[error("Max iterations ({0}) reached without completion")]
    MaxIterationsReached(usize),

    #[error("Failed to extract selection from response: {0}")]
    ExtractionFailed(String),
}

/// Result of a ReAct loop iteration.
#[derive(Debug, Clone, PartialEq)]
pub enum ReActResult {
    /// The task is complete with the final response
    Complete(String),

    /// Continue to the next iteration with updated context
    Continue { context: String },
}

/// Configuration for ReAct loop execution.
#[derive(Debug, Clone)]
pub struct ReActConfig {
    /// Maximum number of iterations before giving up
    pub max_iterations: usize,

    /// Whether to include the selection prompt in the context
    pub include_selection_prompt: bool,

    /// Custom completion marker (defaults to "DONE")
    pub completion_marker: String,

    /// Whether to accumulate all results in context
    pub accumulate_results: bool,
}

impl Default for ReActConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            include_selection_prompt: true,
            completion_marker: "DONE".to_string(),
            accumulate_results: true,
        }
    }
}

impl ReActConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set whether to include the selection prompt in context
    pub fn with_include_selection_prompt(mut self, include: bool) -> Self {
        self.include_selection_prompt = include;
        self
    }

    /// Set a custom completion marker
    pub fn with_completion_marker(mut self, marker: impl Into<String>) -> Self {
        self.completion_marker = marker.into();
        self
    }

    /// Set whether to accumulate all results
    pub fn with_accumulate_results(mut self, accumulate: bool) -> Self {
        self.accumulate_results = accumulate;
        self
    }
}

/// Execute a ReAct-style loop with action selection and expansion.
///
/// This function implements the ReAct (Reasoning + Acting) pattern:
/// 1. Present available actions to the LLM
/// 2. LLM selects an action or indicates completion
/// 3. Expand the selected action into a prompt
/// 4. Execute the expanded prompt
/// 5. Accumulate results and repeat
///
/// # Type Parameters
///
/// - `T`: Type implementing Selectable (usually an enum with actions)
/// - `A`: Agent that executes prompts and returns String responses
/// - `F`: Function that extracts the selected action ID from LLM response
///
/// # Arguments
///
/// - `agent`: The agent that will execute prompts
/// - `registry`: Registry containing available selectable actions
/// - `initial_task`: The initial task description
/// - `selector`: Function to extract action ID from LLM response.
///               Returns `Ok(None)` when task is complete, `Ok(Some(id))` when action selected.
/// - `config`: Configuration for the ReAct loop
///
/// # Returns
///
/// Returns the final response when the task is complete, or an error if
/// max iterations reached or other failure occurs.
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::intent::expandable::{react_loop, ReActConfig};
///
/// // Define a selector function that extracts action IDs
/// let selector = |response: &str| {
///     if response.contains("DONE") {
///         Ok(None)  // Task complete
///     } else if let Some(id) = extract_tag(response, "action") {
///         Ok(Some(id))  // Action selected
///     } else {
///         Err(ReActError::ExtractionFailed("No action found".into()))
///     }
/// };
///
/// let result = react_loop(
///     &agent,
///     &registry,
///     "Solve this problem",
///     selector,
///     ReActConfig::default(),
/// ).await?;
/// ```
pub async fn react_loop<T, A, F>(
    agent: &A,
    registry: &SelectionRegistry<T>,
    initial_task: impl Into<Payload>,
    selector: F,
    config: ReActConfig,
) -> Result<String, ReActError>
where
    T: Selectable + Clone,
    A: crate::agent::Agent<Output = String>,
    F: Fn(&str) -> Result<Option<String>, ReActError>,
{
    let mut context = initial_task.into().to_text();

    for _iteration in 0..config.max_iterations {
        // 1. Build prompt with available actions
        let mut prompt = String::new();

        if config.include_selection_prompt {
            prompt.push_str(&registry.to_prompt_section());
            prompt.push_str("\n\n");
        }

        prompt.push_str(&format!(
            "Current context:\n{}\n\nSelect an action or respond with '{}' if the task is complete.",
            context, config.completion_marker
        ));

        // 2. Get LLM response
        let response = agent.execute(Payload::from(prompt)).await?;

        // 3. Extract selected action ID (or check for completion)
        match selector(&response)? {
            None => {
                // Task complete
                return Ok(response);
            }
            Some(action_id) => {
                // 4. Get the selected item and expand it
                let item = registry
                    .get(&action_id)
                    .ok_or_else(|| ReActError::SelectionNotFound(action_id.clone()))?;

                let expanded = item.expand();

                // 5. Execute the expanded action
                let result = agent.execute(expanded).await?;

                // 6. Update context
                if config.accumulate_results {
                    context = format!("{}\n\n[Action: {}]\nResult: {}", context, action_id, result);
                } else {
                    context = result;
                }
            }
        }
    }

    Err(ReActError::MaxIterationsReached(config.max_iterations))
}

/// Helper function to create a simple selector based on a tag extractor.
///
/// This creates a selector function that:
/// - Returns `Ok(None)` if the completion marker is found
/// - Returns `Ok(Some(id))` if an action tag is found
/// - Returns an error if neither is found
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::intent::expandable::simple_tag_selector;
///
/// let selector = simple_tag_selector("action", "DONE");
/// ```
pub fn simple_tag_selector(
    tag: &'static str,
    completion_marker: &'static str,
) -> impl Fn(&str) -> Result<Option<String>, ReActError> {
    move |response: &str| {
        // Check for completion first
        if response.contains(completion_marker) {
            return Ok(None);
        }

        // Try to extract action tag
        use crate::extract::FlexibleExtractor;
        use crate::extract::core::ContentExtractor;

        let extractor = FlexibleExtractor::new();
        if let Some(action_id) = extractor.extract_tagged(response, tag) {
            Ok(Some(action_id))
        } else {
            Err(ReActError::ExtractionFailed(format!(
                "No <{}> tag or '{}' found in response",
                tag, completion_marker
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    enum TestAction {
        Greet { name: String },
        Calculate { expr: String },
    }

    impl Expandable for TestAction {
        fn expand(&self) -> Payload {
            match self {
                TestAction::Greet { name } => Payload::from(format!("Say hello to {}", name)),
                TestAction::Calculate { expr } => Payload::from(format!("Calculate: {}", expr)),
            }
        }
    }

    impl Selectable for TestAction {
        fn selection_id(&self) -> &str {
            match self {
                TestAction::Greet { .. } => "greet",
                TestAction::Calculate { .. } => "calculate",
            }
        }

        fn description(&self) -> &str {
            match self {
                TestAction::Greet { .. } => "Greet a person by name",
                TestAction::Calculate { .. } => "Perform a calculation",
            }
        }
    }

    #[test]
    fn test_expandable() {
        let action = TestAction::Greet {
            name: "Alice".to_string(),
        };
        let payload = action.expand();
        assert_eq!(payload.to_text(), "Say hello to Alice");
    }

    #[test]
    fn test_selectable() {
        let action = TestAction::Greet {
            name: "Bob".to_string(),
        };
        assert_eq!(action.selection_id(), "greet");
        assert_eq!(action.description(), "Greet a person by name");
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = SelectionRegistry::new();
        registry.register(TestAction::Greet {
            name: "Charlie".to_string(),
        });
        registry.register(TestAction::Calculate {
            expr: "2+2".to_string(),
        });

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());

        let greet = registry.get("greet").unwrap();
        assert_eq!(greet.selection_id(), "greet");
    }

    #[test]
    fn test_registry_to_prompt_section() {
        let mut registry = SelectionRegistry::new();
        registry.register(TestAction::Greet {
            name: "Dave".to_string(),
        });
        registry.register(TestAction::Calculate {
            expr: "5*5".to_string(),
        });

        let section = registry.to_prompt_section();
        assert!(section.contains("## Available Actions"));
        assert!(section.contains("- `greet`: Greet a person by name"));
        assert!(section.contains("- `calculate`: Perform a calculation"));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_registry_duplicate_panic() {
        let mut registry = SelectionRegistry::new();
        registry.register(TestAction::Greet {
            name: "Eve".to_string(),
        });
        registry.register(TestAction::Greet {
            name: "Frank".to_string(),
        });
    }

    #[test]
    fn test_registry_try_register_duplicate() {
        let mut registry = SelectionRegistry::new();
        registry
            .try_register(TestAction::Greet {
                name: "Grace".to_string(),
            })
            .unwrap();

        let result = registry.try_register(TestAction::Greet {
            name: "Heidi".to_string(),
        });
        assert!(matches!(result, Err(RegistryError::DuplicateId { .. })));
    }
}
