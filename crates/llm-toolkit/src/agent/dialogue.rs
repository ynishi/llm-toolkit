//! Dialogue component for multi-agent conversational interactions.
//!
//! This module provides abstractions for managing turn-based dialogues between
//! multiple agents, with configurable turn-taking strategies.

use super::{Agent, AgentError, Payload};

/// Represents a single turn in the dialogue.
#[derive(Debug, Clone)]
pub struct DialogueTurn {
    pub participant_name: String,
    pub content: String,
}

/// Represents the execution model for dialogue strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionModel {
    /// All participants respond in parallel to the same input.
    Broadcast,
    /// Participants execute sequentially, with output chained as input.
    Sequential,
}

/// A trait for defining how the next speaker(s) in a dialogue are chosen.
pub trait TurnTakingStrategy: Send + Sync {
    /// Selects the next participant(s) to speak.
    ///
    /// # Arguments
    /// * `participants` - A slice of all participants in the dialogue.
    /// * `history` - The history of the conversation so far.
    ///
    /// # Returns
    /// A vector of indices of the participants who should speak next.
    fn select_next_participants(
        &mut self,
        participants: &[Box<dyn Agent<Output = String>>],
        history: &[DialogueTurn],
    ) -> Vec<usize>;

    /// Returns the execution model for this strategy.
    fn execution_model(&self) -> ExecutionModel;
}

/// A broadcast turn-taking strategy.
///
/// All participants speak in each turn, responding to the same input in parallel.
pub struct Broadcast;

impl Broadcast {
    /// Creates a new broadcast strategy.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Broadcast {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnTakingStrategy for Broadcast {
    fn select_next_participants(
        &mut self,
        participants: &[Box<dyn Agent<Output = String>>],
        _history: &[DialogueTurn],
    ) -> Vec<usize> {
        (0..participants.len()).collect()
    }

    fn execution_model(&self) -> ExecutionModel {
        ExecutionModel::Broadcast
    }
}

/// A sequential turn-taking strategy.
///
/// Participants execute in order, with the output of one becoming the input to the next.
pub struct Sequential;

impl Sequential {
    /// Creates a new sequential strategy.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnTakingStrategy for Sequential {
    fn select_next_participants(
        &mut self,
        participants: &[Box<dyn Agent<Output = String>>],
        _history: &[DialogueTurn],
    ) -> Vec<usize> {
        (0..participants.len()).collect()
    }

    fn execution_model(&self) -> ExecutionModel {
        ExecutionModel::Sequential
    }
}

/// A dialogue manager for multi-agent conversations.
///
/// The dialogue maintains a list of participants and a conversation history,
/// using a turn-taking strategy to determine execution behavior.
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::dialogue::Dialogue;
/// use llm_toolkit::agent::impls::ClaudeCodeAgent;
///
/// // Broadcast mode: all agents respond in parallel
/// let mut dialogue = Dialogue::broadcast()
///     .add_participant(ClaudeCodeAgent::new())
///     .add_participant(ClaudeCodeAgent::new());
///
/// let responses = dialogue.run("Discuss AI ethics".to_string()).await?;
///
/// // Sequential mode: output chains from one agent to the next
/// let mut dialogue = Dialogue::sequential()
///     .add_participant(ClaudeCodeAgent::new())
///     .add_participant(ClaudeCodeAgent::new());
///
/// let final_output = dialogue.run("Process this input".to_string()).await?;
/// ```
pub struct Dialogue {
    participants: Vec<Box<dyn Agent<Output = String>>>,
    history: Vec<DialogueTurn>,
    strategy: Box<dyn TurnTakingStrategy>,
}

impl Dialogue {
    /// Creates a new dialogue with the specified strategy.
    ///
    /// This is private - use `broadcast()` or `sequential()` instead.
    fn new(strategy: impl TurnTakingStrategy + 'static) -> Self {
        Self {
            participants: Vec::new(),
            history: Vec::new(),
            strategy: Box::new(strategy),
        }
    }

    /// Creates a new dialogue with broadcast execution.
    ///
    /// In broadcast mode, all participants respond in parallel to the same input.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(agent1)
    ///     .add_participant(agent2);
    /// ```
    pub fn broadcast() -> Self {
        Self::new(Broadcast::new())
    }

    /// Creates a new dialogue with sequential execution.
    ///
    /// In sequential mode, the output of one participant becomes the input to the next.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::sequential()
    ///     .add_participant(summarizer)
    ///     .add_participant(translator)
    ///     .add_participant(formatter);
    /// ```
    pub fn sequential() -> Self {
        Self::new(Sequential::new())
    }

    /// Adds a participant to the dialogue.
    ///
    /// Returns `&mut Self` to enable builder-style chaining.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(ClaudeCodeAgent::new())
    ///     .add_participant(ClaudeCodeAgent::new());
    /// ```
    pub fn add_participant(&mut self, agent: impl Agent<Output = String> + 'static) -> &mut Self {
        self.participants.push(Box::new(agent));
        self
    }

    /// Runs the dialogue with the configured execution model.
    ///
    /// The behavior depends on the execution model:
    /// - **Broadcast**: All participants respond in parallel to the full conversation history.
    ///   Returns a vector of all participant responses.
    /// - **Sequential**: Participants execute in order, with each agent's output becoming the
    ///   next agent's input. Returns a vector containing only the final agent's output.
    ///
    /// # Arguments
    ///
    /// * `initial_prompt` - The prompt to start the dialogue
    ///
    /// # Returns
    ///
    /// A vector of response strings. The structure depends on the execution model:
    /// - Broadcast: Contains responses from all participants
    /// - Sequential: Contains only the final participant's output
    ///
    /// # Errors
    ///
    /// Returns `AgentError` if any agent fails during execution.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// // Broadcast mode
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(agent1)
    ///     .add_participant(agent2);
    /// let all_responses = dialogue.run("Discuss AI ethics".to_string()).await?;
    ///
    /// // Sequential mode
    /// let mut dialogue = Dialogue::sequential()
    ///     .add_participant(summarizer)
    ///     .add_participant(translator);
    /// let final_output = dialogue.run("Process this".to_string()).await?;
    /// ```
    pub async fn run(&mut self, initial_prompt: String) -> Result<Vec<String>, AgentError> {
        // Add the initial prompt to the history
        self.history.push(DialogueTurn {
            participant_name: "System".to_string(),
            content: initial_prompt.clone(),
        });

        match self.strategy.execution_model() {
            ExecutionModel::Broadcast => {
                // Broadcast mode: all participants respond in parallel to the same input
                let history_text = self.format_history();
                let payload: Payload = history_text.into();

                // Execute all participants in parallel
                let mut tasks = Vec::new();
                for participant in &self.participants {
                    let payload_clone = payload.clone();
                    tasks.push(participant.execute(payload_clone));
                }

                // Wait for all responses
                let responses = futures::future::join_all(tasks).await;

                // Collect results and update history
                let mut results = Vec::new();
                for (idx, response_result) in responses.into_iter().enumerate() {
                    let response = response_result?;
                    let agent_name = self.participants[idx].name();

                    self.history.push(DialogueTurn {
                        participant_name: agent_name,
                        content: response.clone(),
                    });

                    results.push(response);
                }

                Ok(results)
            }
            ExecutionModel::Sequential => {
                // Sequential mode: output of one agent becomes input to the next
                let mut current_input = initial_prompt;

                for participant in &self.participants {
                    let agent_name = participant.name();
                    let payload: Payload = current_input.clone().into();

                    // Execute the agent with the current input
                    let response = participant.execute(payload).await?;

                    // Record the turn in history
                    self.history.push(DialogueTurn {
                        participant_name: agent_name,
                        content: response.clone(),
                    });

                    // Update the current input for the next agent
                    current_input = response;
                }

                // Return only the final output
                Ok(vec![current_input])
            }
        }
    }

    /// Formats the conversation history as a single string.
    ///
    /// This creates a formatted transcript of the dialogue that can be used
    /// as input for the next agent.
    fn format_history(&self) -> String {
        self.history
            .iter()
            .map(|turn| format!("[{}]: {}", turn.participant_name, turn.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns a reference to the conversation history.
    pub fn history(&self) -> &[DialogueTurn] {
        &self.history
    }

    /// Returns the number of participants in the dialogue.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // Mock agent for testing
    struct MockAgent {
        name: String,
        responses: Vec<String>,
        call_count: std::sync::Arc<std::sync::Mutex<usize>>,
    }

    impl MockAgent {
        fn new(name: impl Into<String>, responses: Vec<String>) -> Self {
            Self {
                name: name.into(),
                responses,
                call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
            }
        }
    }

    #[async_trait]
    impl Agent for MockAgent {
        type Output = String;

        fn expertise(&self) -> &str {
            "Mock agent for testing"
        }

        fn name(&self) -> String {
            self.name.clone()
        }

        async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
            let mut count = self.call_count.lock().unwrap();
            let response_idx = *count % self.responses.len();
            *count += 1;
            Ok(self.responses[response_idx].clone())
        }
    }

    #[tokio::test]
    async fn test_broadcast_strategy() {
        let mut dialogue = Dialogue::broadcast();
        dialogue
            .add_participant(MockAgent::new("Agent1", vec!["Response 1".to_string()]))
            .add_participant(MockAgent::new("Agent2", vec!["Response 2".to_string()]));

        let responses = dialogue.run("Initial prompt".to_string()).await.unwrap();

        // Should return 2 responses (one from each agent)
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0], "Response 1");
        assert_eq!(responses[1], "Response 2");

        // Check history: System + 2 agent responses
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].participant_name, "System");
        assert_eq!(dialogue.history()[1].participant_name, "Agent1");
        assert_eq!(dialogue.history()[2].participant_name, "Agent2");
    }

    #[test]
    fn test_broadcast_selects_all() {
        let mut strategy = Broadcast::new();
        let participants: Vec<Box<dyn Agent<Output = String>>> = vec![
            Box::new(MockAgent::new("A", vec![])),
            Box::new(MockAgent::new("B", vec![])),
            Box::new(MockAgent::new("C", vec![])),
        ];

        let history = vec![];

        assert_eq!(
            strategy.select_next_participants(&participants, &history),
            vec![0, 1, 2]
        );
    }

    #[tokio::test]
    async fn test_dialogue_with_no_participants() {
        let mut dialogue = Dialogue::broadcast();
        // Running with no participants should complete without error
        let result = dialogue.run("Test".to_string()).await;
        assert!(result.is_ok());
        let responses = result.unwrap();
        assert_eq!(responses.len(), 0);
        // Should only have the system message
        assert_eq!(dialogue.history().len(), 1);
    }

    #[tokio::test]
    async fn test_dialogue_format_history() {
        let mut dialogue = Dialogue::broadcast();
        dialogue.add_participant(MockAgent::new("Agent1", vec!["Hello".to_string()]));

        dialogue.run("Start".to_string()).await.unwrap();

        let formatted = dialogue.format_history();
        assert!(formatted.contains("[System]: Start"));
        assert!(formatted.contains("[Agent1]: Hello"));
    }

    #[tokio::test]
    async fn test_sequential_strategy() {
        let mut dialogue = Dialogue::sequential();

        // Create agents with distinct responses so we can track the chain
        dialogue
            .add_participant(MockAgent::new(
                "Summarizer",
                vec!["Summary: input received".to_string()],
            ))
            .add_participant(MockAgent::new(
                "Translator",
                vec!["Translated: previous output".to_string()],
            ))
            .add_participant(MockAgent::new(
                "Finalizer",
                vec!["Final output: all done".to_string()],
            ));

        let result = dialogue.run("Initial prompt".to_string()).await.unwrap();

        // Sequential mode returns only the final output
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Final output: all done");

        // Check that history contains the correct number of turns
        // 1 (System) + 3 (agents) = 4 total
        assert_eq!(dialogue.history().len(), 4);

        // Verify the history structure
        assert_eq!(dialogue.history()[0].participant_name, "System");
        assert_eq!(dialogue.history()[0].content, "Initial prompt");

        assert_eq!(dialogue.history()[1].participant_name, "Summarizer");
        assert_eq!(dialogue.history()[1].content, "Summary: input received");

        assert_eq!(dialogue.history()[2].participant_name, "Translator");
        assert_eq!(dialogue.history()[2].content, "Translated: previous output");

        assert_eq!(dialogue.history()[3].participant_name, "Finalizer");
        assert_eq!(dialogue.history()[3].content, "Final output: all done");
    }
}
