//! Dialogue component for multi-agent conversational interactions.
//!
//! This module provides abstractions for managing turn-based dialogues between
//! multiple agents, with configurable turn-taking strategies.

use super::chat::Chat;
use super::{Agent, AgentError, Payload};
use crate::ToPrompt;
use crate::agent::persona::{Persona, PersonaTeam, PersonaTeamGenerationRequest};
use serde::{Deserialize, Serialize};

/// Blueprint for creating a Dialogue.
///
/// Provides a high-level description of the dialogue setup, including
/// the agenda, context, optional participants, and execution strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueBlueprint {
    /// The agenda or topic for the dialogue
    pub agenda: String,

    /// Contextual information about the dialogue
    pub context: String,

    /// Optional pre-defined participants
    pub participants: Option<Vec<Persona>>,

    /// Optional execution strategy hint ("broadcast" or "sequential")
    pub execution_strategy: Option<String>,
}

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

/// Internal representation of a dialogue participant.
///
/// Wraps a persona and its associated agent implementation.
struct Participant {
    persona: Persona,
    agent: Box<dyn Agent<Output = String>>,
}

impl Participant {
    /// Returns the name of the participant from their persona.
    fn name(&self) -> &str {
        &self.persona.name
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
///     .add_participant(persona1, ClaudeCodeAgent::new())
///     .add_participant(persona2, ClaudeCodeAgent::new());
///
/// let responses = dialogue.run("Discuss AI ethics".to_string()).await?;
///
/// // Sequential mode: output chains from one agent to the next
/// let mut dialogue = Dialogue::sequential()
///     .add_participant(persona1, ClaudeCodeAgent::new())
///     .add_participant(persona2, ClaudeCodeAgent::new());
///
/// let final_output = dialogue.run("Process this input".to_string()).await?;
/// ```
pub struct Dialogue {
    participants: Vec<Participant>,
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
    ///     .add_participant(persona1, summarizer)
    ///     .add_participant(persona2, translator)
    ///     .add_participant(persona3, formatter);
    /// ```
    pub fn sequential() -> Self {
        Self::new(Sequential::new())
    }

    /// Creates a Dialogue from a blueprint.
    ///
    /// If the blueprint contains pre-defined participants, they are used directly.
    /// Otherwise, an LLM generates a team of personas based on the blueprint's context.
    ///
    /// # Arguments
    ///
    /// * `blueprint` - The dialogue blueprint containing agenda, context, and optional participants
    /// * `generator_agent` - LLM agent for generating personas (used only if blueprint.participants is None)
    /// * `dialogue_agent` - LLM agent for the actual dialogue interactions
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::{Dialogue, DialogueBlueprint};
    /// use llm_toolkit::agent::impls::{ClaudeCodeAgent, ClaudeCodeJsonAgent};
    ///
    /// // Create blueprint with auto-generated team
    /// let blueprint = DialogueBlueprint {
    ///     agenda: "1on1 Feature Planning".to_string(),
    ///     context: "Product planning meeting for new 1on1 feature in HR SaaS".to_string(),
    ///     participants: None,  // Will be auto-generated
    ///     execution_strategy: Some("broadcast".to_string()),
    /// };
    ///
    /// let mut dialogue = Dialogue::from_blueprint(
    ///     blueprint,
    ///     ClaudeCodeJsonAgent::new(),  // For team generation
    ///     ClaudeCodeAgent::new(),       // For dialogue
    /// ).await?;
    /// ```
    pub async fn from_blueprint<G, D>(
        blueprint: DialogueBlueprint,
        generator_agent: G,
        dialogue_agent: D,
    ) -> Result<Self, AgentError>
    where
        G: Agent<Output = PersonaTeam>,
        D: Agent<Output = String> + Clone + 'static,
    {
        // Determine execution strategy from blueprint
        let strategy: Box<dyn TurnTakingStrategy> = match blueprint.execution_strategy.as_deref() {
            Some("sequential") => Box::new(Sequential::new()),
            _ => Box::new(Broadcast::new()),
        };

        let mut dialogue = Self {
            participants: Vec::new(),
            history: Vec::new(),
            strategy,
        };

        // Use provided participants or generate them
        let personas = match blueprint.participants {
            Some(personas) => personas,
            None => {
                // Generate PersonaTeam using LLM
                let request = PersonaTeamGenerationRequest::new(blueprint.context);
                let prompt = request.to_prompt();
                let team = generator_agent.execute(prompt.into()).await?;
                team.personas
            }
        };

        // Build participants from personas
        for persona in personas {
            let chat_agent = Chat::new(dialogue_agent.clone())
                .with_persona(persona.clone())
                .with_history(true)
                .build();

            dialogue.participants.push(Participant {
                persona,
                agent: Box::new(chat_agent),
            });
        }

        Ok(dialogue)
    }

    /// Creates a Dialogue from a pre-generated PersonaTeam.
    ///
    /// This is useful for loading and reusing persona teams across different tasks.
    ///
    /// # Arguments
    ///
    /// * `team` - The PersonaTeam to create participants from
    /// * `llm_agent` - The base LLM agent to use for all participants
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::persona::PersonaTeam;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// // Load team from JSON
    /// let team = PersonaTeam::load("teams/dev_team.json")?;
    ///
    /// // Create dialogue
    /// let mut dialogue = Dialogue::from_persona_team(
    ///     team,
    ///     ClaudeCodeAgent::new(),
    /// )?;
    ///
    /// let result = dialogue.run("Discuss API design").await?;
    /// ```
    pub fn from_persona_team<T>(team: PersonaTeam, llm_agent: T) -> Result<Self, AgentError>
    where
        T: Agent<Output = String> + Clone + 'static,
    {
        // Determine execution strategy from team hint
        let strategy: Box<dyn TurnTakingStrategy> = match team.execution_strategy.as_deref() {
            Some("sequential") => Box::new(Sequential::new()),
            _ => Box::new(Broadcast::new()),
        };

        let mut dialogue = Self {
            participants: Vec::new(),
            history: Vec::new(),
            strategy,
        };

        // Build participants from personas
        for persona in team.personas {
            let chat_agent = Chat::new(llm_agent.clone())
                .with_persona(persona.clone())
                .with_history(true)
                .build();

            dialogue.participants.push(Participant {
                persona,
                agent: Box::new(chat_agent),
            });
        }

        Ok(dialogue)
    }

    /// Adds a participant to the dialogue dynamically.
    ///
    /// Unlike StrategyMap (which has a fixed execution plan), Dialogue
    /// supports adding participants at runtime for flexible conversation scenarios.
    ///
    /// # Arguments
    ///
    /// * `persona` - The persona to add
    /// * `llm_agent` - The LLM agent to use for this participant
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::persona::Persona;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// let mut dialogue = Dialogue::broadcast();
    /// // ... initial setup ...
    ///
    /// // Mid-discussion: bring in a domain expert
    /// let expert = Persona {
    ///     name: "Dr. Smith".to_string(),
    ///     role: "Security Consultant".to_string(),
    ///     background: "20 years in enterprise security...".to_string(),
    ///     communication_style: "Detail-oriented and cautious...".to_string(),
    /// };
    ///
    /// dialogue.add_participant(expert, ClaudeCodeAgent::new());
    /// ```
    pub fn add_participant<T>(&mut self, persona: Persona, llm_agent: T) -> &mut Self
    where
        T: Agent<Output = String> + 'static,
    {
        let chat_agent = Chat::new(llm_agent)
            .with_persona(persona.clone())
            .with_history(true)
            .build();

        self.participants.push(Participant {
            persona,
            agent: Box::new(chat_agent),
        });

        self
    }

    /// Removes a participant from the dialogue by name.
    ///
    /// This is useful for guest participants who are only needed for specific
    /// parts of the conversation.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the participant to remove
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or an error if the participant is not found.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    /// use llm_toolkit::agent::persona::Persona;
    /// use llm_toolkit::agent::impls::ClaudeCodeAgent;
    ///
    /// let mut dialogue = Dialogue::from_persona_team(team, llm)?;
    ///
    /// // Add a guest customer success manager
    /// let cs_guest = Persona {
    ///     name: "CS Manager".to_string(),
    ///     /* ... */
    /// };
    /// dialogue.add_participant(cs_guest, ClaudeCodeAgent::new());
    /// assert_eq!(dialogue.participant_count(), 6); // 5 core + 1 guest
    ///
    /// // Get customer feedback
    /// let feedback = dialogue.run("Review feature UX from customer perspective").await?;
    ///
    /// // Remove guest after their input
    /// dialogue.remove_participant("CS Manager")?;
    /// assert_eq!(dialogue.participant_count(), 5); // Back to core team
    ///
    /// // Continue with core team
    /// let next_steps = dialogue.run("Define implementation plan").await?;
    /// ```
    pub fn remove_participant(&mut self, name: &str) -> Result<(), AgentError> {
        let position = self
            .participants
            .iter()
            .position(|p| p.name() == name)
            .ok_or_else(|| {
                AgentError::ExecutionFailed(format!(
                    "Cannot remove participant '{}': participant not found",
                    name
                ))
            })?;

        self.participants.remove(position);
        Ok(())
    }

    /// Runs the dialogue with the configured execution model.
    ///
    /// The behavior depends on the execution model:
    /// - **Broadcast**: All participants respond in parallel to the full conversation history.
    ///   Returns a vector of dialogue turns from all participants.
    /// - **Sequential**: Participants execute in order, with each agent's output becoming the
    ///   next agent's input. Returns a vector containing only the final agent's turn.
    ///
    /// # Arguments
    ///
    /// * `initial_prompt` - The prompt to start the dialogue
    ///
    /// # Returns
    ///
    /// A vector of `DialogueTurn` containing participant names and their responses.
    /// The structure depends on the execution model:
    /// - Broadcast: Contains turns from all participants
    /// - Sequential: Contains only the final participant's turn
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
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    /// let all_turns = dialogue.run("Discuss AI ethics".to_string()).await?;
    ///
    /// // Sequential mode
    /// let mut dialogue = Dialogue::sequential()
    ///     .add_participant(persona1, summarizer)
    ///     .add_participant(persona2, translator);
    /// let final_turn = dialogue.run("Process this".to_string()).await?;
    /// ```
    pub async fn run(&mut self, initial_prompt: String) -> Result<Vec<DialogueTurn>, AgentError> {
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
                    tasks.push(participant.agent.execute(payload_clone));
                }

                // Wait for all responses
                let responses = futures::future::join_all(tasks).await;

                // Collect results and update history
                let mut turns = Vec::new();
                for (idx, response_result) in responses.into_iter().enumerate() {
                    let response = response_result?;
                    let participant_name = self.participants[idx].name().to_string();

                    let turn = DialogueTurn {
                        participant_name: participant_name.clone(),
                        content: response,
                    };

                    self.history.push(turn.clone());
                    turns.push(turn);
                }

                Ok(turns)
            }
            ExecutionModel::Sequential => {
                // Sequential mode: output of one agent becomes input to the next
                let mut current_input = initial_prompt;
                let mut final_turn = None;

                for participant in &self.participants {
                    let participant_name = participant.name().to_string();
                    let payload: Payload = current_input.clone().into();

                    // Execute the agent with the current input
                    let response = participant.agent.execute(payload).await?;

                    // Record the turn in history
                    let turn = DialogueTurn {
                        participant_name,
                        content: response.clone(),
                    };

                    self.history.push(turn.clone());
                    final_turn = Some(turn);

                    // Update the current input for the next agent
                    current_input = response;
                }

                // Return only the final turn
                Ok(final_turn.into_iter().collect())
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

    /// Returns references to the personas of all participants.
    ///
    /// This provides access to participant information such as names, roles,
    /// backgrounds, and communication styles without exposing internal agent implementations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    ///
    /// let personas = dialogue.participants();
    /// for persona in personas {
    ///     println!("Participant: {} ({})", persona.name, persona.role);
    /// }
    /// ```
    pub fn participants(&self) -> Vec<&Persona> {
        self.participants.iter().map(|p| &p.persona).collect()
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
    #[derive(Clone)]
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
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona1 = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent 1".to_string(),
            communication_style: "Direct".to_string(),
        };

        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Tester".to_string(),
            background: "Test agent 2".to_string(),
            communication_style: "Direct".to_string(),
        };

        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Agent1", vec!["Response 1".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new("Agent2", vec!["Response 2".to_string()]),
            );

        let turns = dialogue.run("Initial prompt".to_string()).await.unwrap();

        // Should return 2 turns (one from each participant)
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].participant_name, "Agent1");
        assert_eq!(turns[0].content, "Response 1");
        assert_eq!(turns[1].participant_name, "Agent2");
        assert_eq!(turns[1].content, "Response 2");

        // Check history: System + 2 participant responses
        assert_eq!(dialogue.history().len(), 3);
        assert_eq!(dialogue.history()[0].participant_name, "System");
        assert_eq!(dialogue.history()[1].participant_name, "Agent1");
        assert_eq!(dialogue.history()[2].participant_name, "Agent2");
    }

    #[tokio::test]
    async fn test_dialogue_with_no_participants() {
        let mut dialogue = Dialogue::broadcast();
        // Running with no participants should complete without error
        let result = dialogue.run("Test".to_string()).await;
        assert!(result.is_ok());
        let turns = result.unwrap();
        assert_eq!(turns.len(), 0);
        // Should only have the system message
        assert_eq!(dialogue.history().len(), 1);
    }

    #[tokio::test]
    async fn test_dialogue_format_history() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();
        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test agent".to_string(),
            communication_style: "Direct".to_string(),
        };

        dialogue.add_participant(persona, MockAgent::new("Agent1", vec!["Hello".to_string()]));

        dialogue.run("Start".to_string()).await.unwrap();

        let formatted = dialogue.format_history();
        assert!(formatted.contains("[System]: Start"));
        assert!(formatted.contains("[Agent1]: Hello"));
    }

    #[tokio::test]
    async fn test_sequential_strategy() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::sequential();

        let persona1 = Persona {
            name: "Summarizer".to_string(),
            role: "Summarizer".to_string(),
            background: "Summarizes inputs".to_string(),
            communication_style: "Concise".to_string(),
        };

        let persona2 = Persona {
            name: "Translator".to_string(),
            role: "Translator".to_string(),
            background: "Translates content".to_string(),
            communication_style: "Formal".to_string(),
        };

        let persona3 = Persona {
            name: "Finalizer".to_string(),
            role: "Finalizer".to_string(),
            background: "Finalizes output".to_string(),
            communication_style: "Professional".to_string(),
        };

        // Create agents with distinct responses so we can track the chain
        dialogue
            .add_participant(
                persona1,
                MockAgent::new("Summarizer", vec!["Summary: input received".to_string()]),
            )
            .add_participant(
                persona2,
                MockAgent::new(
                    "Translator",
                    vec!["Translated: previous output".to_string()],
                ),
            )
            .add_participant(
                persona3,
                MockAgent::new("Finalizer", vec!["Final output: all done".to_string()]),
            );

        let turns = dialogue.run("Initial prompt".to_string()).await.unwrap();

        // Sequential mode returns only the final turn
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].participant_name, "Finalizer");
        assert_eq!(turns[0].content, "Final output: all done");

        // Check that history contains the correct number of turns
        // 1 (System) + 3 (participants) = 4 total
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

    #[tokio::test]
    async fn test_from_persona_team_broadcast() {
        use crate::agent::persona::{Persona, PersonaTeam};

        let mut team = PersonaTeam::new("Test Team".to_string(), "Testing scenario".to_string());
        team.add_persona(Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Senior engineer".to_string(),
            communication_style: "Technical".to_string(),
        });
        team.add_persona(Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
        });
        team.execution_strategy = Some("broadcast".to_string());

        let llm = MockAgent::new("Mock", vec!["Response".to_string()]);
        let mut dialogue = Dialogue::from_persona_team(team, llm).unwrap();

        assert_eq!(dialogue.participant_count(), 2);

        let turns = dialogue.run("Test prompt".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2); // Broadcast returns all turns
    }

    #[tokio::test]
    async fn test_from_persona_team_sequential() {
        use crate::agent::persona::{Persona, PersonaTeam};

        let mut team = PersonaTeam::new(
            "Sequential Team".to_string(),
            "Sequential testing".to_string(),
        );
        team.add_persona(Persona {
            name: "First".to_string(),
            role: "Analyzer".to_string(),
            background: "Data analyst".to_string(),
            communication_style: "Analytical".to_string(),
        });
        team.add_persona(Persona {
            name: "Second".to_string(),
            role: "Synthesizer".to_string(),
            background: "Content creator".to_string(),
            communication_style: "Creative".to_string(),
        });
        team.execution_strategy = Some("sequential".to_string());

        let llm = MockAgent::new("Mock", vec!["Step output".to_string()]);
        let mut dialogue = Dialogue::from_persona_team(team, llm).unwrap();

        let turns = dialogue.run("Process this".to_string()).await.unwrap();
        assert_eq!(turns.len(), 1); // Sequential returns only final turn
    }

    #[tokio::test]
    async fn test_add_participant_dynamically() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        // Start with one participant
        let initial_persona = Persona {
            name: "Initial".to_string(),
            role: "Initial Agent".to_string(),
            background: "Initial background".to_string(),
            communication_style: "Direct".to_string(),
        };
        dialogue.add_participant(
            initial_persona,
            MockAgent::new("Initial", vec!["Response 1".to_string()]),
        );
        assert_eq!(dialogue.participant_count(), 1);

        // Add a participant dynamically
        let expert = Persona {
            name: "Expert".to_string(),
            role: "Domain Expert".to_string(),
            background: "20 years experience".to_string(),
            communication_style: "Authoritative".to_string(),
        };

        let llm = MockAgent::new("ExpertLLM", vec!["Expert response".to_string()]);
        dialogue.add_participant(expert, llm);

        assert_eq!(dialogue.participant_count(), 2);

        let turns = dialogue.run("Consult experts".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[tokio::test]
    async fn test_persona_team_round_trip() {
        use crate::agent::persona::{Persona, PersonaTeam};
        use tempfile::NamedTempFile;

        // Create team
        let mut team = PersonaTeam::new(
            "Round Trip Team".to_string(),
            "Testing save/load".to_string(),
        );
        team.add_persona(Persona {
            name: "Charlie".to_string(),
            role: "Tester".to_string(),
            background: "QA specialist".to_string(),
            communication_style: "Thorough".to_string(),
        });

        // Save
        let temp_file = NamedTempFile::new().unwrap();
        team.save(temp_file.path()).unwrap();

        // Load
        let loaded_team = PersonaTeam::load(temp_file.path()).unwrap();

        // Create dialogue from loaded team
        let llm = MockAgent::new("Mock", vec!["Response".to_string()]);
        let dialogue = Dialogue::from_persona_team(loaded_team, llm).unwrap();

        assert_eq!(dialogue.participant_count(), 1);
    }

    #[tokio::test]
    async fn test_remove_participant() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona1 = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
        };
        let persona2 = Persona {
            name: "Agent2".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
        };
        let persona3 = Persona {
            name: "Agent3".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
        };

        // Add 3 participants
        dialogue.add_participant(persona1, MockAgent::new("Agent1", vec!["R1".to_string()]));
        dialogue.add_participant(persona2, MockAgent::new("Agent2", vec!["R2".to_string()]));
        dialogue.add_participant(persona3, MockAgent::new("Agent3", vec!["R3".to_string()]));
        assert_eq!(dialogue.participant_count(), 3);

        // Remove middle participant by name
        dialogue.remove_participant("Agent2").unwrap();
        assert_eq!(dialogue.participant_count(), 2);

        // Verify remaining agents work
        let turns = dialogue.run("Test".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].participant_name, "Agent1");
        assert_eq!(turns[1].participant_name, "Agent3");
    }

    #[tokio::test]
    async fn test_remove_participant_not_found() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        let persona = Persona {
            name: "Agent1".to_string(),
            role: "Tester".to_string(),
            background: "Test".to_string(),
            communication_style: "Direct".to_string(),
        };
        dialogue.add_participant(persona, MockAgent::new("Agent1", vec!["R1".to_string()]));

        // Try to remove non-existent participant
        let result = dialogue.remove_participant("NonExistent");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("participant not found")
        );
    }

    #[tokio::test]
    async fn test_from_blueprint_with_predefined_participants() {
        use crate::agent::persona::{Persona, PersonaTeam};

        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Senior engineer".to_string(),
            communication_style: "Technical".to_string(),
        };

        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
        };

        let blueprint = DialogueBlueprint {
            agenda: "Feature Planning".to_string(),
            context: "Planning new feature".to_string(),
            participants: Some(vec![persona1, persona2]),
            execution_strategy: Some("broadcast".to_string()),
        };

        // Mock generator agent - won't be used since participants are provided
        #[derive(Clone)]
        struct MockGeneratorAgent;

        #[async_trait]
        impl Agent for MockGeneratorAgent {
            type Output = PersonaTeam;

            fn expertise(&self) -> &str {
                "Generator"
            }

            fn name(&self) -> String {
                "Generator".to_string()
            }

            async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
                // This should never be called when participants are provided
                panic!("Generator agent should not be called when participants are provided");
            }
        }

        let dialogue_agent = MockAgent::new("DialogueAgent", vec!["Response".to_string()]);

        let mut dialogue = Dialogue::from_blueprint(blueprint, MockGeneratorAgent, dialogue_agent)
            .await
            .unwrap();

        assert_eq!(dialogue.participant_count(), 2);

        let turns = dialogue.run("Test".to_string()).await.unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].participant_name, "Alice");
        assert_eq!(turns[1].participant_name, "Bob");
    }

    #[tokio::test]
    async fn test_guest_participant_workflow() {
        use crate::agent::persona::Persona;

        // Create core team
        let mut dialogue = Dialogue::broadcast();

        let core1 = Persona {
            name: "CoreMember1".to_string(),
            role: "Core Member".to_string(),
            background: "Core team member".to_string(),
            communication_style: "Direct".to_string(),
        };
        let core2 = Persona {
            name: "CoreMember2".to_string(),
            role: "Core Member".to_string(),
            background: "Core team member".to_string(),
            communication_style: "Direct".to_string(),
        };

        dialogue.add_participant(
            core1,
            MockAgent::new("CoreMember1", vec!["Core response 1".to_string()]),
        );
        dialogue.add_participant(
            core2,
            MockAgent::new("CoreMember2", vec!["Core response 2".to_string()]),
        );
        assert_eq!(dialogue.participant_count(), 2);

        // Add guest for specific topic
        let guest = Persona {
            name: "Guest Expert".to_string(),
            role: "Domain Specialist".to_string(),
            background: "Guest invited for this session".to_string(),
            communication_style: "Expert".to_string(),
        };
        let guest_llm = MockAgent::new("Guest", vec!["Guest insight".to_string()]);
        dialogue.add_participant(guest, guest_llm);
        assert_eq!(dialogue.participant_count(), 3);

        // Discussion with guest
        let with_guest = dialogue
            .run("Topic requiring expert input".to_string())
            .await
            .unwrap();
        assert_eq!(with_guest.len(), 3); // All 3 participants

        // Remove guest after their input (by name)
        dialogue.remove_participant("Guest Expert").unwrap();
        assert_eq!(dialogue.participant_count(), 2);

        // Continue with core team only
        let core_only = dialogue
            .run("Continue discussion".to_string())
            .await
            .unwrap();
        assert_eq!(core_only.len(), 2); // Back to 2 core members
    }

    #[tokio::test]
    async fn test_participants_getter() {
        use crate::agent::persona::Persona;

        let mut dialogue = Dialogue::broadcast();

        // Initially empty
        assert_eq!(dialogue.participants().len(), 0);

        // Add first participant
        let persona1 = Persona {
            name: "Alice".to_string(),
            role: "Developer".to_string(),
            background: "Senior engineer".to_string(),
            communication_style: "Technical".to_string(),
        };
        dialogue.add_participant(
            persona1.clone(),
            MockAgent::new("Alice", vec!["Response 1".to_string()]),
        );

        // Add second participant
        let persona2 = Persona {
            name: "Bob".to_string(),
            role: "Designer".to_string(),
            background: "UX specialist".to_string(),
            communication_style: "User-focused".to_string(),
        };
        dialogue.add_participant(
            persona2.clone(),
            MockAgent::new("Bob", vec!["Response 2".to_string()]),
        );

        // Add third participant
        let persona3 = Persona {
            name: "Charlie".to_string(),
            role: "Manager".to_string(),
            background: "Product manager".to_string(),
            communication_style: "Strategic".to_string(),
        };
        dialogue.add_participant(
            persona3.clone(),
            MockAgent::new("Charlie", vec!["Response 3".to_string()]),
        );

        // Get participants and verify
        let participants = dialogue.participants();
        assert_eq!(participants.len(), 3);

        // Check names
        assert_eq!(participants[0].name, "Alice");
        assert_eq!(participants[1].name, "Bob");
        assert_eq!(participants[2].name, "Charlie");

        // Check roles
        assert_eq!(participants[0].role, "Developer");
        assert_eq!(participants[1].role, "Designer");
        assert_eq!(participants[2].role, "Manager");

        // Check backgrounds
        assert_eq!(participants[0].background, "Senior engineer");
        assert_eq!(participants[1].background, "UX specialist");
        assert_eq!(participants[2].background, "Product manager");

        // Check communication styles
        assert_eq!(participants[0].communication_style, "Technical");
        assert_eq!(participants[1].communication_style, "User-focused");
        assert_eq!(participants[2].communication_style, "Strategic");

        // Remove one and verify list updates
        dialogue.remove_participant("Bob").unwrap();
        let participants_after_removal = dialogue.participants();
        assert_eq!(participants_after_removal.len(), 2);
        assert_eq!(participants_after_removal[0].name, "Alice");
        assert_eq!(participants_after_removal[1].name, "Charlie");
    }
}
