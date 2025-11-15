use crate::{
    Agent, AgentError, ToPrompt,
    agent::{
        dialogue::{
            Dialogue, DialogueBlueprint, DialogueContext, DialogueMessage, DialogueTurn,
            ExecutionModel, MentionMatchStrategy, MessageId, MessageStore, ReactionStrategy,
            Speaker, TalkStyle, format_dialogue_history_as_text,
            message::{self, SentAgents},
        },
        persona::{PersonaTeam, PersonaTeamGenerationRequest},
    },
};

impl Dialogue {
    /// Creates a new dialogue with the specified execution model.
    ///
    /// This is private - use `broadcast()` or `sequential()` instead.
    fn new(execution_model: ExecutionModel) -> Self {
        Self {
            participants: Vec::new(),
            message_store: MessageStore::new(),
            execution_model,
            context: None,
            reaction_strategy: ReactionStrategy::default(),
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
        Self::new(ExecutionModel::Broadcast)
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
        Self::new(ExecutionModel::Sequential)
    }

    /// Creates a new dialogue with mentioned execution.
    ///
    /// In mentioned mode, only participants explicitly mentioned with `@name` will respond.
    /// If no mentions are found, it falls back to broadcast mode (all participants respond).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// let mut dialogue = Dialogue::mentioned()
    ///     .add_participant(alice_persona, agent1)
    ///     .add_participant(bob_persona, agent2);
    ///
    /// // Only Alice will respond
    /// let turns = dialogue.run("@Alice what do you think?").await?;
    ///
    /// // Both Alice and Bob will respond
    /// let turns = dialogue.run("@Alice @Bob discuss this").await?;
    ///
    /// // Falls back to broadcast - all participants respond
    /// let turns = dialogue.run("What does everyone think?").await?;
    /// ```
    pub fn mentioned() -> Self {
        Self::new(ExecutionModel::Mentioned {
            strategy: MentionMatchStrategy::default(),
        })
    }

    /// Creates a dialogue with Mentioned execution strategy and custom matching strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The mention matching strategy to use
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::{Dialogue, MentionMatchStrategy};
    ///
    /// // Use full name matching for participants with spaces
    /// let mut dialogue = Dialogue::mentioned_with_strategy(MentionMatchStrategy::Name)
    ///     .add_participant(persona1, agent1); // "Ayaka Nakamura"
    ///
    /// // "@Ayaka Nakamura please review" - matches "Ayaka Nakamura"
    /// let turns = dialogue.run("@Ayaka Nakamura please review this").await?;
    ///
    /// // Use partial matching for prefix-based mentions
    /// let mut dialogue = Dialogue::mentioned_with_strategy(MentionMatchStrategy::Partial)
    ///     .add_participant(persona2, agent2); // "Ayaka Nakamura"
    ///
    /// // "@Ayaka" matches "Ayaka Nakamura" (longest prefix match)
    /// let turns = dialogue.run("@Ayaka what do you think?").await?;
    /// ```
    pub fn mentioned_with_strategy(strategy: MentionMatchStrategy) -> Self {
        Self::new(ExecutionModel::Mentioned { strategy })
    }

    /// Sets initial conversation history for session resumption.
    ///
    /// This method allows you to inject a saved conversation history into a new
    /// dialogue instance, enabling session restoration and continuation of
    /// previous discussions.
    ///
    /// Following the Orchestrator Step pattern, this creates a new dialogue
    /// instance with pre-populated history rather than mutating existing state.
    ///
    /// # Arguments
    ///
    /// * `history` - A vector of `DialogueTurn` representing the conversation history
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// // Session 1: Initial conversation
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    /// let turns = dialogue.run("Discuss project architecture").await?;
    /// dialogue.save_history("session_123.json")?;
    ///
    /// // --- Process restart or session end ---
    ///
    /// // Session 2: Resume conversation
    /// let saved_history = Dialogue::load_history("session_123.json")?;
    /// let mut dialogue = Dialogue::broadcast()
    ///     .with_history(saved_history)  // ← Inject saved history
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    ///
    /// // Continue from where we left off
    /// let more_turns = dialogue.run("Continue from last discussion").await?;
    /// ```
    pub fn with_history(mut self, history: Vec<DialogueTurn>) -> Self {
        // Convert DialogueTurn to DialogueMessage and populate MessageStore
        // Assume each DialogueTurn is from turn 1, in order
        let mut turn_counter = 1;

        for dialogue_turn in history {
            let message = DialogueMessage {
                id: MessageId::new(),
                turn: turn_counter,
                speaker: dialogue_turn.speaker.clone(),
                content: dialogue_turn.content,
                timestamp: message::current_unix_timestamp(),
                metadata: Default::default(),
                sent_agents: SentAgents::All, // Historical messages are considered already sent
            };

            self.message_store.push(message);

            // Increment turn when we see a System message
            if matches!(dialogue_turn.speaker, Speaker::System) {
                turn_counter += 1;
            }
        }

        self
    }

    /// Sets initial conversation history as a SYSTEM prompt for session resumption.
    ///
    /// This method provides a simpler alternative to `with_history()` by converting
    /// the entire conversation history into a single SYSTEM message. This approach:
    /// - Is simpler to implement and maintain
    /// - Leverages modern LLMs' long context capabilities
    /// - Ensures agents can "remember" previous conversations
    ///
    /// The history is formatted as a human-readable conversation log and prepended
    /// to the first prompt that agents receive.
    ///
    /// # When to use this vs `with_history()`
    ///
    /// - **Use `with_history_as_system_prompt()`** when:
    ///   - You want simple session restoration with minimal complexity
    ///   - Your conversation history fits within the LLM's context window
    ///   - You don't need structured history management
    ///
    /// - **Use `with_history()`** when:
    ///   - You need the structured MessageStore for querying/filtering
    ///   - You want agents to manage their own history independently
    ///   - You're building advanced dialogue features
    ///
    /// # Arguments
    ///
    /// * `history` - A vector of `DialogueTurn` representing the conversation history
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::Dialogue;
    ///
    /// // Session 1: Initial conversation
    /// let mut dialogue = Dialogue::broadcast()
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    /// let turns = dialogue.run("Discuss project architecture").await?;
    /// dialogue.save_history("session_123.json")?;
    ///
    /// // --- Process restart or session end ---
    ///
    /// // Session 2: Resume conversation with system prompt approach
    /// let saved_history = Dialogue::load_history("session_123.json")?;
    /// let mut dialogue = Dialogue::broadcast()
    ///     .with_history_as_system_prompt(saved_history)  // ← Inject as system message
    ///     .add_participant(persona1, agent1)
    ///     .add_participant(persona2, agent2);
    ///
    /// // Agents will have context from previous conversation
    /// let more_turns = dialogue.run("Continue from last discussion").await?;
    /// ```
    pub fn with_history_as_system_prompt(mut self, history: Vec<DialogueTurn>) -> Self {
        if history.is_empty() {
            return self;
        }

        // Format the history as a readable conversation log
        // Store it in the context which will be prepended to all prompts
        let history_text = format_dialogue_history_as_text(&history);

        // Add the history as additional context that will be included
        // in the dialogue context for all participants
        let context = self
            .context
            .take()
            .unwrap_or_default()
            .with_additional_context(history_text);
        self.context = Some(context);

        self
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
    ///     execution_strategy: Some(ExecutionModel::Broadcast),
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
        // Determine execution model from blueprint
        let execution_model = blueprint
            .execution_strategy
            .unwrap_or(ExecutionModel::Broadcast);

        let mut dialogue = Self::new(execution_model);

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
        dialogue.participants = Self::create_participants(personas, dialogue_agent);

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
        // Determine execution model from team hint
        let execution_model = team.execution_strategy.unwrap_or(ExecutionModel::Broadcast);

        let mut dialogue = Self::new(execution_model);

        // Build participants from personas
        dialogue.participants = Self::create_participants(team.personas, llm_agent);

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
    ///     visual_identity: None,
    ///     capabilities: None,
    /// };
    ///
    /// dialogue.add_participant(expert, ClaudeCodeAgent::new());
    /// ```
    /// Sets the dialogue context, which shapes the tone and behavior of the conversation.
    ///
    /// The context provides implicit instructions to all participants, eliminating
    /// the need to explain the conversation's purpose in each message.
    ///
    /// # Arguments
    ///
    /// * `context` - The dialogue context to set
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::{Dialogue, DialogueContext};
    ///
    /// // Brainstorming session
    /// let mut dialogue = Dialogue::broadcast()
    ///     .with_context(DialogueContext::Brainstorm)
    ///     .add_participant(persona1, agent1);
    ///
    /// // Custom context
    /// let mut dialogue = Dialogue::sequential()
    ///     .with_context(DialogueContext::Custom(
    ///         "This is a technical deep-dive. Focus on implementation details."
    ///     ))
    ///     .add_participant(persona1, agent1);
    /// ```
    /// Sets the full dialogue context (talk style, environment, additional context).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let context = DialogueContext::default()
    ///     .with_talk_style(TalkStyle::Brainstorm)
    ///     .with_environment("Production environment")
    ///     .with_additional_context("Focus on security".to_string());
    ///
    /// dialogue.with_context(context);
    /// ```
    pub fn with_context(&mut self, context: DialogueContext) -> &mut Self {
        self.context = Some(context);
        self
    }

    /// Sets the talk style for the dialogue.
    ///
    /// This is a convenience method for setting only the talk style without
    /// constructing a full DialogueContext.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// dialogue.with_talk_style(TalkStyle::Debate);
    /// ```
    pub fn with_talk_style(&mut self, style: TalkStyle) -> &mut Self {
        let context = self
            .context
            .take()
            .unwrap_or_default()
            .with_talk_style(style);
        self.context = Some(context);
        self
    }

    /// Sets the reaction strategy for the dialogue.
    ///
    /// This controls when agents should react to messages. By default, agents
    /// react to all messages (ReactionStrategy::Always).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::ReactionStrategy;
    ///
    /// // Only react to User messages
    /// dialogue.with_reaction_strategy(ReactionStrategy::UserOnly);
    ///
    /// // Don't react to System messages
    /// dialogue.with_reaction_strategy(ReactionStrategy::ExceptSystem);
    ///
    /// // Custom logic
    /// dialogue.with_reaction_strategy(ReactionStrategy::Custom(Arc::new(|payload, _| {
    ///     // Only react if payload contains user messages
    ///     payload.to_messages().iter().any(|msg| matches!(msg.speaker, Speaker::User { .. }))
    /// })));
    /// ```
    pub fn with_reaction_strategy(&mut self, strategy: ReactionStrategy) -> &mut Self {
        self.reaction_strategy = strategy;
        self
    }

    /// Sets the environment information for the dialogue.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// dialogue.with_environment("ClaudeCode environment");
    /// ```
    pub fn with_environment(&mut self, env: impl Into<String>) -> &mut Self {
        let context = self
            .context
            .take()
            .unwrap_or_default()
            .with_environment(env);
        self.context = Some(context);
        self
    }

    /// Adds additional context to the dialogue.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// dialogue.with_additional_context("Focus on performance".to_string());
    /// ```
    pub fn with_additional_context(&mut self, ctx: String) -> &mut Self {
        let context = self
            .context
            .take()
            .unwrap_or_default()
            .with_additional_context(ctx);
        self.context = Some(context);
        self
    }
}
