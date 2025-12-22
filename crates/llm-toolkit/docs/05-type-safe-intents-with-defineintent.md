## Type-Safe Intents with `define_intent!`


To achieve the highest level of type safety and developer experience, the `#[define_intent]` macro automates the entire process of creating and extracting intents.

It solves a critical problem: by defining the prompt, the intent `enum`, and the extraction logic in a single place, it becomes impossible for the prompt-building code and the response-parsing code to diverge.

**Usage:**

Simply annotate an enum with `#[define_intent]` and provide the prompt template and extractor tag in an `#[intent(...)]` attribute.

```rust
use llm_toolkit::{define_intent, IntentExtractor, IntentError};
use std::str::FromStr;

#[define_intent]
#[intent(
    prompt = r#"
Please classify the user's request. The available intents are:
{{ intents_doc }}

User request: <query>{{ user_request }}</query>
"#,
    extractor_tag = "intent"
)]
/// The user's primary intent.
pub enum UserIntent {
    /// The user wants to know the weather.
    GetWeather,
    /// The user wants to send a message.
    SendMessage,
}

// The macro automatically generates:
// 1. A function: `build_user_intent_prompt(user_request: &str) -> String`
// 2. A struct: `pub struct UserIntentExtractor;` which implements `IntentExtractor<UserIntent>`

// --- How to use the generated code ---

// 1. Build the prompt
let prompt = build_user_intent_prompt("what's the weather like in London?");
// The prompt will include the formatted documentation from the enum.

// 2. Use the generated extractor to parse the LLM's response
let llm_response = "Understood. The user wants to know the weather. <intent>GetWeather</intent>";
let extractor = UserIntentExtractor;
let intent = extractor.extract_intent(llm_response).unwrap();

assert_eq!(intent, UserIntent::GetWeather);
```

This macro provides:
- **Ultimate Type Safety:** The prompt and the parser are guaranteed to be in sync.
- **Improved DX:** Eliminates boilerplate code for prompt functions and extractors.
- **Single Source of Truth:** The `enum` becomes the single, reliable source for all intent-related logic.

### Multi-Tag Mode for Complex Action Extraction

For more complex scenarios where you need to extract multiple action tags from a single LLM response, the `define_intent!` macro supports a `multi_tag` mode. This is particularly useful for agent-like applications where the LLM might use multiple XML-style action tags in a single response.

**Setup:**

To use multi-tag mode, add both dependencies to your `Cargo.toml`:

```toml
[dependencies]
llm-toolkit = { version = "0.8.3", features = ["derive"] }
quick-xml = "0.38"  # Required for multi_tag mode
```

Then define your actions:

```rust
use llm_toolkit::define_intent;

#[define_intent(mode = "multi_tag")]
#[intent(
    prompt = r#"Based on the user request, generate a response using the following available actions.

**Available Actions:**
{{ actions_doc }}

**User Request:**
{{ user_request }}"#
)]
#[derive(Debug, Clone, PartialEq)]
pub enum ChatAction {
    /// Get the current weather
    #[action(tag = "GetWeather")]
    GetWeather,

    /// Show an image to the user
    #[action(tag = "ShowImage")]
    ShowImage {
        /// The URL of the image to display
        #[action(attribute)]
        href: String,
    },

    /// Send a message to someone
    #[action(tag = "SendMessage")]
    SendMessage {
        /// The recipient of the message
        #[action(attribute)]
        to: String,
        /// The content of the message
        #[action(inner_text)]
        content: String,
    },
}
```

**Action Tag Attributes:**
- `#[action(tag = "TagName")]` - Defines the XML tag name for this action
- `#[action(attribute)]` - Maps a field to an XML attribute (e.g., `<Tag field="value" />`)
- `#[action(inner_text)]` - Maps a field to the inner text content (e.g., `<Tag>field_value</Tag>`)

**Generated Functions:**
The macro generates:
1. `build_chat_action_prompt(user_request: &str) -> String` - Builds the prompt with action documentation
2. `ChatActionExtractor` struct with methods:
   - `extract_actions(&self, text: &str) -> Result<Vec<ChatAction>, IntentError>` - Extract all actions from response
   - `transform_actions<F>(&self, text: &str, transformer: F) -> String` - Transform action tags using a closure
   - `strip_actions(&self, text: &str) -> String` - Remove all action tags from text

**Usage Example:**

```rust
// 1. Build the prompt
let prompt = build_chat_action_prompt("What's the weather and show me a cat picture?");

// 2. Extract multiple actions from LLM response
let llm_response = r#"
Here's the weather: <GetWeather />
And here's a cat picture: <ShowImage href="https://cataas.com/cat" />
<SendMessage to="user">I've completed both requests!</SendMessage>
"#;

let extractor = ChatActionExtractor;
let actions = extractor.extract_actions(llm_response)?;
// Returns: [ChatAction::GetWeather, ChatAction::ShowImage { href: "https://cataas.com/cat" }, ...]

// 3. Transform action tags to human-readable descriptions
let transformed = extractor.transform_actions(llm_response, |action| match action {
    ChatAction::GetWeather => "[Checking weather...]".to_string(),
    ChatAction::ShowImage { href } => format!("[Displaying image from {}]", href),
    ChatAction::SendMessage { to, content } => format!("[Message to {}: {}]", to, content),
});
// Result: "Here's the weather: [Checking weather...]\nAnd here's a cat picture: [Displaying image from https://cataas.com/cat]\n[Message to user: I've completed both requests!]"

// 4. Strip all action tags for clean text output
let clean_text = extractor.strip_actions(llm_response);
// Result: "Here's the weather: \nAnd here's a cat picture: \n"
```

**When to Use Multi-Tag Mode:**
- **Agent Applications:** When building AI agents that perform multiple actions per response
- **Rich LLM Interactions:** When you need structured actions mixed with natural language
- **Action Processing Pipelines:** When you need to extract, transform, or clean action-based responses

##### 3. Stateful Agents with Personas

For creating stateful, character-driven agents that maintain conversational history, `llm-toolkit` provides the `PersonaAgent` decorator and a convenient `persona` attribute for the `#[agent]` macro. This allows you to give your agents a consistent personality and memory.

**Use Case:** Building chatbots, game characters, or any AI that needs to remember past interactions and respond in character.

**Method 1: Manual Wrapping with `PersonaAgent` (for custom logic)**

You can manually wrap any existing agent with `PersonaAgent` to add persona and dialogue history.

```rust
use llm_toolkit::agent::{Agent, Persona, PersonaAgent};
use llm_toolkit::agent::impls::ClaudeCodeAgent;

// 1. Define a persona
let philosopher_persona = Persona {
    name: "Unit 734",
    role: "Philosopher Robot",
    background: "An android created to explore the nuances of human consciousness.",
    communication_style: "Speaks in a calm, measured tone, often using rhetorical questions.",
};

// 2. Create a base agent
let base_agent = ClaudeCodeAgent::default();

// 3. Wrap it with PersonaAgent
let character_agent = PersonaAgent::new(base_agent, philosopher_persona);

// 4. Interact
let response1 = character_agent.execute("Please introduce yourself.".into()).await?;
let response2 = character_agent.execute("What is your purpose?".into()).await?; // Remembers the first interaction
```

**Method 2: Simplified Usage with `#[agent(persona = ...)]` (Recommended)**

For maximum convenience, you can directly specify a persona in the `#[agent]` macro. The macro will automatically handle the `PersonaAgent` wrapping for you, preserving the inner agent's output type (structured data, attachments, etc.).

```rust
use llm_toolkit::agent::{Agent, persona::Persona};
use std::sync::OnceLock;

// Define a persona using a static or a function
const YUI_PERSONA: Persona = Persona {
    name: "Yui",
    role: "World-Class Pro Engineer",
    background: "A professional and precise AI assistant.",
    communication_style: "Clear, concise, and detail-oriented.",
};

// Use the persona directly in the agent macro
#[llm_toolkit::agent(
    expertise = "Analyzing technical requirements and providing implementation details.",
    persona = "self::YUI_PERSONA"
)]
struct YuiAgent;

// The agent is now stateful and will respond as Yui
let yui = YuiAgent::default();
let response = yui.execute("Introduce yourself.".into()).await?;
// Yui will introduce herself according to her persona and remember this interaction.
```

**Features:**
- ✅ **Stateful Conversation**: Automatically manages and includes dialogue history in prompts.
- ✅ **Consistent Personality**: Enforces a character's persona across multiple turns.
- ✅ **Excellent DX**: The `#[agent(persona = ...)]` attribute makes creating character agents trivial.
- ✅ **Composable**: `PersonaAgent` can wrap *any* agent that implements `Agent`.
- ✅ **Multimodal-Friendly**: Accepts full `Payload` inputs so persona agents can inspect attachments.

**Visual Identity for Enhanced Recognition**

Personas support optional visual identities (icons, taglines, and colors) that strengthen LLM role adherence and improve multi-agent dialogue clarity:

```rust
use llm_toolkit::agent::persona::{Persona, VisualIdentity};

let alice = Persona::new("Alice", "UI/UX Designer")
    .with_background("10 years of user-centered design experience")
    .with_communication_style("Visual, empathetic, user-focused")
    .with_visual_identity(
        VisualIdentity::new("🎨")
            .with_tagline("User-Centered Design Advocate")
            .with_color("#FF6B6B")
    );

// Or use the convenience method for quick icon addition
let bob = Persona::new("Bob", "Backend Engineer")
    .with_background("Senior engineer specializing in distributed systems")
    .with_communication_style("Technical, pragmatic")
    .with_icon("🔧");

// Visual identities appear in dialogue history and speaker names
println!("{}", alice.display_name()); // "🎨 Alice"
```

**Benefits:**
- ✅ **Enhanced Recognition**: Icons provide visual anchors for LLMs
- ✅ **Improved Clarity**: Easier to distinguish agents in conversation logs
- ✅ **Stronger Adherence**: LLMs maintain role consistency better
- ✅ **Human Readability**: Users quickly identify agents at a glance
- ✅ **Professional**: Taglines communicate expertise clearly
- ✅ **Future-Ready**: Color codes enable UI integration

See `examples/persona_visual_identity.rs` for a complete demonstration.

##### 4. Multi-Agent Dialogue Simulation

For use cases that require simulating conversations *between* multiple AI agents, the `Dialogue` component provides a powerful and flexible solution. It manages the turn-taking, shared history, and execution flow, enabling complex multi-agent interactions like brainstorming sessions or workflow pipelines.

**Core Concepts:**

-   **`Dialogue`**: The main orchestrator for the conversation.
-   **Execution Strategy**: Determines how agents interact. Five strategies are provided:
    -   **`Sequential`**: A pipeline where agents execute in a chain (`A -> B -> C`), with the output of one becoming the input for the next. Ideal for data processing workflows.
    -   **`Broadcast`**: A 1-to-N pattern where all agents respond to the same prompt. Ideal for brainstorming or getting multiple perspectives.
    -   **`Ordered Sequential`**: Similar to Sequential, but allows you to specify the exact execution order by participant names (e.g., `Designer -> PM -> Engineer`). Any participants not in the order list execute afterward in their original order.
    -   **`Ordered Broadcast`**: Similar to Broadcast, but responses are yielded in a custom-specified order rather than completion order. All agents still execute in parallel, but results are buffered and returned in your desired sequence.
    -   **`Mentioned`**: Only `@mentioned` participants respond (e.g., `@Alice @Bob what do you think?`). Falls back to Broadcast if no mentions are found. Perfect for selective participation in group conversations. Supports multiple **mention matching strategies**:
        - **`ExactWord`** (default): Matches `@word` patterns without spaces (e.g., `@Alice`, `@Bob123`, `@太郎`)
        - **`Name`**: Matches full names including spaces (e.g., `@Ayaka Nakamura` matches participant "Ayaka Nakamura"). Requires explicit delimiter (space, comma, period, etc.) after the name.
        - **`Partial`**: Matches by prefix, selecting the longest candidate (e.g., `@Ayaka` matches "Ayaka Nakamura")

        **Delimiter Support**:
        - **ExactWord/Partial**: Mentions are recognized until whitespace or common delimiters (`,` `.` `!` `?` `;` `:` `()` `[]` `{}` `<>` `"` `'` `` ` `` `/` `\` `|`). Example: `@Alice, what do you think?` or `@Bob!`
        - **Name**: Requires space or basic punctuation (`,` `.` `!` `?` `;` `:`) after the full name. For Japanese honorifics, use spaces: `@あやか なかむら さん` (not `@あやか なかむらさん`)

**Usage Example:**

```rust
use llm_toolkit::agent::chat::Chat;
use llm_toolkit::agent::dialogue::{Dialogue, SequentialOrder};
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;

// (Mock agent and personas for demonstration)
# #[derive(Clone)]
# struct MockLLMAgent { agent_type: String }
# #[async_trait]
# impl Agent for MockLLMAgent {
#     type Output = String;
#     type Expertise = &'static str;
#     fn expertise(&self) -> &&'static str {
#         const EXPERTISE: &str = "mock";
#         &EXPERTISE
#     }
#     async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
#         let last_line = intent.to_text().lines().last().unwrap_or("").to_string();
#         Ok(format!("[{}] processed: '{}'", self.agent_type, last_line))
#     }
# }
# const SUMMARIZER_PERSONA: Persona = Persona { name: "Summarizer", role: "Summarizer", background: "...", communication_style: "..." };
# const TRANSLATOR_PERSONA: Persona = Persona { name: "Translator", role: "Translator", background: "...", communication_style: "..." };
# const CRITIC_PERSONA: Persona = Persona { name: "Critic", role: "Critic", background: "...", communication_style: "..." };

// --- Pattern 1: Sequential Pipeline ---
let summarizer = Chat::new(MockLLMAgent { agent_type: "Summarizer".to_string() })
    .with_persona(SUMMARIZER_PERSONA).with_history(false).build();
let translator = Chat::new(MockLLMAgent { agent_type: "Translator".to_string() })
    .with_persona(TRANSLATOR_PERSONA).with_history(false).build();

let mut dialogue = Dialogue::sequential();
dialogue.add_participant(summarizer).add_participant(translator);
let final_result = dialogue.run("A long article text...").await?;
// final_result: Ok(vec!["[Translator] processed: '[Summarizer] processed: 'A long article text...'"])

// Need a different execution chain? Pin the order by persona name.
dialogue.with_sequential_order(SequentialOrder::Explicit(vec![
    "Translator".to_string(), // run Translator first
    "Summarizer".to_string(), // then Summarizer
    // any other participants (e.g., reviewers) run afterward in their original order
]));

// Or create an ordered sequential dialogue from the start:
let mut ordered_dialogue = Dialogue::ordered_sequential(vec![
    "Designer".to_string(),
    "PM".to_string(),
    "Engineer".to_string(),
]);
// Participants execute in: Designer -> PM -> Engineer order

// --- Pattern 2: Broadcast ---
let critic = Chat::new(MockLLMAgent { agent_type: "Critic".to_string() })
    .with_persona(CRITIC_PERSONA).with_history(false).build();
let translator_b = Chat::new(MockLLMAgent { agent_type: "Translator".to_string() })
    .with_persona(TRANSLATOR_PERSONA).with_history(false).build();

let mut dialogue = Dialogue::broadcast();
dialogue.add_participant(critic).add_participant(translator_b);
let responses = dialogue.run("The new API design is complete.").await?;
// responses: Ok(vec!["[Critic] processed: 'The new API design is complete.'", "[Translator] processed: 'The new API design is complete.'"])

// Want responses in a specific order? Use ordered_broadcast:
let mut ordered_broadcast = Dialogue::ordered_broadcast(vec![
    "Jordan".to_string(),  // UX Designer's perspective first
    "Alex".to_string(),    // Engineer's technical view second
    "Sam".to_string(),     // PM's business view last
]);
// All agents run in parallel, but results are yielded in: Jordan, Alex, Sam order

// --- Pattern 3: Mentioned (Selective Participation) ---
# const ALICE_PERSONA: Persona = Persona { name: "Alice", role: "Backend", background: "...", communication_style: "..." };
# const BOB_PERSONA: Persona = Persona { name: "Bob", role: "Frontend", background: "...", communication_style: "..." };
# const CHARLIE_PERSONA: Persona = Persona { name: "Charlie", role: "QA", background: "...", communication_style: "..." };
let alice = Chat::new(MockLLMAgent { agent_type: "Alice".to_string() })
    .with_persona(ALICE_PERSONA).with_history(false).build();
let bob = Chat::new(MockLLMAgent { agent_type: "Bob".to_string() })
    .with_persona(BOB_PERSONA).with_history(false).build();
let charlie = Chat::new(MockLLMAgent { agent_type: "Charlie".to_string() })
    .with_persona(CHARLIE_PERSONA).with_history(false).build();

let mut dialogue = Dialogue::mentioned();
dialogue
    .add_participant(alice)
    .add_participant(bob)
    .add_participant(charlie);

// Only Alice and Bob respond
let turn1 = dialogue.run("@Alice @Bob what's your initial take?").await?;
// turn1: Ok(vec![DialogueTurn from Alice, DialogueTurn from Bob])

// Charlie can respond to their discussion
let turn2 = dialogue.run("@Charlie your QA perspective?").await?;
// turn2: Ok(vec![DialogueTurn from Charlie])

// No mentions → falls back to Broadcast (everyone responds)
let turn3 = dialogue.run("Any final thoughts?").await?;
// turn3: Ok(vec![DialogueTurn from Alice, Bob, Charlie])

// Get participant names for UI auto-completion
let names = dialogue.participant_names();
// names: vec!["Alice", "Bob", "Charlie"]
```

**Mention Matching Strategies:**

For participants with space-containing names like "Ayaka Nakamura", use alternative matching strategies:

```rust
use llm_toolkit::agent::dialogue::{Dialogue, MentionMatchStrategy};

# const AYAKA_PERSONA: Persona = Persona {
#     name: "Ayaka Nakamura",
#     role: "Designer",
#     background: "...",
#     communication_style: "..."
# };
let ayaka = Chat::new(MockLLMAgent { agent_type: "Ayaka".to_string() })
    .with_persona(AYAKA_PERSONA).with_history(false).build();

// Strategy 1: Name matching (exact full name with spaces)
let mut dialogue = Dialogue::mentioned_with_strategy(MentionMatchStrategy::Name);
dialogue.add_participant(ayaka);
let turn = dialogue.run("@Ayaka Nakamura please review this design").await?;
// Matches "Ayaka Nakamura" exactly

// Strategy 2: Partial matching (prefix-based, longest match)
let mut dialogue = Dialogue::mentioned_with_strategy(MentionMatchStrategy::Partial);
dialogue.add_participant(ayaka);
let turn = dialogue.run("@Ayaka what do you think?").await?;
// Matches "Ayaka Nakamura" by prefix "Ayaka"
```

###### Mid-Dialogue Participation with JoiningStrategy

Add participants to an ongoing dialogue with controlled history visibility using `join_in_progress()`:

```rust
use llm_toolkit::agent::dialogue::{Dialogue, JoiningStrategy};
use llm_toolkit::agent::persona::Persona;

let mut dialogue = Dialogue::broadcast();
dialogue.add_participant(alice_persona, alice_agent);
dialogue.add_participant(bob_persona, bob_agent);

// Turn 1-2: Initial conversation
dialogue.run("What's the plan?").await?;
dialogue.run("Let's proceed").await?;

// Add expert consultant mid-conversation with NO historical bias
let consultant_persona = Persona {
    name: "Carol".to_string(),
    role: "Security Expert".to_string(),
    background: "Security specialist".to_string(),
    communication_style: "Analytical".to_string(),
    visual_identity: None,
    capabilities: None,
};

dialogue.join_in_progress(
    consultant_persona,
    consultant_agent,
    JoiningStrategy::Fresh  // No history - fresh perspective
);

// Turn 3: Carol participates without historical context
let turn3 = dialogue.run("Carol, security review please").await?;
```

**Available JoiningStrategy Options:**

```rust
// 1. Fresh: No history (ideal for unbiased expert consultation)
JoiningStrategy::Fresh

// 2. Full: Complete history (ideal for new team members needing context)
JoiningStrategy::Full

// 3. Recent: Only recent N turns (ideal for focused review)
JoiningStrategy::recent_with_turns(5)  // Last 5 turns only

// 4. Range: Custom turn range (advanced scenarios)
JoiningStrategy::range(10, Some(20))  // Turns 10-20
```

**Use Cases:**

- **Fresh**: External consultants, code reviewers, unbiased analysis
- **Full**: New team members, context-dependent tasks, comprehensive review
- **Recent**: Focused review, memory-constrained scenarios, quick catchup
- **Range**: Testing, debugging, specific conversation segment analysis

**Supported in All Modes:**
- ✅ Broadcast: All participants respond in parallel
- ✅ Mentioned: Only @mentioned participants respond
- ✅ Sequential: Participants execute in chain order
- ✅ Moderator: Delegates to above modes
- ✅ Streaming API (`partial_session()`): All modes supported

###### Adding Pre-Configured Agents with Custom Settings

When you need to configure PersonaAgent settings (like ContextConfig) before adding to a Dialogue, use `add_agent()`:

```rust
use llm_toolkit::agent::persona::{PersonaAgent, ContextConfig};
use llm_toolkit::agent::dialogue::Dialogue;

// Configure ContextConfig for better long conversation handling
let config = ContextConfig {
    long_conversation_threshold: 5000,
    recent_messages_count: 10,
    participants_after_context: true,  // Place Participants after Context
    include_trailing_prompt: true,     // Add "YOU (name):" at the end
};

// Create PersonaAgent with custom config
let alice_persona = Persona {
    name: "Alice".to_string(),
    role: "Engineer".to_string(),
    background: "Senior developer".to_string(),
    communication_style: "Technical".to_string(),
    visual_identity: None,
    capabilities: None,
};

let persona_agent = PersonaAgent::new(base_agent, alice_persona.clone())
    .with_context_config(config);

// Add pre-configured agent to dialogue
let mut dialogue = Dialogue::sequential();
dialogue.add_agent(alice_persona, persona_agent);
```

Alternatively, you can use the Chat builder with `with_context_config()`:

```rust
let chat_agent = Chat::new(base_agent)
    .with_persona(persona)
    .with_context_config(config)
    .build();

dialogue.add_participant(persona, chat_agent);
```

###### Streaming Results with `partial_session`

Interactive shells and UI frontends can consume responses incrementally:

```rust
let mut session = dialogue.partial_session("Draft release plan");

while let Some(turn) = session.next_turn().await {
    let turn = turn?; // handle AgentError per participant
    println!("[{}] {}", turn.speaker.name(), turn.content);
}
```

- **Broadcast** sessions stream each agent’s reply as soon as it finishes (fast responders appear first).
- **Sequential** sessions expose intermediate outputs (`turn.content`) before they’re fed into the next participant, so you can surface progress step-by-step.

You can also rely on the built-in `tracing` instrumentation (`target = "llm_toolkit::dialogue"`) to monitor progress without polling the session manually. Attach a `tracing_subscriber` layer and watch for `dialogue_turn_completed` / `dialogue_turn_failed` events to drive dashboards or aggregate metrics.

Need deterministic ordering instead of completion order? Create the session with `partial_session_with_order(prompt, BroadcastOrder::ParticipantOrder)` to buffer results until all earlier participants have responded.

The existing `Dialogue::run` helper still collects everything for you (and, in sequential mode, keeps returning only the final turn) by internally driving a `partial_session` to completion.

###### Multi-Turn Conversations

Execute multiple turns by calling `run()` repeatedly. The dialogue automatically maintains conversation history, so agents have full context in each subsequent turn:

```rust
let mut dialogue = Dialogue::sequential()
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Turn 1: Initial discussion
let turn1 = dialogue.run("Let's discuss the architecture").await?;

// Turn 2: Agents see turn1 in history
let turn2 = dialogue.run("What are the trade-offs?").await?;

// Turn 3: Agents see all previous turns
let turn3 = dialogue.run("Make a final decision").await?;
```

Each `run()` call:
- Executes one turn with all (or mentioned) participants
- Automatically stores results in the message history
- Provides full context to agents in subsequent turns

**Continuing the Conversation:**

Use system messages to guide multi-turn dialogues:

```rust
use llm_toolkit::agent::{Payload, PayloadMessage, Speaker};

// Turn 1: Initial prompt
dialogue.run("Brainstorm ideas for the new feature").await?;

// Turn 2: Ask agents to continue
let continue_prompt = Payload::from_messages(vec![
    PayloadMessage::new(Speaker::System, "NEXT: Refine the top 3 ideas"),
]);
dialogue.run(continue_prompt).await?;

// Turn 3: Request convergence
let decide_prompt = Payload::from_messages(vec![
    PayloadMessage::new(Speaker::System, "NEXT: Make a final decision"),
]);
dialogue.run(decide_prompt).await?;
```

**Convergence-Based Loop:**

For iterative discussions, implement your own convergence logic:

```rust
let mut dialogue = Dialogue::broadcast()
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Initial prompt
dialogue.run("Discuss the API design").await?;

// Continue until convergence or max turns
for turn_num in 1..10 {
    // Check for convergence keywords
    let history = dialogue.history();
    let last_content = history.last().map(|t| &t.content).unwrap_or("");

    if last_content.contains("DECISION:") || last_content.contains("AGREED:") {
        println!("Converged after {} turns", turn_num);
        break;
    }

    // Continue the discussion
    let next_prompt = Payload::from_messages(vec![
        PayloadMessage::new(
            Speaker::System,
            format!("NEXT: Continue discussion (Turn {})", turn_num + 1)
        ),
    ]);
    dialogue.run(next_prompt).await?;
}
```

This approach gives you full control over:
- Turn limits (cost control)
- Convergence conditions (content-based, time-based, etc.)
- Custom prompts for each turn
- Logging and debugging at each step

**Available Methods:**

The `Dialogue` component provides several methods for managing conversations:

-   **`participants() -> Vec<&Persona>`**: Access the list of participant personas. Useful for inspecting names, roles, backgrounds, and communication styles.
-   **`participant_names() -> Vec<&str>`**: Get the names of all participants as strings. Ideal for UI auto-completion of `@mentions`.
-   **`participant_count() -> usize`**: Get the current number of participants.
-   **`add_participant(persona, agent)`**: Dynamically add a new participant to the conversation.
-   **`remove_participant(name)`**: Remove a participant by name (useful for guest participants).
-   **`history() -> &[DialogueTurn]`**: Access the complete conversation history.
-   **`with_context(DialogueContext)`**: Apply a full dialogue context (talk style, environment, additional context) that the toolkit prepends as system guidance before each turn in both `run` and `partial_session`.
-   **`with_talk_style(TalkStyle)`**: Convenient method to set only the conversation style (Brainstorm, Debate, etc.).
-   **`with_environment(String)`**: Set environment information (e.g., "Production environment", "ClaudeCode").
-   **`with_additional_context(impl ToPrompt)`**: Add structured or string-based additional context that gets converted to prompts.

```rust
// Inspect participants
let personas = dialogue.participants();
for persona in personas {
    println!("Participant: {} ({})", persona.name, persona.role);
}

// Dynamically manage participants
dialogue.add_participant(expert_persona, expert_agent);
dialogue.run("Get expert opinion").await?;
dialogue.remove_participant("Expert")?;

// Access conversation history
for turn in dialogue.history() {
    println!("[{}]: {}", turn.speaker.name(), turn.content);
}

// Apply conversation context using convenient builder methods
use llm_toolkit::agent::dialogue::{DialogueContext, TalkStyle};

// Option 1: Use convenient builder methods
let mut dialogue = Dialogue::sequential();
dialogue
    .with_talk_style(TalkStyle::Brainstorm)    // Set conversation style
    .with_environment("Production environment") // Add environment info
    .with_additional_context("Focus on security and performance".to_string())
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Option 2: Build a complete context and apply it
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Debate)
    .with_environment("ClaudeCode environment")
    .with_additional_context("Technical deep-dive".to_string());

dialogue.with_context(context);

// Option 3: Use structured, type-safe additional context
#[derive(ToPrompt)]
struct ProjectContext {
    language: String,
    focus_areas: Vec<String>,
}

dialogue.with_additional_context(ProjectContext {
    language: "Rust".to_string(),
    focus_areas: vec!["Performance".to_string(), "Safety".to_string()],
});

dialogue.partial_session("Kickoff agenda").await?;
```

**Available TalkStyles:**
- `TalkStyle::Brainstorm` - Creative, exploratory, building on ideas
- `TalkStyle::Debate` - Challenging ideas, diverse perspectives
- `TalkStyle::DecisionMaking` - Analytical, weighing options
- `TalkStyle::ProblemSolving` - Systematic, solution-focused
- `TalkStyle::Review` - Constructive feedback, detailed analysis
- `TalkStyle::Planning` - Structured, forward-thinking
- `TalkStyle::Casual` - Relaxed, friendly conversation

The `DialogueContext` is generic and accepts any type implementing `ToPrompt` for additional context, enabling structured, type-safe context management:

```rust
// Custom context types are automatically converted to prompts
#[derive(ToPrompt)]
struct TeamContext {
    team_size: usize,
    experience_level: String,
    constraints: Vec<String>,
}

dialogue.with_additional_context(TeamContext {
    team_size: 5,
    experience_level: "Senior".to_string(),
    constraints: vec!["No breaking changes".to_string()],
});
```

**Capability Declaration and Policy Enforcement:**

The `Capability` system provides a hybrid architecture for managing what agents can do and what they're allowed to do in specific sessions:

- **Bottom-up (Static)**: Agents declare capabilities via `Agent::capabilities()` and `Persona.capabilities`
- **Top-down (Dynamic)**: Dialogues enforce policies via `DialogueContext.policy` to restrict what's allowed in a session

This separation distinguishes "what CAN be done" from "what is ALLOWED in this context."

```rust
use llm_toolkit::agent::{Capability, Persona};
use llm_toolkit::agent::dialogue::{Dialogue, DialogueContext};

// Declare agent capabilities on Persona
let researcher = Persona::new(
    "Alice",
    "Research Specialist",
    "Expert in data gathering and analysis",
    "Analytical and thorough",
)
.with_capabilities(vec![
    Capability::new("web:search").with_description("Search the web for information"),
    Capability::new("db:query").with_description("Query internal databases"),
    Capability::new("file:read").with_description("Read files from disk"),
]);

let writer = Persona::new(
    "Bob",
    "Technical Writer",
    "Creates clear documentation",
    "Clear and concise",
)
.with_capabilities(vec![
    Capability::new("file:write").with_description("Write content to files"),
    Capability::new("file:read").with_description("Read files from disk"),
]);

// Create a dialogue with policy restrictions
let context = DialogueContext::new()
    // Allow Alice only web search in this session (restrict db:query and file:read)
    .with_policy("Alice", vec![
        Capability::new("web:search"),
    ])
    // Allow Bob both capabilities
    .with_policy("Bob", vec![
        Capability::new("file:write"),
        Capability::new("file:read"),
    ]);

let dialogue = Dialogue::broadcast()
    .with_context(context)
    .add_participant(researcher, agent1)
    .add_participant(writer, agent2);

// During execution:
// - Alice sees Bob's capabilities: [file:write, file:read]
// - Bob sees Alice's capabilities: [web:search] (policy-filtered from original 3)
// - This information is distributed via ParticipantInfo
```

**How Capabilities Flow:**

1. **Declaration**: Agents declare what they can do via `Persona.with_capabilities()`
2. **Policy Filtering**: `DialogueContext.with_policy()` restricts allowed capabilities per participant
3. **Distribution**: `ParticipantInfo` carries filtered capabilities to other agents
4. **Coordination**: Agents discover what others can do and coordinate accordingly

**Capability Format:**

Capabilities use a `category:action` naming convention for clarity:

```rust
// Simple capability (just name)
let cap1 = Capability::new("api:weather");

// With description for LLM understanding
let cap2 = Capability::new("db:query")
    .with_description("Query the user database");

// Convenience conversions
let cap3: Capability = "file:read".into();
let cap4: Capability = ("api:call", "Call external APIs").into();
```

**Use Cases:**

- ✅ **Access Control**: Restrict capabilities based on session context (e.g., read-only mode)
- ✅ **Orchestrator Precision**: Select agents based on concrete capabilities, not just expertise
- ✅ **Dialogue Coordination**: Agents discover and leverage each other's capabilities
- ✅ **Dynamic Policy**: Different sessions can enforce different restrictions on the same agents

**Reaction Control with ReactionStrategy:**

The `ReactionStrategy` allows you to control when agents should react to messages, providing fine-grained control over dialogue participation.

```rust
use llm_toolkit::agent::dialogue::{Dialogue, ReactionStrategy};

let mut dialogue = Dialogue::broadcast();
dialogue
    .with_reaction_strategy(ReactionStrategy::UserOnly)  // Only react to user messages
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);
```

**Available Strategies:**

- **`ReactionStrategy::Always`** (default) - React to all messages
  ```rust
  // Default behavior - agents react to everything
  let dialogue = Dialogue::broadcast();  // Uses Always by default
  ```

- **`ReactionStrategy::UserOnly`** - Only react to User messages
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::UserOnly);
  // Agents only react when users speak, ignoring agent-to-agent or system messages
  ```

- **`ReactionStrategy::AgentOnly`** - Only react to Agent messages
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::AgentOnly);
  // Useful for observer agents that analyze other agents' responses
  ```

- **`ReactionStrategy::ExceptSystem`** - React to all messages except System messages
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::ExceptSystem);
  // Ignore system notifications while responding to users and other agents
  ```

- **`ReactionStrategy::Conversational`** - React to User or Agent messages only
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::Conversational);
  // Engage in conversation between users and agents, ignoring system messages
  ```

- **`ReactionStrategy::ExceptContextInfo`** - React to all messages except ContextInfo
  ```rust
  dialogue.with_reaction_strategy(ReactionStrategy::ExceptContextInfo);
  // React to all message types including System, but skip ContextInfo background information
  ```

**Message Metadata and Types:**

The dialogue system supports rich message metadata including message types for context-aware processing:

```rust
use llm_toolkit::agent::dialogue::message::{MessageMetadata, MessageType};
use llm_toolkit::agent::{Payload, Speaker};

// Send a context information message (won't trigger reactions)
let context_payload = Payload::new().add_message_with_metadata(
    Speaker::System,
    "Background: Project uses Rust",
    MessageMetadata::new().with_type(MessageType::ContextInfo),
);

dialogue.run(context_payload).await?;  // Stored in history but doesn't trigger reactions
```

**Available Message Types:**

- `MessageType::Conversational` - Regular dialogue messages (default)
- `MessageType::ContextInfo` - Background information (never triggers reactions)
- `MessageType::Notification` - Status updates
- `MessageType::Alert` - Important notifications

**Use Cases:**

- ✅ **Selective Participation**: Control which messages trigger agent responses
- ✅ **Observer Agents**: Create agents that only analyze other agents' outputs
- ✅ **User-Focused Dialogues**: Ensure agents only respond to user input
- ✅ **Context Management**: Share background information without triggering unnecessary reactions

**Session Resumption and History Injection:**

The `Dialogue` component supports saving and resuming conversations, enabling persistent multi-turn dialogues across process restarts or session boundaries.

```rust
use llm_toolkit::agent::dialogue::Dialogue;

// Session 1: Initial conversation
let mut dialogue = Dialogue::broadcast()
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

let turns = dialogue.run("Discuss project architecture").await?;

// Save the conversation history
dialogue.save_history("session_123.json")?;

// --- Process restart or session end ---

// Session 2: Resume conversation from saved history
let saved_history = Dialogue::load_history("session_123.json")?;

let mut dialogue = Dialogue::broadcast()
    .with_history(saved_history)  // Inject saved history
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Continue from where we left off with full context
let more_turns = dialogue.run("Continue from last discussion").await?;
```

**Alternative: Simple Session Resumption with System Prompt**

For simpler use cases where you want agents to "remember" previous conversations without complex structured history management, use `with_history_as_system_prompt()`:

```rust
// Session 2: Resume with simpler approach
let saved_history = Dialogue::load_history("session_123.json")?;

let mut dialogue = Dialogue::broadcast()
    .with_history_as_system_prompt(saved_history)  // ← Inject as system context
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

// Agents receive full conversation context and can reference previous discussion
let more_turns = dialogue.run("Continue from last discussion").await?;
```

**When to use each approach:**

- **`with_history_as_system_prompt()`** - Use when:
  - ✅ You want simple session restoration with minimal complexity
  - ✅ Your conversation history fits within the LLM's context window
  - ✅ You need agents to "remember" and reference previous conversations
  - ✅ You don't need to query or filter the structured MessageStore

- **`with_history()`** - Use when:
  - ✅ You need structured MessageStore for querying/filtering history
  - ✅ You want agents to manage their own conversation history independently
  - ✅ You're building advanced dialogue features with complex history management
  - ✅ You need fine-grained control over message distribution

**DialogueTurn Structure:**

The `DialogueTurn` struct represents a single turn in the conversation with full speaker attribution:

```rust
pub struct DialogueTurn {
    pub speaker: Speaker,  // Who spoke (System/User/Agent with full role info)
    pub content: String,   // What was said
}
```

The `speaker` field uses the `Speaker` enum to preserve complete attribution information including roles, which is essential for session resumption and conversation analysis.

Key methods for session management:

-   **`with_history(history: Vec<DialogueTurn>)`**: Builder method to inject conversation history into a new dialogue instance as structured messages in the MessageStore. Following the Orchestrator Step pattern, this creates a fresh instance with pre-populated history rather than mutating existing state. **Preserves full speaker information including roles.** Use this for advanced dialogue features requiring structured history queries.

-   **`with_history_as_system_prompt(history: Vec<DialogueTurn>)`**: Builder method to inject conversation history as a formatted system prompt that all agents receive. This simpler approach converts the entire conversation history into readable text that agents can reference, ensuring they "remember" previous discussions. **Ideal for straightforward session restoration** when you don't need structured history management.

-   **`save_history(path)`**: Persists the current conversation history to a JSON file with complete speaker attribution.

-   **`load_history(path)`**: Loads conversation history from a JSON file, restoring all speaker details.

Use cases:
- ✅ **Persistent Conversations**: Maintain dialogue context across application restarts
- ✅ **Session Management**: Save and restore user conversation sessions
- ✅ **Conversation Archival**: Store dialogue history for later analysis
- ✅ **Stateful Chatbots**: Implement chatbots with long-term memory
- ✅ **Agent Memory**: Enable agents to reference and build upon previous conversations

See `examples/dialogue_session_resumption.rs` and `examples/dialogue_session_resumption_system_prompt.rs` for complete demonstrations.

**Multimodal Input Support:**

The `Dialogue` API accepts `impl Into<Payload>`, enabling both text-only and multimodal input (text + attachments) with complete backward compatibility.

```rust
use llm_toolkit::agent::Payload;
use llm_toolkit::attachment::Attachment;

// Text-only input (backward compatible)
dialogue.run("Discuss AI ethics").await?;
dialogue.partial_session("Brainstorm ideas");

// Multimodal input with single attachment
let payload = Payload::text("What's in this image?")
    .with_attachment(Attachment::local("screenshot.png"));
dialogue.run(payload).await?;

// Multiple attachments
let payload = Payload::text("Analyze these files")
    .with_attachment(Attachment::local("data.csv"))
    .with_attachment(Attachment::local("metadata.json"));
dialogue.partial_session(payload);
```

All dialogue methods (`run`, `partial_session`, `partial_session_with_order`) accept any type implementing `Into<Payload>`, including:
- `String` or `&str` for text-only input
- `Payload` for multimodal input with attachments

This design enables:
- ✅ **100% Backward Compatibility**: Existing code works without changes
- ✅ **Extensibility**: New `Payload` features automatically work
- ✅ **Type Safety**: Compiler-enforced correct usage
- ✅ **Zero Method Proliferation**: No `_with_payload` variants needed

**Multi-Message Payloads and Speaker Attribution:**

The `Dialogue` API supports multi-message payloads with explicit speaker attribution, enabling complex conversation structures with System prompts, User inputs, and Agent responses.

```rust
use llm_toolkit::agent::{Payload, PayloadMessage};
use llm_toolkit::agent::dialogue::message::Speaker;

// Create a payload with multiple messages from different speakers
let payload = Payload::from_messages(vec![
    PayloadMessage::system("Context: Project planning meeting"),
    PayloadMessage::user(
        "Alice",
        "Product Manager",
        "What features should we prioritize?",
    ),
]);

// All messages are stored with proper speaker attribution
let turns = dialogue.run(payload).await?;

// Access conversation history with full speaker information
for turn in dialogue.history() {
    match &turn.speaker {
        Speaker::System => println!("[System]: {}", turn.content),
        Speaker::User { name, role } => println!("[{} ({})]: {}", name, role, turn.content),
        Speaker::Agent { name, role } => println!("[{} ({})]: {}", name, role, turn.content),
    }
}
```

The `Speaker` enum provides three variants:
- **`System`**: System-generated prompts or instructions
- **`User { name, role }`**: Human user messages with name and role
- **`Agent { name, role }`**: AI agent responses with persona information

This enables:
- ✅ **Proper Attribution**: Distinguish between System, User, and Agent messages
- ✅ **Role Preservation**: User and Agent roles are preserved in history
- ✅ **Complex Conversations**: Support multi-speaker turns with System + User messages
- ✅ **Session Resumption**: Full speaker context is maintained across save/load cycles

**Dynamic Instructions with Prepend Methods:**

Control agent behavior on a per-turn basis by prepending instructions to payloads without modifying `Persona` definitions:

```rust
use llm_toolkit::agent::Payload;
use llm_toolkit::agent::dialogue::Speaker;

// Prepend a system instruction for this specific turn
let payload = Payload::text("Discuss the architecture")
    .prepend_system("IMPORTANT: Keep responses under 300 characters. Be concise.");

dialogue.run(payload).await?;

// Or use prepend_message for custom speaker attribution
let payload = Payload::text("User question")
    .prepend_message(Speaker::System, "Answer in bullet points only.");
```

This enables:
- ✅ **Dynamic Constraints**: Add turn-specific constraints (e.g., "be concise", "be detailed")
- ✅ **Temporary Instructions**: Inject context-specific guidance without permanent changes
- ✅ **Conversation Control**: Prevent verbosity escalation in multi-agent dialogues
- ✅ **Chaining Support**: Multiple `prepend_*` calls apply in FIFO order

**Use Cases:**
- **Verbosity Control**: Add "Keep responses under 300 characters" when agents start getting too verbose
- **Mode Switching**: Switch between detailed and concise modes based on user preference
- **Context-Specific Behavior**: "Analyze this image concisely" for image attachments
- **Multi-Agent Coordination**: Prevent agents from repeating what others have said

**Enhanced Context Formatting:**

Dialogue participants receive enhanced context that includes:

1. **Participants Table**: Shows all conversation participants with clear "(YOU)" marker
2. **Recent History**: Previous messages with turn numbers and timestamps
3. **Current Task**: The current prompt or request

```text
# Persona Profile
**Name**: Alice
**Role**: Product Manager
...

# Request
# Participants

- **Alice (YOU)** (Product Manager)
  Expert in product strategy

- **Bob** (Engineer)
  Senior backend engineer

# Recent History

## Turn 1 (2024-01-15 10:30:22)
### Bob (Engineer)
I think we should focus on performance...

# Current Task
What should be our next priority?
```

This structured format helps agents:
- ✅ **Understand Context**: Know who else is in the conversation
- ✅ **Self-Identification**: Clear "(YOU)" marker for the current speaker
- ✅ **Temporal Awareness**: Turn numbers and timestamps for conversation flow
- ✅ **Role Clarity**: See the roles and backgrounds of other participants

