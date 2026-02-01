## Dialogue Context and Talk Styles

The `DialogueContext` and `TalkStyle` types provide flexible configuration for multi-agent dialogues, allowing you to customize conversation behavior and tone.

## DialogueContext

`DialogueContext` encapsulates the overall context for a dialogue session, including talk style, environment information, and additional context items.

```rust
use llm_toolkit::agent::dialogue::{DialogueContext, TalkStyle};

// Basic usage with builder pattern
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Brainstorm)
    .with_environment("Development environment")
    .with_additional_context("Focus on performance optimization".to_string());
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `talk_style` | `Option<T>` | Conversation style/mode |
| `environment` | `Option<String>` | Environment information |
| `additional_context` | `Vec<S>` | Additional context items |
| `policy` | `Option<HashMap<...>>` | Capability restrictions per participant |

### Policy-Based Capability Control

Restrict participant capabilities dynamically per session:

```rust
use llm_toolkit::agent::dialogue::DialogueContext;
use llm_toolkit::agent::Capability;

let context = DialogueContext::default()
    .with_policy("FileAgent", vec![
        Capability::new("file:read"),  // Allow read only
        // file:write is NOT allowed in this session
    ])
    .with_policy("APIAgent", vec![
        Capability::new("api:weather"),
    ]);
```

## TalkStyle

`TalkStyle` defines predefined conversation modes with specific characteristics. Each style generates appropriate prompts to guide agent behavior.

### Available Styles

| Style | Purpose | Key Characteristics |
|-------|---------|---------------------|
| `Brainstorm` | Creative idea generation | Encourage wild ideas, build on others, defer judgment |
| `Casual` | Relaxed conversation | Natural, friendly, conversational tone |
| `DecisionMaking` | Analytical discussion | Systematic analysis, weigh trade-offs, reach conclusions |
| `Debate` | Constructive argument | Challenge ideas, present alternatives, seek truth |
| `ProblemSolving` | Solution-focused | Define clearly, break down, generate solutions |
| `Review` | Constructive feedback | Specific, balanced, actionable improvements |
| `Planning` | Forward-thinking | Structured plans, consider resources, identify risks |
| `Research` | Fact-based investigation | Prioritize facts, use trusted sources, acknowledge uncertainty |

### Usage Examples

```rust
use llm_toolkit::agent::dialogue::{Dialogue, DialogueContext, TalkStyle};

// Brainstorming session
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Brainstorm);

// Research session with environment context
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Research)
    .with_environment("Academic research context")
    .with_additional_context("Topic: Machine learning optimization".to_string());
```

### Research Style Details

The `Research` style is designed for fact-based investigation with source awareness:

- Prioritizes verifiable evidence
- Encourages source selection based on participant expertise
- Requires transparency about uncertainty
- Promotes cross-referencing multiple sources

```rust
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Research);

// The generated prompt includes:
// - Guidelines for fact-based research
// - Source selection by expertise domain
// - Expected behaviors for citing and evaluating sources
```

## Custom Talk Styles

### Using TalkStyleTemplate (Recommended)

The simplest way to create custom styles is with `TalkStyleTemplate`:

```rust
use llm_toolkit::agent::dialogue::{TalkStyle, TalkStyleTemplate, DialogueContext};

// Create a custom template with builder pattern
let template = TalkStyleTemplate::new("Security Audit")
    .with_description("Review code for security vulnerabilities and best practices.")
    .with_guideline("Check for injection vulnerabilities (SQL, command, etc.)")
    .with_guideline("Verify input validation and sanitization")
    .with_guideline("Review authentication and authorization logic")
    .with_expected_behavior("Reference CVE IDs when applicable")
    .with_expected_behavior("Suggest specific remediation steps");

// Use via TalkStyle::Template variant
let context = DialogueContext::default()
    .with_talk_style(TalkStyle::Template(template));
```

`TalkStyleTemplate` fields:

| Field | Description |
|-------|-------------|
| `name` | Style name displayed in the prompt header |
| `description` | Brief description of the session's purpose |
| `guidelines` | List of guidelines for participants |
| `expected_behaviors` | List of expected behaviors during the session |

### Using ToPrompt Trait (Advanced)

For more control, implement `ToPrompt` directly:

```rust
use llm_toolkit::prompt::ToPrompt;

#[derive(Clone)]
struct TechnicalReview {
    focus_areas: Vec<String>,
}

impl ToPrompt for TechnicalReview {
    fn to_prompt(&self) -> String {
        format!(
            "## Dialogue Style: Technical Review\n\n\
             Focus areas: {}\n\n\
             Guidelines:\n\
             - Analyze code quality and architecture\n\
             - Check for security vulnerabilities\n\
             - Evaluate performance implications",
            self.focus_areas.join(", ")
        )
    }
}

// Use with DialogueContext (requires explicit type parameters)
let context: DialogueContext<TechnicalReview, String> = DialogueContext::new()
    .with_talk_style(TechnicalReview {
        focus_areas: vec!["memory safety".into(), "error handling".into()],
    });
```

## Integration with Dialogue

`DialogueContext` integrates seamlessly with the `Dialogue` system:

```rust
use llm_toolkit::agent::dialogue::{Dialogue, DialogueContext, TalkStyle};

let context = DialogueContext::default()
    .with_talk_style(TalkStyle::ProblemSolving)
    .with_environment("Production debugging session");

let mut dialogue = Dialogue::broadcast()
    .with_context(context)
    .add_participant(persona1, agent1)
    .add_participant(persona2, agent2);

let turns = dialogue.run("Analyze the memory leak issue").await?;
```

## Prompt Generation

`DialogueContext` implements `ToPrompt`, generating structured prompts:

```text
# Dialogue Context

## Environment
Production debugging session

## Dialogue Style: Problem-Solving Session
This is a focused problem-solving session. Be systematic and solution-oriented.
[... style-specific guidelines ...]

## Additional Context
- Focus on memory profiling results
- Consider recent deployment changes
```
