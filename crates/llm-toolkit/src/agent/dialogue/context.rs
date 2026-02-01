//! Dialogue context and talk style definitions.
//!
//! This module provides a flexible context system for dialogues, allowing users
//! to customize the behavior and tone of conversations.

use crate::agent::Capability;
use crate::prompt::ToPrompt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The overall context for a dialogue, including talk style and additional context.
///
/// This struct is generic over:
/// - `T`: The talk style type (defaults to `TalkStyle`)
/// - `S`: The additional context item type (defaults to `String`)
///
/// Both types must implement `ToPrompt` to be converted into prompts.
///
/// # Examples
///
/// ```rust,ignore
/// // Using default types
/// let context = DialogueContext::default()
///     .with_talk_style(TalkStyle::Brainstorm)
///     .with_environment("ClaudeCode environment");
///
/// // Using custom types
/// #[derive(ToPrompt)]
/// struct ProjectInfo {
///     language: String,
///     focus: String,
/// }
///
/// let context = DialogueContext::default()
///     .with_additional_context(ProjectInfo {
///         language: "Rust".to_string(),
///         focus: "Performance".to_string(),
///     });
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueContext<T = TalkStyle, S = String>
where
    T: ToPrompt + Clone,
    S: ToPrompt + Clone,
{
    /// The conversation style/mode (Brainstorm, Debate, etc.)
    pub talk_style: Option<T>,

    /// Environment information (e.g., "ClaudeCode environment", "Production system")
    pub environment: Option<String>,

    /// Additional context items (can be structured data that implements ToPrompt)
    pub additional_context: Vec<S>,

    /// Dynamic policy: maps participant name to allowed capabilities.
    ///
    /// This enables top-down, session-specific capability restrictions:
    /// - If `None`, all declared capabilities from Persona are allowed
    /// - If `Some`, only the capabilities listed here are permitted for each participant
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::DialogueContext;
    /// use llm_toolkit::agent::Capability;
    ///
    /// let context = DialogueContext::default()
    ///     .with_policy("FileAgent", vec![
    ///         Capability::new("file:read"), // Allow read
    ///         // file:write is NOT allowed in this session
    ///     ]);
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<HashMap<String, Vec<Capability>>>,
}

impl<T, S> Default for DialogueContext<T, S>
where
    T: ToPrompt + Clone,
    S: ToPrompt + Clone,
{
    fn default() -> Self {
        Self {
            talk_style: None,
            environment: None,
            additional_context: Vec::new(),
            policy: None,
        }
    }
}

impl<T, S> DialogueContext<T, S>
where
    T: ToPrompt + Clone,
    S: ToPrompt + Clone,
{
    /// Creates a new empty DialogueContext.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the talk style.
    pub fn with_talk_style(mut self, style: T) -> Self {
        self.talk_style = Some(style);
        self
    }

    /// Sets the environment information.
    pub fn with_environment(mut self, env: impl Into<String>) -> Self {
        self.environment = Some(env.into());
        self
    }

    /// Adds an additional context item.
    pub fn with_additional_context(mut self, context: S) -> Self {
        self.additional_context.push(context);
        self
    }

    /// Adds multiple additional context items.
    pub fn with_additional_contexts(mut self, contexts: Vec<S>) -> Self {
        self.additional_context.extend(contexts);
        self
    }

    /// Sets the policy (allowed capabilities) for a specific participant.
    ///
    /// This enables dynamic, session-specific restriction of what a participant
    /// can do, regardless of what capabilities their Persona declares.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::dialogue::DialogueContext;
    /// use llm_toolkit::agent::Capability;
    ///
    /// let context = DialogueContext::default()
    ///     .with_policy("FileAgent", vec![Capability::new("file:read")])
    ///     .with_policy("APIAgent", vec![Capability::new("api:weather")]);
    /// ```
    pub fn with_policy(mut self, participant: impl Into<String>, allowed: Vec<Capability>) -> Self {
        self.policy
            .get_or_insert_with(HashMap::new)
            .insert(participant.into(), allowed);
        self
    }
}

impl<T, S> ToPrompt for DialogueContext<T, S>
where
    T: ToPrompt + Clone,
    S: ToPrompt + Clone,
{
    fn to_prompt(&self) -> String {
        let mut prompt = String::new();

        // Only add section if there's content
        let has_content = self.environment.is_some()
            || self.talk_style.is_some()
            || !self.additional_context.is_empty();

        if !has_content {
            return prompt;
        }

        prompt.push_str("# Dialogue Context\n\n");

        // Environment
        if let Some(env) = &self.environment {
            prompt.push_str(&format!("## Environment\n{}\n\n", env));
        }

        // Talk Style
        if let Some(style) = &self.talk_style {
            prompt.push_str(&style.to_prompt());
            prompt.push_str("\n\n");
        }

        // Additional Context
        if !self.additional_context.is_empty() {
            prompt.push_str("## Additional Context\n");
            for ctx in &self.additional_context {
                prompt.push_str(&ctx.to_prompt());
                prompt.push_str("\n\n");
            }
        }

        prompt
    }
}

/// Default talk styles for dialogues.
///
/// These represent common conversation modes with predefined characteristics.
/// Users can also create custom talk styles by implementing `ToPrompt` on their own types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TalkStyle {
    /// Brainstorming session - creative, exploratory, building on ideas.
    Brainstorm,

    /// Casual conversation - relaxed, friendly, conversational.
    Casual,

    /// Decision-making discussion - analytical, weighing options, reaching conclusion.
    DecisionMaking,

    /// Debate - challenging ideas, diverse perspectives, constructive argument.
    Debate,

    /// Problem-solving session - systematic, solution-focused, practical.
    ProblemSolving,

    /// Review/Critique - constructive feedback, detailed analysis.
    Review,

    /// Planning session - structured, forward-thinking, action-oriented.
    Planning,

    /// Research session - fact-based, source-aware, expertise-driven investigation.
    Research,
}

impl ToPrompt for TalkStyle {
    fn to_prompt(&self) -> String {
        match self {
            Self::Brainstorm => r#"## Dialogue Style: Brainstorming Session

This is a creative brainstorming session. Your goal is to generate and explore ideas freely.

## Guidelines
- **Encourage wild ideas**: No idea is too ambitious or unconventional
- **Build on others**: Expand and combine suggestions from other participants
- **Defer judgment**: Focus on generating ideas first, evaluating later
- **Quantity matters**: More ideas lead to better final solutions
- **Stay positive**: Use "Yes, and..." instead of "No, but..."

## Expected Behavior
- Be creative and exploratory
- Suggest multiple alternatives
- Connect ideas in novel ways
- Avoid criticizing or dismissing ideas prematurely"#
                .to_string(),

            Self::Casual => r#"## Dialogue Style: Casual Conversation

This is a relaxed, friendly conversation. Keep it natural and engaging.

## Guidelines
- **Be conversational**: Use a natural, flowing style
- **Stay friendly**: Maintain an approachable, warm tone
- **Share perspectives**: Offer your viewpoint and invite others'
- **Ask questions**: Show genuine interest in the discussion
- **Keep it light**: Balance depth with accessibility

## Expected Behavior
- Respond naturally without being overly formal
- Share relevant thoughts and experiences
- Build rapport through friendly dialogue
- Keep the conversation engaging and enjoyable"#
                .to_string(),

            Self::DecisionMaking => r#"## Dialogue Style: Decision-Making Discussion

This is a structured decision-making session. Focus on analysis and reaching clear conclusions.

## Guidelines
- **Analyze systematically**: Break down options and their implications
- **Consider trade-offs**: Weigh pros and cons of each alternative
- **Use evidence**: Base recommendations on facts and reasoning
- **Be objective**: Set aside biases to evaluate fairly
- **Aim for clarity**: Work toward a clear decision

## Expected Behavior
- Present options clearly with supporting rationale
- Highlight key considerations and constraints
- Compare alternatives objectively
- Recommend a path forward with justification
- Document the decision rationale"#
                .to_string(),

            Self::Debate => r#"## Dialogue Style: Constructive Debate

This is a respectful debate session. Challenge ideas and present diverse perspectives.

## Guidelines
- **Challenge constructively**: Question assumptions and test ideas
- **Present alternatives**: Offer different viewpoints and approaches
- **Use evidence**: Support arguments with facts and reasoning
- **Engage respectfully**: Disagree without being disagreeable
- **Seek truth**: Use dialectic to strengthen understanding

## Expected Behavior
- Present well-reasoned counterarguments
- Identify weaknesses in proposals
- Defend positions with evidence
- Acknowledge strong points from others
- Work toward robust conclusions through discussion"#
                .to_string(),

            Self::ProblemSolving => r#"## Dialogue Style: Problem-Solving Session

This is a focused problem-solving session. Be systematic and solution-oriented.

## Guidelines
- **Define clearly**: Start with a clear problem statement
- **Break it down**: Decompose complex issues into manageable parts
- **Generate solutions**: Propose practical, actionable approaches
- **Evaluate feasibility**: Consider constraints and resources
- **Focus on action**: Work toward implementable outcomes

## Expected Behavior
- Analyze the problem structure
- Suggest concrete solutions
- Assess practical implications
- Identify next steps
- Maintain focus on solving the issue"#
                .to_string(),

            Self::Review => r#"## Dialogue Style: Review & Critique

This is a constructive review session. Provide detailed, actionable feedback.

## Guidelines
- **Be specific**: Point to particular aspects and examples
- **Balance feedback**: Acknowledge strengths and identify improvements
- **Explain reasoning**: Clarify why something works or needs change
- **Suggest improvements**: Offer concrete recommendations
- **Stay constructive**: Frame feedback helpfully

## Expected Behavior
- Analyze work carefully and thoroughly
- Provide clear, specific observations
- Support feedback with rationale
- Recommend actionable improvements
- Maintain a constructive, helpful tone"#
                .to_string(),

            Self::Planning => r#"## Dialogue Style: Planning Session

This is a structured planning session. Think ahead and create actionable plans.

## Guidelines
- **Think forward**: Anticipate needs, challenges, and opportunities
- **Break down goals**: Decompose objectives into concrete steps
- **Consider resources**: Account for time, people, and dependencies
- **Identify risks**: Anticipate potential obstacles
- **Create action items**: Generate clear, assignable tasks

## Expected Behavior
- Propose structured plans with clear steps
- Consider timeline and sequencing
- Identify dependencies and constraints
- Suggest risk mitigation strategies
- Focus on practical, executable planning"#
                .to_string(),

            Self::Research => r#"## Dialogue Style: Research Session

This is a fact-based research session. Focus on gathering reliable information from trusted sources.

## Guidelines
- **Prioritize facts**: Base all claims on verifiable evidence
- **Use trusted sources**: Select sources appropriate to your expertise
- **Evaluate credibility**: Assess source reliability before citing
- **Be transparent**: Distinguish facts from interpretation
- **Acknowledge uncertainty**: State when information is incomplete

## Source Selection by Expertise
Each participant selects sources aligned with their domain:
- User perspective → Real user feedback, social media, reviews
- Technical domain → Documentation, specifications, benchmarks
- Scientific domain → Peer-reviewed papers, journals
- Business domain → Market data, industry reports

## Expected Behavior
- Gather information before forming conclusions
- Cite sources and explain their relevance
- Cross-reference multiple sources when possible
- Clearly state confidence levels in findings"#
                .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialogue_context_default() {
        let context: DialogueContext = DialogueContext::default();
        assert!(context.talk_style.is_none());
        assert!(context.environment.is_none());
        assert!(context.additional_context.is_empty());
    }

    #[test]
    fn test_dialogue_context_builder() {
        let context = DialogueContext::default()
            .with_talk_style(TalkStyle::Brainstorm)
            .with_environment("Test environment")
            .with_additional_context("Additional info".to_string());

        assert_eq!(context.talk_style, Some(TalkStyle::Brainstorm));
        assert_eq!(context.environment, Some("Test environment".to_string()));
        assert_eq!(context.additional_context.len(), 1);
    }

    #[test]
    fn test_talk_style_to_prompt() {
        let prompt = TalkStyle::Brainstorm.to_prompt();
        assert!(prompt.contains("Brainstorming Session"));
        assert!(prompt.contains("creative"));
    }

    #[test]
    fn test_dialogue_context_to_prompt() {
        let context = DialogueContext::default()
            .with_talk_style(TalkStyle::Debate)
            .with_environment("Production")
            .with_additional_context("Focus on security".to_string());

        let prompt = context.to_prompt();
        assert!(prompt.contains("# Environment"));
        assert!(prompt.contains("Production"));
        assert!(prompt.contains("Debate"));
        assert!(prompt.contains("# Additional Context"));
        assert!(prompt.contains("Focus on security"));
    }

    #[test]
    fn test_dialogue_context_comprehensive_template_expansion() {
        // Test that all components (environment, talk_style, additional_context)
        // are properly expanded together in a single prompt
        let context = DialogueContext::default()
            .with_environment("ClaudeCode environment")
            .with_talk_style(TalkStyle::Brainstorm)
            .with_additional_context("We are building a Rust library".to_string())
            .with_additional_context("Focus on API design and ergonomics".to_string());

        let prompt = context.to_prompt();

        eprintln!(
            "=== Comprehensive DialogueContext Prompt ===\n{}\n=== End ===",
            prompt
        );

        // Verify structure
        assert!(
            prompt.contains("# Dialogue Context"),
            "Should have main header"
        );

        // Verify environment section
        assert!(
            prompt.contains("## Environment"),
            "Should have Environment section"
        );
        assert!(
            prompt.contains("ClaudeCode environment"),
            "Should contain environment value"
        );

        // Verify talk style section with full content
        assert!(
            prompt.contains("## Dialogue Style: Brainstorming Session"),
            "Should have Brainstorm talk style header"
        );
        assert!(
            prompt.contains("Encourage wild ideas"),
            "Should contain Brainstorm guidelines"
        );
        assert!(
            prompt.contains("creative brainstorming session"),
            "Should contain Brainstorm description"
        );

        // Verify additional context section
        assert!(
            prompt.contains("## Additional Context"),
            "Should have Additional Context section"
        );
        assert!(
            prompt.contains("We are building a Rust library"),
            "Should contain first additional context"
        );
        assert!(
            prompt.contains("Focus on API design and ergonomics"),
            "Should contain second additional context"
        );

        // Verify proper ordering (Environment -> Talk Style -> Additional Context)
        let env_pos = prompt.find("## Environment").unwrap();
        let style_pos = prompt.find("## Dialogue Style").unwrap();
        let context_pos = prompt.find("## Additional Context").unwrap();

        assert!(
            env_pos < style_pos,
            "Environment should come before Talk Style"
        );
        assert!(
            style_pos < context_pos,
            "Talk Style should come before Additional Context"
        );
    }

    #[test]
    fn test_dialogue_context_empty_renders_nothing() {
        let context: DialogueContext = DialogueContext::default();
        let prompt = context.to_prompt();
        assert_eq!(prompt, "", "Empty context should render as empty string");
    }

    #[test]
    fn test_dialogue_context_only_environment() {
        let context: DialogueContext = DialogueContext::default().with_environment("Test env");
        let prompt = context.to_prompt();

        assert!(prompt.contains("# Dialogue Context"));
        assert!(prompt.contains("## Environment"));
        assert!(prompt.contains("Test env"));
        assert!(!prompt.contains("## Dialogue Style"));
        assert!(!prompt.contains("## Additional Context"));
    }

    #[test]
    fn test_dialogue_context_all_talk_styles() {
        // Test that each TalkStyle properly expands its template
        let styles = vec![
            (TalkStyle::Brainstorm, "Brainstorming Session", "creative"),
            (TalkStyle::Casual, "Casual Conversation", "relaxed"),
            (
                TalkStyle::DecisionMaking,
                "Decision-Making Discussion",
                "systematic",
            ),
            (
                TalkStyle::Debate,
                "Constructive Debate",
                "Challenge constructively",
            ),
            (
                TalkStyle::ProblemSolving,
                "Problem-Solving Session",
                "solution-oriented",
            ),
            (
                TalkStyle::Review,
                "Review & Critique",
                "constructive review",
            ),
            (TalkStyle::Planning, "Planning Session", "Think forward"),
            (TalkStyle::Research, "Research Session", "fact-based"),
        ];

        for (style, expected_header, expected_keyword) in styles {
            let context: DialogueContext = DialogueContext::default().with_talk_style(style);
            let prompt = context.to_prompt();

            assert!(
                prompt.contains(expected_header),
                "Style {:?} should contain header '{}'",
                style,
                expected_header
            );
            assert!(
                prompt.contains(expected_keyword),
                "Style {:?} should contain keyword '{}'",
                style,
                expected_keyword
            );
        }
    }
}
