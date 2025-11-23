//! Agent-based context detector - LLM-powered semantic context detection
//!
//! This module provides an Agent-based detector that uses any Agent implementation
//! (typically a lightweight LLM like Haiku) to perform semantic analysis of Payload
//! contents and infer high-level context.

use super::context_detector::ContextDetector;
use super::detected_context::{ConfidenceScores, DetectedContext};
use super::error::AgentError;
use super::payload::{Payload, PayloadContent};
use super::Agent;
use crate::context::TaskHealth;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Agent-based context detector using LLM for semantic analysis.
///
/// This detector wraps any Agent implementation (typically a lightweight LLM
/// like Claude Haiku) to perform semantic context detection from Payload contents.
///
/// # Design
///
/// - **Semantic**: Uses LLM's language understanding for nuanced detection
/// - **Flexible**: Can detect complex patterns beyond keyword matching
/// - **Layer 2**: Intended as enrichment layer after RuleBasedDetector
/// - **Configurable**: Can use any Agent (Haiku, GPT-4o-mini, etc.)
///
/// # Examples
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Payload, EnvContext, AgentBasedDetector};
/// use llm_toolkit::agent::context_detector::DetectContextExt;
/// use llm_toolkit::agent::impls::ClaudeCodeAgent;
///
/// // Layer 1: Rule-based (fast)
/// let rule_detector = RuleBasedDetector::new();
/// let payload = Payload::text("The authentication system keeps failing")
///     .with_env_context(EnvContext::new().with_redesign_count(3))
///     .detect_with(&rule_detector).await?;
///
/// // Layer 2: Agent-based (semantic)
/// let agent = ClaudeCodeAgent::new(); // or haiku_agent
/// let llm_detector = AgentBasedDetector::new(agent);
/// let payload = payload.detect_with(&llm_detector).await?;
///
/// // Now has both rule-based and LLM-enriched context
/// ```
///
/// # Using Orc's Internal Agent
///
/// ```rust,ignore
/// // In orchestrator context, use Orc's internal agent for consistency:
/// let orc = Orchestrator::new(blueprint)?;
/// let detector = AgentBasedDetector::new(orc.internal_agent().clone());
/// ```
#[derive(Debug, Clone)]
pub struct AgentBasedDetector<T>
where
    T: Agent<Output = String>,
{
    /// The underlying agent for LLM analysis
    inner_agent: T,

    /// Detection prompt template (optional override)
    prompt_template: Option<String>,
}

/// Detection request DTO sent to the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DetectionRequest {
    /// Raw text content from payload
    text_content: String,

    /// Environment context information
    env_summary: EnvContextSummary,

    /// Existing detected context (from previous layers)
    existing_context: Option<DetectedContextSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnvContextSummary {
    redesign_count: usize,
    has_journal: bool,
    success_rate: Option<f64>,
    consecutive_failures: Option<usize>,
    strategy_phase: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DetectedContextSummary {
    task_type: Option<String>,
    task_health: Option<String>,
    user_states: Vec<String>,
}

/// Detection response from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DetectionResponse {
    /// Detected task type (security-review, code-review, debug, etc.)
    task_type: Option<String>,

    /// Detected task health (OnTrack, AtRisk, OffTrack)
    task_health: Option<String>,

    /// Detected user states (confused, frustrated, blocked, etc.)
    user_states: Vec<String>,

    /// Confidence scores (0.0 - 1.0)
    confidence: ConfidenceScores,

    /// Reasoning explanation (for debugging)
    reasoning: Option<String>,
}

impl<T> AgentBasedDetector<T>
where
    T: Agent<Output = String>,
{
    /// Creates a new agent-based detector with default prompt.
    ///
    /// # Arguments
    ///
    /// * `inner_agent` - The agent to use for LLM-based detection
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use llm_toolkit::agent::{AgentBasedDetector, impls::ClaudeCodeAgent};
    ///
    /// let agent = ClaudeCodeAgent::new();
    /// let detector = AgentBasedDetector::new(agent);
    /// ```
    pub fn new(inner_agent: T) -> Self {
        Self {
            inner_agent,
            prompt_template: None,
        }
    }

    /// Creates a detector with custom prompt template.
    ///
    /// Use `{request}` placeholder for DetectionRequest JSON.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let detector = AgentBasedDetector::new(agent)
    ///     .with_prompt_template(r#"
    /// Analyze the following context and detect:
    /// - task_type: The type of task being performed
    /// - task_health: Current health status
    /// - user_states: User's emotional/cognitive states
    ///
    /// Context: {request}
    ///
    /// Respond in JSON format.
    /// "#);
    /// ```
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }

    /// Returns the default detection prompt.
    fn default_prompt() -> &'static str {
        r#"
You are a context detection system analyzing software development tasks.

Given the following information, detect:
1. **task_type**: The type of task (security-review, code-review, debug, implementation, test, refactoring, documentation)
2. **task_health**: Current health status (OnTrack, AtRisk, OffTrack)
3. **user_states**: User's states (confused, frustrated, blocked, on-track, needs-guidance, exploring, etc.)

# Input
{request}

# Detection Guidelines

## task_type
- **security-review**: Security analysis, vulnerability assessment, auth/authz review
- **code-review**: PR review, code quality assessment, refactoring review
- **debug**: Error investigation, bug fixing, crash analysis
- **implementation**: Feature development, new code creation
- **test**: Test writing, coverage improvement
- **refactoring**: Code restructuring, technical debt reduction
- **documentation**: Docs writing, comment improvement

## task_health
- **OnTrack**: Making progress, no significant blockers
- **AtRisk**: Multiple redesigns, some failures, but recoverable
- **OffTrack**: Severely blocked, high failure rate, needs intervention

## user_states (can be multiple)
- **confused**: Uncertain about approach, asking clarifying questions
- **frustrated**: Repeated failures, expressing frustration
- **blocked**: Cannot proceed, waiting for resolution
- **on-track**: Making steady progress
- **needs-guidance**: Seeking direction or best practices
- **exploring**: Investigating options, experimenting
- **focused**: Deep in implementation, making progress

# Output Format (JSON)
{
  "task_type": "security-review",
  "task_health": "AtRisk",
  "user_states": ["confused", "needs-guidance"],
  "confidence": {
    "task_type": 0.85,
    "task_health": 0.75,
    "user_states": 0.80
  },
  "reasoning": "User is reviewing security code with multiple redesigns. Shows signs of uncertainty."
}

Respond with JSON only, no additional text.
"#
    }

    /// Builds the detection prompt from request.
    fn build_prompt(&self, request: &DetectionRequest) -> String {
        let template = self.prompt_template.as_deref()
            .unwrap_or(Self::default_prompt());

        let request_json = serde_json::to_string_pretty(request)
            .unwrap_or_else(|_| format!("{:?}", request));

        template.replace("{request}", &request_json)
    }

    /// Extracts text content from payload.
    fn extract_text_content(&self, payload: &Payload) -> String {
        payload.contents()
            .iter()
            .filter_map(|c| match c {
                PayloadContent::Text(t) => Some(t.as_str()),
                PayloadContent::Message { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Builds environment context summary.
    fn build_env_summary(&self, payload: &Payload) -> EnvContextSummary {
        if let Some(env) = payload.latest_env_context() {
            EnvContextSummary {
                redesign_count: env.redesign_count,
                has_journal: env.journal_summary.is_some(),
                success_rate: env.journal_summary.as_ref().map(|j| j.success_rate),
                consecutive_failures: env.journal_summary.as_ref()
                    .map(|j| j.consecutive_failures),
                strategy_phase: env.strategy_phase.clone(),
            }
        } else {
            EnvContextSummary {
                redesign_count: 0,
                has_journal: false,
                success_rate: None,
                consecutive_failures: None,
                strategy_phase: None,
            }
        }
    }

    /// Builds detected context summary from existing context.
    fn build_detected_summary(&self, payload: &Payload) -> Option<DetectedContextSummary> {
        payload.latest_detected_context().map(|detected| {
            DetectedContextSummary {
                task_type: detected.task_type.clone(),
                task_health: detected.task_health.map(|h| format!("{:?}", h)),
                user_states: detected.user_states.clone(),
            }
        })
    }

    /// Parses TaskHealth from string.
    fn parse_task_health(&self, s: &str) -> Option<TaskHealth> {
        match s.to_lowercase().as_str() {
            "ontrack" | "on-track" | "on_track" => Some(TaskHealth::OnTrack),
            "atrisk" | "at-risk" | "at_risk" => Some(TaskHealth::AtRisk),
            "offtrack" | "off-track" | "off_track" => Some(TaskHealth::OffTrack),
            _ => None,
        }
    }
}

#[async_trait]
impl<T> ContextDetector for AgentBasedDetector<T>
where
    T: Agent<Output = String>,
{
    async fn detect(&self, payload: &Payload) -> Result<DetectedContext, AgentError> {
        // Build detection request
        let request = DetectionRequest {
            text_content: self.extract_text_content(payload),
            env_summary: self.build_env_summary(payload),
            existing_context: self.build_detected_summary(payload),
        };

        // Build prompt
        let prompt = self.build_prompt(&request);
        let detection_payload = Payload::text(prompt);

        // Execute agent
        let agent_output = self.inner_agent
            .execute(detection_payload)
            .await?;

        // Parse response (Agent returns String, parse as JSON)
        let response: DetectionResponse = serde_json::from_str(&agent_output)
            .map_err(|e| AgentError::ExecutionFailed(format!(
                "Failed to parse detection response: {}. Output: {}", e, agent_output
            )))?;

        // Build DetectedContext from response
        let mut detected = DetectedContext::new();

        if let Some(task_type) = response.task_type {
            detected = detected.with_task_type(task_type);
        }

        if let Some(ref health_str) = response.task_health {
            if let Some(health) = self.parse_task_health(health_str) {
                detected = detected.with_task_health(health);
            }
        }

        for state in response.user_states {
            detected = detected.with_user_state(state);
        }

        detected = detected
            .with_confidence(response.confidence)
            .detected_by("AgentBasedDetector");

        Ok(detected)
    }

    fn name(&self) -> &str {
        "AgentBasedDetector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::EnvContext;

    // Mock agent for testing
    #[derive(Debug, Clone)]
    struct MockAgent {
        response: DetectionResponse,
    }

    #[async_trait]
    impl Agent for MockAgent {
        type Output = String;
        type Expertise = &'static str;

        async fn execute(&self, _payload: Payload) -> Result<String, AgentError> {
            Ok(serde_json::to_string(&self.response).unwrap())
        }

        fn name(&self) -> String {
            "MockAgent".to_string()
        }

        fn expertise(&self) -> &Self::Expertise {
            &"Mock agent for testing context detection"
        }
    }

    #[tokio::test]
    async fn test_agent_based_detection() {
        let mock_response = DetectionResponse {
            task_type: Some("security-review".to_string()),
            task_health: Some("AtRisk".to_string()),
            user_states: vec!["confused".to_string()],
            confidence: ConfidenceScores::new()
                .with_task_type(0.85)
                .with_task_health(0.75),
            reasoning: Some("Test reasoning".to_string()),
        };

        let agent = MockAgent { response: mock_response };
        let detector = AgentBasedDetector::new(agent);

        let env = EnvContext::new().with_redesign_count(3);
        let payload = Payload::text("Review this security code")
            .with_env_context(env);

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_type, Some("security-review".to_string()));
        assert_eq!(detected.task_health, Some(TaskHealth::AtRisk));
        assert_eq!(detected.user_states, vec!["confused"]);
        assert_eq!(detected.detected_by, vec!["AgentBasedDetector"]);
    }

    #[tokio::test]
    async fn test_layered_detection_with_existing_context() {
        // Simulate Layer 1 (Rule-based) already detected something
        let existing_detected = DetectedContext::new()
            .with_task_health(TaskHealth::AtRisk)
            .detected_by("RuleBasedDetector");

        let mock_response = DetectionResponse {
            task_type: Some("debug".to_string()),
            task_health: Some("AtRisk".to_string()),
            user_states: vec!["frustrated".to_string(), "blocked".to_string()],
            confidence: ConfidenceScores::new()
                .with_task_type(0.90)
                .with_user_states(0.85),
            reasoning: None,
        };

        let agent = MockAgent { response: mock_response };
        let detector = AgentBasedDetector::new(agent);

        let payload = Payload::text("This bug keeps coming back!")
            .with_detected_context(existing_detected);

        let detected = detector.detect(&payload).await.unwrap();

        assert_eq!(detected.task_type, Some("debug".to_string()));
        assert_eq!(detected.user_states, vec!["frustrated", "blocked"]);
        // Note: This creates a NEW DetectedContext, caller should merge
    }
}
