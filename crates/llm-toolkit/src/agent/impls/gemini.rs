//! GeminiAgent - A universal agent implementation that wraps the Gemini CLI.
//!
//! This agent can handle a wide variety of tasks by spawning the `gemini` command
//! with prompts and configuration options.

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::process::Command;

/// Supported Gemini models
#[derive(Debug, Clone, Copy, Default)]
pub enum GeminiModel {
    /// Gemini 2.5 Flash - Fast and efficient
    #[default]
    Flash,
    /// Gemini 2.5 Pro - Most capable
    Pro,
}

impl GeminiModel {
    fn as_str(&self) -> &str {
        match self {
            GeminiModel::Flash => "gemini-2.5-flash",
            GeminiModel::Pro => "gemini-2.5-pro",
        }
    }
}

/// A general-purpose agent that executes tasks using the Gemini CLI.
///
/// This agent wraps the `gemini` command-line tool and can handle
/// coding, research, analysis, and other general tasks.
///
/// # Output
///
/// Returns the raw string output from Gemini. For structured output,
/// you can parse this string using `serde_json` or other parsers.
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::agent::impls::GeminiAgent;
/// use llm_toolkit::agent::Agent;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let agent = GeminiAgent::new()
///         .with_model(GeminiModel::Flash);
///
///     let result = agent.execute(
///         "Analyze the Rust ownership model and explain it in 3 bullet points".to_string()
///     ).await?;
///
///     println!("{}", result);
///     Ok(())
/// }
/// ```
pub struct GeminiAgent {
    /// Path to the `gemini` executable. If None, searches in PATH.
    gemini_path: Option<PathBuf>,
    /// Model to use for generation
    model: GeminiModel,
    /// Optional system prompt
    system_prompt: Option<String>,
    /// Skip loading GEMINI.md memory file
    skip_memory: bool,
    /// Execution profile for controlling behavior
    execution_profile: crate::agent::ExecutionProfile,
}

impl GeminiAgent {
    /// Creates a new GeminiAgent with default settings.
    ///
    /// By default:
    /// - Searches for `gemini` in the system PATH
    /// - Uses Gemini 2.5 Flash model
    /// - Skips memory loading (stateless)
    pub fn new() -> Self {
        Self {
            gemini_path: None,
            model: GeminiModel::default(),
            system_prompt: None,
            skip_memory: true,
            execution_profile: crate::agent::ExecutionProfile::default(),
        }
    }

    /// Creates a new GeminiAgent with a custom path to the gemini executable.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            gemini_path: Some(path),
            model: GeminiModel::default(),
            system_prompt: None,
            skip_memory: true,
            execution_profile: crate::agent::ExecutionProfile::default(),
        }
    }

    /// Sets the model to use.
    pub fn with_model(mut self, model: GeminiModel) -> Self {
        self.model = model;
        self
    }

    /// Sets the model using a string identifier.
    ///
    /// Accepts: "flash", "pro", "gemini-2.5-flash", "gemini-2.5-pro"
    pub fn with_model_str(mut self, model: &str) -> Self {
        self.model = match model {
            "flash" | "gemini-2.5-flash" => GeminiModel::Flash,
            "pro" | "gemini-2.5-pro" => GeminiModel::Pro,
            _ => GeminiModel::Flash, // Default fallback
        };
        self
    }

    /// Sets a system prompt for the agent.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Controls whether to skip loading GEMINI.md memory file.
    ///
    /// Default is `true` (skip memory for stateless execution).
    pub fn with_skip_memory(mut self, skip: bool) -> Self {
        self.skip_memory = skip;
        self
    }

    /// Sets the execution profile.
    ///
    /// The profile will be converted to Gemini-specific parameters:
    /// - Creative: temperature=0.9, top_p=0.95
    /// - Balanced: temperature=0.7, top_p=0.9 (default)
    /// - Deterministic: temperature=0.1, top_p=0.8
    pub fn with_execution_profile(mut self, profile: crate::agent::ExecutionProfile) -> Self {
        self.execution_profile = profile;
        self
    }

    /// Checks if the `gemini` CLI is available in the system (static version).
    ///
    /// Returns `true` if the command exists in PATH, `false` otherwise.
    pub fn is_available() -> bool {
        #[cfg(unix)]
        let check_cmd = "which";
        #[cfg(windows)]
        let check_cmd = "where";

        std::process::Command::new(check_cmd)
            .arg("gemini")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Checks availability using tokio (async version for trait implementation).
    async fn check_available() -> Result<(), AgentError> {
        #[cfg(unix)]
        let check_cmd = "which";
        #[cfg(windows)]
        let check_cmd = "where";

        let output = Command::new(check_cmd)
            .arg("gemini")
            .output()
            .await
            .map_err(|e| {
                AgentError::ProcessError(format!("Failed to check gemini availability: {}", e))
            })?;

        if output.status.success() {
            Ok(())
        } else {
            Err(AgentError::ExecutionFailed(
                "gemini CLI not found in PATH. Please install Gemini CLI.".to_string(),
            ))
        }
    }

    /// Builds the command with all arguments.
    fn build_command(&self, prompt: &str) -> Result<Command, AgentError> {
        let cmd_name = self
            .gemini_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "gemini".to_string());

        let mut cmd = Command::new(cmd_name);

        // Add model
        cmd.arg("--model").arg(self.model.as_str());

        // Add execution profile parameters
        let (temperature, top_p) = match self.execution_profile {
            crate::agent::ExecutionProfile::Creative => (0.9, 0.95),
            crate::agent::ExecutionProfile::Balanced => (0.7, 0.9),
            crate::agent::ExecutionProfile::Deterministic => (0.1, 0.8),
        };
        cmd.arg("--temperature")
            .arg(temperature.to_string())
            .arg("--top-p")
            .arg(top_p.to_string());

        log::debug!(
            "GeminiAgent: Using execution profile {:?} (temperature={}, top_p={})",
            self.execution_profile,
            temperature,
            top_p
        );

        // Add skip-memory flag if enabled
        if self.skip_memory {
            cmd.arg("--skip-memory");
        }

        // Add system prompt if provided
        if let Some(sys_prompt) = &self.system_prompt {
            // Write system prompt to temp file
            let temp_dir = std::env::temp_dir();
            let temp_file =
                temp_dir.join(format!("gemini_system_prompt_{}.md", std::process::id()));

            std::fs::write(&temp_file, sys_prompt).map_err(|e| {
                AgentError::ExecutionFailed(format!("Failed to write system prompt file: {}", e))
            })?;

            cmd.arg("--system-prompt-file").arg(&temp_file);
        }

        // Add the prompt
        cmd.arg("--prompt").arg(prompt);

        Ok(cmd)
    }
}

impl Default for GeminiAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for GeminiAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "General-purpose AI assistant powered by Google Gemini, capable of coding, analysis, and research tasks"
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let payload = intent;

        // Extract text content for now (images not yet supported in this integration)
        let text_intent = payload.to_text();

        if payload.has_images() {
            log::warn!("GeminiAgent: Images in payload are not yet supported and will be ignored");
        }

        let mut cmd = self.build_command(&text_intent)?;

        let output = cmd.output().await.map_err(|e| {
            AgentError::ExecutionFailed(format!("Failed to execute gemini command: {}", e))
        })?;

        if output.status.success() {
            let response = String::from_utf8(output.stdout).map_err(|e| {
                AgentError::ParseError(format!("Failed to parse gemini stdout: {}", e))
            })?;

            Ok(response.trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(AgentError::ExecutionFailed(format!(
                "Gemini command failed: {}",
                stderr
            )))
        }
    }

    fn name(&self) -> String {
        "GeminiAgent".to_string()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        Self::check_available().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_agent_creation() {
        let agent = GeminiAgent::new();
        assert_eq!(agent.name(), "GeminiAgent");
        assert!(agent.skip_memory);
    }

    #[test]
    fn test_gemini_agent_with_model() {
        let agent = GeminiAgent::new().with_model(GeminiModel::Pro);

        assert!(matches!(agent.model, GeminiModel::Pro));
    }

    #[test]
    fn test_gemini_agent_with_model_str() {
        let agent = GeminiAgent::new().with_model_str("pro");

        assert!(matches!(agent.model, GeminiModel::Pro));
    }

    #[test]
    fn test_gemini_agent_with_system_prompt() {
        let agent = GeminiAgent::new().with_system_prompt("You are a helpful assistant");

        assert!(agent.system_prompt.is_some());
    }

    #[test]
    fn test_gemini_agent_with_execution_profile() {
        let agent =
            GeminiAgent::new().with_execution_profile(crate::agent::ExecutionProfile::Creative);

        assert!(matches!(
            agent.execution_profile,
            crate::agent::ExecutionProfile::Creative
        ));
    }

    #[test]
    fn test_gemini_agent_default_profile() {
        let agent = GeminiAgent::new();
        assert!(matches!(
            agent.execution_profile,
            crate::agent::ExecutionProfile::Balanced
        ));
    }

    #[test]
    fn test_gemini_agent_execution_profile_parameters() {
        // Test that different profiles result in different parameter conversions
        let creative =
            GeminiAgent::new().with_execution_profile(crate::agent::ExecutionProfile::Creative);
        let balanced =
            GeminiAgent::new().with_execution_profile(crate::agent::ExecutionProfile::Balanced);
        let deterministic = GeminiAgent::new()
            .with_execution_profile(crate::agent::ExecutionProfile::Deterministic);

        // Build commands and verify parameters are set correctly
        // (We can't easily test the actual command without mocking, but we can verify the profile is stored)
        assert!(matches!(
            creative.execution_profile,
            crate::agent::ExecutionProfile::Creative
        ));
        assert!(matches!(
            balanced.execution_profile,
            crate::agent::ExecutionProfile::Balanced
        ));
        assert!(matches!(
            deterministic.execution_profile,
            crate::agent::ExecutionProfile::Deterministic
        ));
    }

    #[test]
    fn test_gemini_agent_builder_pattern_with_profile() {
        let agent = GeminiAgent::new()
            .with_model(GeminiModel::Pro)
            .with_execution_profile(crate::agent::ExecutionProfile::Deterministic)
            .with_system_prompt("Test prompt");

        assert!(matches!(agent.model, GeminiModel::Pro));
        assert!(matches!(
            agent.execution_profile,
            crate::agent::ExecutionProfile::Deterministic
        ));
        assert!(agent.system_prompt.is_some());
    }
}
