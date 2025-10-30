//! GeminiAgent - A universal agent implementation that wraps the Gemini CLI.
//!
//! This agent can handle a wide variety of tasks by spawning the `gemini` command
//! with prompts and configuration options.

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::process::Command;
use tracing::{debug, error, info, instrument, warn};

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
    /// Working directory for command execution
    working_dir: Option<PathBuf>,
    /// Environment variables to set for command execution
    env_vars: HashMap<String, String>,
    /// Additional CLI arguments to pass to gemini command
    extra_args: Vec<String>,
}

impl GeminiAgent {
    /// Creates a new GeminiAgent with default settings.
    ///
    /// By default:
    /// - Searches for `gemini` in the system PATH
    /// - Uses Gemini 2.5 Flash model
    /// - No working directory specified (uses current directory)
    /// - No additional environment variables
    /// - No extra CLI arguments
    pub fn new() -> Self {
        Self {
            gemini_path: None,
            model: GeminiModel::default(),
            working_dir: None,
            env_vars: HashMap::new(),
            extra_args: Vec::new(),
        }
    }

    /// Creates a new GeminiAgent with a custom path to the gemini executable.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            gemini_path: Some(path),
            model: GeminiModel::default(),
            working_dir: None,
            env_vars: HashMap::new(),
            extra_args: Vec::new(),
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

    /// Sets the working directory where the gemini command will be executed.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_cwd("/path/to/project");
    /// ```
    pub fn with_cwd(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }

    /// Alias for `with_cwd` using more explicit name.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_directory("/path/to/project");
    /// ```
    pub fn with_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }

    /// Sets a single environment variable for the gemini command.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_env("GEMINI_API_KEY", "my-key")
    ///     .with_env("PATH", "/usr/local/bin:/usr/bin");
    /// ```
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Sets multiple environment variables at once.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut env_map = HashMap::new();
    /// env_map.insert("PATH".to_string(), "/custom/path".to_string());
    /// env_map.insert("GEMINI_API_KEY".to_string(), "key".to_string());
    ///
    /// let agent = GeminiAgent::new()
    ///     .with_envs(env_map);
    /// ```
    pub fn with_envs(mut self, envs: HashMap<String, String>) -> Self {
        self.env_vars.extend(envs);
        self
    }

    /// Clears all environment variables.
    pub fn clear_env(mut self) -> Self {
        self.env_vars.clear();
        self
    }

    /// Adds a single CLI argument to pass to the gemini command.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_arg("--experimental")
    ///     .with_arg("--timeout")
    ///     .with_arg("60");
    /// ```
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_args.push(arg.into());
        self
    }

    /// Adds multiple CLI arguments at once.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_args(vec!["--experimental", "--verbose"]);
    /// ```
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.extra_args.extend(args);
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
            .map_err(|e| AgentError::ProcessError {
                status_code: None,
                message: format!("Failed to check gemini availability: {}", e),
                is_retryable: true,
                retry_after: None,
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

        // Set working directory if specified
        if let Some(dir) = &self.working_dir {
            debug!("Setting working directory: {}", dir.display());
            cmd.current_dir(dir);
        }

        // Apply environment variables
        for (key, value) in &self.env_vars {
            debug!("Setting environment variable: {}={}", key, value);
            cmd.env(key, value);
        }

        // Add model argument
        cmd.arg("--model").arg(self.model.as_str());

        // Add extra CLI arguments
        for arg in &self.extra_args {
            debug!("Adding extra argument: {}", arg);
            cmd.arg(arg);
        }

        // Add the prompt as positional argument
        cmd.arg(prompt);

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

    #[instrument(skip(self, intent), fields(
        model = ?self.model,
        working_dir = ?self.working_dir,
        has_attachments = intent.has_attachments(),
        prompt_length = intent.to_text().len()
    ))]
    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let payload = intent;

        // Extract text content for now (attachments not yet supported in this integration)
        let text_intent = payload.to_text();

        if payload.has_attachments() {
            warn!(
                target = "llm_toolkit::agent::gemini",
                "Attachments in payload are not yet supported and will be ignored"
            );
        }

        debug!(
            target = "llm_toolkit::agent::gemini",
            "Building gemini command with prompt length: {}", text_intent.len()
        );

        let mut cmd = self.build_command(&text_intent)?;

        debug!(
            target = "llm_toolkit::agent::gemini",
            "Executing gemini command: {:?}", cmd
        );

        let output = cmd.output().await.map_err(|e| {
            error!(
                target = "llm_toolkit::agent::gemini",
                "Failed to execute gemini command: {}", e
            );
            AgentError::ExecutionFailed(format!("Failed to execute gemini command: {}", e))
        })?;

        if output.status.success() {
            let response =
                String::from_utf8(output.stdout).map_err(|e| {
                    error!(
                        target = "llm_toolkit::agent::gemini",
                        "Failed to parse stdout as UTF-8: {}", e
                    );
                    AgentError::ParseError {
                        message: format!("Failed to parse gemini stdout: {}", e),
                        reason: crate::agent::error::ParseErrorReason::UnexpectedEof,
                    }
                })?;

            info!(
                target = "llm_toolkit::agent::gemini",
                "Gemini command completed successfully, response length: {}", response.len()
            );
            Ok(response.trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!(
                target = "llm_toolkit::agent::gemini",
                "Gemini command failed with stderr: {}", stderr
            );
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
        assert!(agent.working_dir.is_none());
        assert!(agent.env_vars.is_empty());
        assert!(agent.extra_args.is_empty());
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
    fn test_gemini_agent_with_cwd() {
        let agent = GeminiAgent::new().with_cwd("/path/to/project");

        assert!(agent.working_dir.is_some());
        assert_eq!(
            agent.working_dir.unwrap(),
            PathBuf::from("/path/to/project")
        );
    }

    #[test]
    fn test_gemini_agent_with_directory() {
        let agent = GeminiAgent::new().with_directory("/path/to/project");

        assert!(agent.working_dir.is_some());
        assert_eq!(
            agent.working_dir.unwrap(),
            PathBuf::from("/path/to/project")
        );
    }

    #[test]
    fn test_gemini_agent_with_env() {
        let agent = GeminiAgent::new()
            .with_env("GEMINI_API_KEY", "my-key")
            .with_env("PATH", "/usr/local/bin");

        assert_eq!(agent.env_vars.len(), 2);
        assert_eq!(agent.env_vars.get("GEMINI_API_KEY"), Some(&"my-key".to_string()));
        assert_eq!(agent.env_vars.get("PATH"), Some(&"/usr/local/bin".to_string()));
    }

    #[test]
    fn test_gemini_agent_with_envs() {
        let mut env_map = HashMap::new();
        env_map.insert("KEY1".to_string(), "value1".to_string());
        env_map.insert("KEY2".to_string(), "value2".to_string());

        let agent = GeminiAgent::new().with_envs(env_map);

        assert_eq!(agent.env_vars.len(), 2);
        assert_eq!(agent.env_vars.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(agent.env_vars.get("KEY2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_gemini_agent_clear_env() {
        let agent = GeminiAgent::new()
            .with_env("KEY1", "value1")
            .with_env("KEY2", "value2")
            .clear_env();

        assert!(agent.env_vars.is_empty());
    }

    #[test]
    fn test_gemini_agent_with_arg() {
        let agent = GeminiAgent::new()
            .with_arg("--experimental")
            .with_arg("--timeout")
            .with_arg("60");

        assert_eq!(agent.extra_args.len(), 3);
        assert_eq!(agent.extra_args[0], "--experimental");
        assert_eq!(agent.extra_args[1], "--timeout");
        assert_eq!(agent.extra_args[2], "60");
    }

    #[test]
    fn test_gemini_agent_with_args() {
        let agent = GeminiAgent::new()
            .with_args(vec!["--experimental".to_string(), "--verbose".to_string()]);

        assert_eq!(agent.extra_args.len(), 2);
        assert_eq!(agent.extra_args[0], "--experimental");
        assert_eq!(agent.extra_args[1], "--verbose");
    }

    #[test]
    fn test_gemini_agent_builder_pattern_comprehensive() {
        let agent = GeminiAgent::new()
            .with_model(GeminiModel::Pro)
            .with_cwd("/project")
            .with_env("PATH", "/custom/path")
            .with_arg("--experimental");

        assert!(matches!(agent.model, GeminiModel::Pro));
        assert_eq!(agent.working_dir, Some(PathBuf::from("/project")));
        assert_eq!(agent.env_vars.get("PATH"), Some(&"/custom/path".to_string()));
        assert_eq!(agent.extra_args.len(), 1);
        assert_eq!(agent.extra_args[0], "--experimental");
    }
}
