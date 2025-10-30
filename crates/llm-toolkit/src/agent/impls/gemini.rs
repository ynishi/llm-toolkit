//! GeminiAgent - A universal agent implementation that wraps the Gemini CLI.
//!
//! This agent can handle a wide variety of tasks by spawning the `gemini` command
//! with prompts and configuration options.

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::{debug, error, info, instrument};

use super::cli_agent::{CliAgent, CliAgentConfig};

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
    /// Common CLI agent configuration
    config: CliAgentConfig,
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
            config: CliAgentConfig::new(),
        }
    }

    /// Creates a new GeminiAgent with a custom path to the gemini executable.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            gemini_path: Some(path),
            model: GeminiModel::default(),
            config: CliAgentConfig::new(),
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

    /// Sets the execution profile.
    ///
    /// # Example
    /// ```rust,ignore
    /// use llm_toolkit::agent::ExecutionProfile;
    ///
    /// let agent = GeminiAgent::new()
    ///     .with_execution_profile(ExecutionProfile::Creative);
    /// ```
    pub fn with_execution_profile(mut self, profile: crate::agent::ExecutionProfile) -> Self {
        self.config = self.config.with_execution_profile(profile);
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
        self.config = self.config.with_cwd(path);
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
        self.config = self.config.with_directory(path);
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
        self.config = self.config.with_env(key, value);
        self
    }

    /// Sets multiple environment variables at once.
    ///
    /// # Example
    /// ```rust,ignore
    /// use std::collections::HashMap;
    ///
    /// let mut env_map = HashMap::new();
    /// env_map.insert("PATH".to_string(), "/custom/path".to_string());
    /// env_map.insert("GEMINI_API_KEY".to_string(), "key".to_string());
    ///
    /// let agent = GeminiAgent::new()
    ///     .with_envs(env_map);
    /// ```
    pub fn with_envs(mut self, envs: std::collections::HashMap<String, String>) -> Self {
        self.config = self.config.with_envs(envs);
        self
    }

    /// Clears all environment variables.
    pub fn clear_env(mut self) -> Self {
        self.config = self.config.clear_env();
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
        self.config = self.config.with_arg(arg);
        self
    }

    /// Adds multiple CLI arguments at once.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_args(vec!["--experimental".to_string(), "--verbose".to_string()]);
    /// ```
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.config = self.config.with_args(args);
        self
    }

    /// Sets the directory where attachment files will be written.
    ///
    /// If not specified, falls back to `working_dir` or system temp directory.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_attachment_dir("/tmp/my-attachments");
    /// ```
    pub fn with_attachment_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config = self.config.with_attachment_dir(path);
        self
    }

    /// Sets whether to keep temporary attachment files after execution.
    ///
    /// By default, temp files are deleted after each execution.
    /// Set to `true` to keep files for debugging purposes.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = GeminiAgent::new()
    ///     .with_keep_attachments(true); // Don't delete temp files
    /// ```
    pub fn with_keep_attachments(mut self, keep: bool) -> Self {
        self.config = self.config.with_keep_attachments(keep);
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
}

impl Default for GeminiAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl CliAgent for GeminiAgent {
    fn config(&self) -> &CliAgentConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut CliAgentConfig {
        &mut self.config
    }

    fn cli_path(&self) -> Option<&Path> {
        self.gemini_path.as_deref()
    }

    fn cli_command_name(&self) -> &str {
        "gemini"
    }

    fn build_command(&self, prompt: &str) -> Result<Command, AgentError> {
        let cmd_name = self
            .gemini_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "gemini".to_string());

        let mut cmd = Command::new(cmd_name);

        // Apply common configuration (working dir, env vars)
        self.config.apply_to_command(&mut cmd);

        // Add model argument
        cmd.arg("--model").arg(self.model.as_str());

        // Add extra CLI arguments from config
        for arg in &self.config.extra_args {
            debug!(
                target: "llm_toolkit::agent::gemini",
                "Adding extra argument: {}", arg
            );
            cmd.arg(arg);
        }

        // Add the prompt as positional argument
        cmd.arg(prompt);

        Ok(cmd)
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
        working_dir = ?self.config.working_dir,
        has_attachments = intent.has_attachments(),
        prompt_length = intent.to_text().len()
    ))]
    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        let payload = intent;

        // Process attachments using shared config method
        let (final_prompt, _temp_dir) = self.config.process_payload_attachments(&payload).await?;

        debug!(
            target = "llm_toolkit::agent::gemini",
            "Building gemini command with prompt length: {}",
            final_prompt.len()
        );

        let mut cmd = self.build_command(&final_prompt)?;

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
            let response = String::from_utf8(output.stdout).map_err(|e| {
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
                "Gemini command completed successfully, response length: {}",
                response.len()
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
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_gemini_agent_creation() {
        let agent = GeminiAgent::new();
        assert_eq!(agent.name(), "GeminiAgent");
        assert!(agent.config.working_dir.is_none());
        assert!(agent.config.env_vars.is_empty());
        assert!(agent.config.extra_args.is_empty());
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

        assert!(agent.config.working_dir.is_some());
        assert_eq!(
            agent.config.working_dir.unwrap(),
            PathBuf::from("/path/to/project")
        );
    }

    #[test]
    fn test_gemini_agent_with_directory() {
        let agent = GeminiAgent::new().with_directory("/path/to/project");

        assert!(agent.config.working_dir.is_some());
        assert_eq!(
            agent.config.working_dir.unwrap(),
            PathBuf::from("/path/to/project")
        );
    }

    #[test]
    fn test_gemini_agent_with_env() {
        let agent = GeminiAgent::new()
            .with_env("GEMINI_API_KEY", "my-key")
            .with_env("PATH", "/usr/local/bin");

        assert_eq!(agent.config.env_vars.len(), 2);
        assert_eq!(
            agent.config.env_vars.get("GEMINI_API_KEY"),
            Some(&"my-key".to_string())
        );
        assert_eq!(
            agent.config.env_vars.get("PATH"),
            Some(&"/usr/local/bin".to_string())
        );
    }

    #[test]
    fn test_gemini_agent_with_envs() {
        let mut env_map = HashMap::new();
        env_map.insert("KEY1".to_string(), "value1".to_string());
        env_map.insert("KEY2".to_string(), "value2".to_string());

        let agent = GeminiAgent::new().with_envs(env_map);

        assert_eq!(agent.config.env_vars.len(), 2);
        assert_eq!(
            agent.config.env_vars.get("KEY1"),
            Some(&"value1".to_string())
        );
        assert_eq!(
            agent.config.env_vars.get("KEY2"),
            Some(&"value2".to_string())
        );
    }

    #[test]
    fn test_gemini_agent_clear_env() {
        let agent = GeminiAgent::new()
            .with_env("KEY1", "value1")
            .with_env("KEY2", "value2")
            .clear_env();

        assert!(agent.config.env_vars.is_empty());
    }

    #[test]
    fn test_gemini_agent_with_arg() {
        let agent = GeminiAgent::new()
            .with_arg("--experimental")
            .with_arg("--timeout")
            .with_arg("60");

        assert_eq!(agent.config.extra_args.len(), 3);
        assert_eq!(agent.config.extra_args[0], "--experimental");
        assert_eq!(agent.config.extra_args[1], "--timeout");
        assert_eq!(agent.config.extra_args[2], "60");
    }

    #[test]
    fn test_gemini_agent_with_args() {
        let agent = GeminiAgent::new()
            .with_args(vec!["--experimental".to_string(), "--verbose".to_string()]);

        assert_eq!(agent.config.extra_args.len(), 2);
        assert_eq!(agent.config.extra_args[0], "--experimental");
        assert_eq!(agent.config.extra_args[1], "--verbose");
    }

    #[test]
    fn test_gemini_agent_builder_pattern_comprehensive() {
        let agent = GeminiAgent::new()
            .with_model(GeminiModel::Pro)
            .with_cwd("/project")
            .with_env("PATH", "/custom/path")
            .with_arg("--experimental");

        assert!(matches!(agent.model, GeminiModel::Pro));
        assert_eq!(agent.config.working_dir, Some(PathBuf::from("/project")));
        assert_eq!(
            agent.config.env_vars.get("PATH"),
            Some(&"/custom/path".to_string())
        );
        assert_eq!(agent.config.extra_args.len(), 1);
        assert_eq!(agent.config.extra_args[0], "--experimental");
    }
}
