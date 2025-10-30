//! ClaudeCodeAgent - A universal agent implementation that wraps the Claude CLI.
//!
//! This agent can handle a wide variety of tasks by spawning the `claude` command
//! with the `-p` flag to pass prompts directly.

use crate::agent::{Agent, AgentError, Payload};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::{debug, error, info, instrument};

use super::cli_agent::{CliAgent, CliAgentConfig};

/// Supported Claude models
#[derive(Debug, Clone, Copy, Default)]
pub enum ClaudeModel {
    /// Claude Sonnet 4.5 - Balanced performance and speed
    #[default]
    Sonnet45,
    /// Claude Sonnet 4 - Previous generation balanced model
    Sonnet4,
    /// Claude Opus 4 - Most capable model
    Opus4,
}

impl ClaudeModel {
    fn as_str(&self) -> &str {
        match self {
            ClaudeModel::Sonnet45 => "claude-sonnet-4.5",
            ClaudeModel::Sonnet4 => "claude-sonnet-4",
            ClaudeModel::Opus4 => "claude-opus-4",
        }
    }
}

/// A general-purpose agent that executes tasks using the Claude CLI.
///
/// This agent wraps the `claude` command-line tool and can handle
/// coding, research, analysis, and other general tasks.
///
/// # Output
///
/// Returns the raw string output from Claude. For structured output,
/// you can parse this string using `serde_json` or other parsers.
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Agent, ClaudeCodeAgent};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let agent = ClaudeCodeAgent::new();
///
///     let result = agent.execute(
///         "Analyze the Rust ownership model and explain it in 3 bullet points".to_string()
///     ).await?;
///
///     println!("{}", result);
///     Ok(())
/// }
/// ```
pub struct ClaudeCodeAgent {
    /// Path to the `claude` executable. If None, searches in PATH.
    claude_path: Option<PathBuf>,
    /// Model to use for generation
    model: Option<ClaudeModel>,
    /// Common CLI agent configuration
    config: CliAgentConfig,
}

impl ClaudeCodeAgent {
    /// Creates a new ClaudeCodeAgent with default settings.
    ///
    /// By default:
    /// - Searches for `claude` in the system PATH
    /// - Uses default model (Sonnet 4.5)
    /// - No working directory specified (uses current directory)
    /// - No additional environment variables
    /// - No extra CLI arguments
    pub fn new() -> Self {
        Self {
            claude_path: None,
            model: None,
            config: CliAgentConfig::new(),
        }
    }

    /// Creates a new ClaudeCodeAgent with a custom path to the claude executable.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            claude_path: Some(path),
            model: None,
            config: CliAgentConfig::new(),
        }
    }

    /// Sets the model to use.
    pub fn with_model(mut self, model: ClaudeModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the model using a string identifier.
    ///
    /// Accepts: "sonnet", "sonnet-4.5", "sonnet-4", "opus", "opus-4", etc.
    pub fn with_model_str(mut self, model: &str) -> Self {
        self.model = Some(match model {
            "sonnet" | "sonnet-4.5" | "claude-sonnet-4.5" => ClaudeModel::Sonnet45,
            "sonnet-4" | "claude-sonnet-4" => ClaudeModel::Sonnet4,
            "opus" | "opus-4" | "claude-opus-4" => ClaudeModel::Opus4,
            _ => ClaudeModel::Sonnet45, // Default fallback
        });
        self
    }

    /// Sets the execution profile.
    ///
    /// # Example
    /// ```rust,ignore
    /// use llm_toolkit::agent::ExecutionProfile;
    ///
    /// let agent = ClaudeCodeAgent::new()
    ///     .with_execution_profile(ExecutionProfile::Creative);
    /// ```
    pub fn with_execution_profile(mut self, profile: crate::agent::ExecutionProfile) -> Self {
        self.config = self.config.with_execution_profile(profile);
        self
    }

    /// Sets the working directory where the claude command will be executed.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = ClaudeCodeAgent::new()
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
    /// let agent = ClaudeCodeAgent::new()
    ///     .with_directory("/path/to/project");
    /// ```
    pub fn with_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.config = self.config.with_directory(path);
        self
    }

    /// Sets a single environment variable for the claude command.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = ClaudeCodeAgent::new()
    ///     .with_env("CLAUDE_API_KEY", "my-key")
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
    /// env_map.insert("CLAUDE_API_KEY".to_string(), "key".to_string());
    ///
    /// let agent = ClaudeCodeAgent::new()
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

    /// Adds a single CLI argument to pass to the claude command.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = ClaudeCodeAgent::new()
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
    /// let agent = ClaudeCodeAgent::new()
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
    /// let agent = ClaudeCodeAgent::new()
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
    /// let agent = ClaudeCodeAgent::new()
    ///     .with_keep_attachments(true); // Don't delete temp files
    /// ```
    pub fn with_keep_attachments(mut self, keep: bool) -> Self {
        self.config = self.config.with_keep_attachments(keep);
        self
    }

    /// Checks if the `claude` CLI is available in the system (static version).
    ///
    /// Returns `true` if the command exists in PATH, `false` otherwise.
    /// Uses `which` on Unix/macOS or `where` on Windows for a quick check.
    pub fn is_available() -> bool {
        #[cfg(unix)]
        let check_cmd = "which";
        #[cfg(windows)]
        let check_cmd = "where";

        std::process::Command::new(check_cmd)
            .arg("claude")
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
            .arg("claude")
            .output()
            .await
            .map_err(|e| AgentError::ProcessError {
                status_code: None,
                message: format!("Failed to check claude availability: {}", e),
                is_retryable: true,
                retry_after: None,
            })?;

        if output.status.success() {
            Ok(())
        } else {
            Err(AgentError::ExecutionFailed(
                "claude CLI not found in PATH. Please install Claude CLI.".to_string(),
            ))
        }
    }
}

impl Default for ClaudeCodeAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl CliAgent for ClaudeCodeAgent {
    fn config(&self) -> &CliAgentConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut CliAgentConfig {
        &mut self.config
    }

    fn cli_path(&self) -> Option<&Path> {
        self.claude_path.as_deref()
    }

    fn cli_command_name(&self) -> &str {
        "claude"
    }

    fn build_command(&self, prompt: &str) -> Result<Command, AgentError> {
        let cmd_name = self
            .claude_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "claude".to_string());

        let mut cmd = Command::new(cmd_name);

        // Apply common configuration (working dir, env vars)
        self.config.apply_to_command(&mut cmd);

        // Add prompt argument
        cmd.arg("-p").arg(prompt);

        // Add model if specified
        if let Some(model) = &self.model {
            cmd.arg("--model").arg(model.as_str());
            debug!(
                target: "llm_toolkit::agent::claude_code",
                "Using model: {}", model.as_str()
            );
        }

        // Add extra CLI arguments from config
        for arg in &self.config.extra_args {
            debug!(
                target: "llm_toolkit::agent::claude_code",
                "Adding extra argument: {}", arg
            );
            cmd.arg(arg);
        }

        Ok(cmd)
    }
}

#[async_trait]
impl Agent for ClaudeCodeAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "A general-purpose AI agent capable of coding, research, analysis, \
         writing, and problem-solving across various domains. Can handle \
         complex multi-step tasks autonomously."
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
            target: "llm_toolkit::agent::claude_code",
            "Building claude command with prompt length: {}", final_prompt.len()
        );

        let mut cmd = self.build_command(&final_prompt)?;

        debug!(
            target: "llm_toolkit::agent::claude_code",
            "Executing claude command: {:?}", cmd
        );

        let output = cmd.output().await.map_err(|e| {
            error!(
                target: "llm_toolkit::agent::claude_code",
                "Failed to execute claude command: {}", e
            );
            AgentError::ProcessError {
                status_code: None,
                message: format!(
                    "Failed to spawn claude process: {}. \
                     Make sure 'claude' is installed and in PATH.",
                    e
                ),
                is_retryable: true,
                retry_after: None,
            }
        })?;

        if output.status.success() {
            let stdout = String::from_utf8(output.stdout).map_err(|e| {
                error!(
                    target: "llm_toolkit::agent::claude_code",
                    "Failed to parse stdout as UTF-8: {}", e
                );
                AgentError::Other(format!("Failed to parse claude output as UTF-8: {}", e))
            })?;

            info!(
                target: "llm_toolkit::agent::claude_code",
                "Claude command completed successfully, response length: {}", stdout.len()
            );
            Ok(stdout)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!(
                target: "llm_toolkit::agent::claude_code",
                "Claude command failed with stderr: {}", stderr
            );
            Err(AgentError::ExecutionFailed(format!(
                "Claude command failed with status {}: {}",
                output.status, stderr
            )))
        }
    }

    fn name(&self) -> String {
        "ClaudeCodeAgent".to_string()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        Self::check_available().await
    }
}

/// A typed variant of ClaudeCodeAgent that attempts to parse JSON output.
///
/// This agent is useful when you expect structured output from Claude.
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::agent::{Agent, ClaudeCodeJsonAgent};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Deserialize, Serialize)]
/// struct Analysis {
///     summary: String,
///     key_points: Vec<String>,
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let agent = ClaudeCodeJsonAgent::<Analysis>::new();
///
///     let result = agent.execute(
///         "Analyze Rust's ownership model and return JSON with 'summary' \
///          and 'key_points' (array of strings)".to_string()
///     ).await?;
///
///     println!("Summary: {}", result.summary);
///     Ok(())
/// }
/// ```
pub struct ClaudeCodeJsonAgent<T> {
    inner: ClaudeCodeAgent,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ClaudeCodeJsonAgent<T>
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    /// Creates a new ClaudeCodeJsonAgent.
    pub fn new() -> Self {
        Self {
            inner: ClaudeCodeAgent::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new ClaudeCodeJsonAgent with a custom path.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            inner: ClaudeCodeAgent::with_path(path),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sets the model to use.
    pub fn with_model(mut self, model: ClaudeModel) -> Self {
        self.inner = self.inner.with_model(model);
        self
    }

    /// Sets the model using a string identifier.
    pub fn with_model_str(mut self, model: &str) -> Self {
        self.inner = self.inner.with_model_str(model);
        self
    }

    /// Sets the working directory where the claude command will be executed.
    pub fn with_cwd(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.with_cwd(path);
        self
    }

    /// Alias for `with_cwd` using more explicit name.
    pub fn with_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.with_directory(path);
        self
    }

    /// Sets a single environment variable for the claude command.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner = self.inner.with_env(key, value);
        self
    }

    /// Sets multiple environment variables at once.
    pub fn with_envs(mut self, envs: HashMap<String, String>) -> Self {
        self.inner = self.inner.with_envs(envs);
        self
    }

    /// Clears all environment variables.
    pub fn clear_env(mut self) -> Self {
        self.inner = self.inner.clear_env();
        self
    }

    /// Adds a single CLI argument to pass to the claude command.
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.inner = self.inner.with_arg(arg);
        self
    }

    /// Adds multiple CLI arguments at once.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.inner = self.inner.with_args(args);
        self
    }

    /// Sets the directory where attachment files will be written.
    pub fn with_attachment_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.with_attachment_dir(path);
        self
    }

    /// Sets whether to keep temporary attachment files after execution.
    pub fn with_keep_attachments(mut self, keep: bool) -> Self {
        self.inner = self.inner.with_keep_attachments(keep);
        self
    }
}

impl<T> Default for ClaudeCodeJsonAgent<T>
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T> Agent for ClaudeCodeJsonAgent<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    type Output = T;

    fn expertise(&self) -> &str {
        self.inner.expertise()
    }

    async fn execute(&self, intent: Payload) -> Result<Self::Output, AgentError> {
        log::info!(
            "📊 ClaudeCodeJsonAgent<{}> executing...",
            std::any::type_name::<T>()
        );

        let raw_output = self.inner.execute(intent).await?;

        log::debug!("Extracting JSON from raw output...");

        // Try to extract JSON from the output (might be wrapped in markdown, etc.)
        let json_str = crate::extract_json(&raw_output).map_err(|e| {
            log::error!("Failed to extract JSON: {}", e);
            AgentError::ParseError {
                message: format!(
                    "Failed to extract JSON from claude output: {}. Raw output: {}",
                    e, raw_output
                ),
                reason: crate::agent::error::ParseErrorReason::MarkdownExtractionFailed,
            }
        })?;

        log::debug!("Parsing JSON into {}...", std::any::type_name::<T>());

        let result = serde_json::from_str(&json_str).map_err(|e| {
            log::error!("Failed to parse JSON: {}", e);

            // Determine the parse error reason based on serde_json error type
            let reason = if e.is_eof() {
                crate::agent::error::ParseErrorReason::UnexpectedEof
            } else if e.is_syntax() {
                crate::agent::error::ParseErrorReason::InvalidJson
            } else {
                crate::agent::error::ParseErrorReason::SchemaMismatch
            };

            AgentError::ParseError {
                message: format!("Failed to parse JSON: {}. Extracted JSON: {}", e, json_str),
                reason,
            }
        })?;

        log::info!(
            "✅ ClaudeCodeJsonAgent<{}> completed",
            std::any::type_name::<T>()
        );

        Ok(result)
    }

    fn name(&self) -> String {
        format!("ClaudeCodeJsonAgent<{}>", std::any::type_name::<T>())
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        self.inner.is_available().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_code_agent_creation() {
        let agent = ClaudeCodeAgent::new();
        assert_eq!(agent.name(), "ClaudeCodeAgent");
        assert!(!agent.expertise().is_empty());
    }

    #[test]
    fn test_claude_code_agent_with_path() {
        let path = PathBuf::from("/usr/local/bin/claude");
        let agent = ClaudeCodeAgent::with_path(path.clone());
        assert_eq!(agent.claude_path, Some(path));
    }

    #[test]
    fn test_claude_code_agent_with_cwd() {
        let agent = ClaudeCodeAgent::new().with_cwd("/path/to/project");

        assert!(agent.config.working_dir.is_some());
        assert_eq!(
            agent.config.working_dir.unwrap(),
            PathBuf::from("/path/to/project")
        );
    }

    #[test]
    fn test_claude_code_agent_with_directory() {
        let agent = ClaudeCodeAgent::new().with_directory("/path/to/project");

        assert!(agent.config.working_dir.is_some());
        assert_eq!(
            agent.config.working_dir.unwrap(),
            PathBuf::from("/path/to/project")
        );
    }

    #[test]
    fn test_claude_code_agent_with_env() {
        let agent = ClaudeCodeAgent::new()
            .with_env("CLAUDE_API_KEY", "my-key")
            .with_env("PATH", "/usr/local/bin");

        assert_eq!(agent.config.env_vars.len(), 2);
        assert_eq!(
            agent.config.env_vars.get("CLAUDE_API_KEY"),
            Some(&"my-key".to_string())
        );
        assert_eq!(
            agent.config.env_vars.get("PATH"),
            Some(&"/usr/local/bin".to_string())
        );
    }

    #[test]
    fn test_claude_code_agent_with_envs() {
        let mut env_map = HashMap::new();
        env_map.insert("KEY1".to_string(), "value1".to_string());
        env_map.insert("KEY2".to_string(), "value2".to_string());

        let agent = ClaudeCodeAgent::new().with_envs(env_map);

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
    fn test_claude_code_agent_clear_env() {
        let agent = ClaudeCodeAgent::new()
            .with_env("KEY1", "value1")
            .with_env("KEY2", "value2")
            .clear_env();

        assert!(agent.config.env_vars.is_empty());
    }

    #[test]
    fn test_claude_code_agent_with_arg() {
        let agent = ClaudeCodeAgent::new()
            .with_arg("--experimental")
            .with_arg("--timeout")
            .with_arg("60");

        assert_eq!(agent.config.extra_args.len(), 3);
        assert_eq!(agent.config.extra_args[0], "--experimental");
        assert_eq!(agent.config.extra_args[1], "--timeout");
        assert_eq!(agent.config.extra_args[2], "60");
    }

    #[test]
    fn test_claude_code_agent_with_args() {
        let agent = ClaudeCodeAgent::new()
            .with_args(vec!["--experimental".to_string(), "--verbose".to_string()]);

        assert_eq!(agent.config.extra_args.len(), 2);
        assert_eq!(agent.config.extra_args[0], "--experimental");
        assert_eq!(agent.config.extra_args[1], "--verbose");
    }

    #[test]
    fn test_claude_code_agent_builder_pattern() {
        let agent = ClaudeCodeAgent::new()
            .with_model(ClaudeModel::Opus4)
            .with_cwd("/project")
            .with_env("PATH", "/custom/path")
            .with_arg("--experimental");

        assert!(matches!(agent.model, Some(ClaudeModel::Opus4)));
        assert_eq!(agent.config.working_dir, Some(PathBuf::from("/project")));
        assert_eq!(
            agent.config.env_vars.get("PATH"),
            Some(&"/custom/path".to_string())
        );
        assert_eq!(agent.config.extra_args.len(), 1);
        assert_eq!(agent.config.extra_args[0], "--experimental");
    }
}
