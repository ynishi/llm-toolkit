//! Common trait and utilities for CLI-based agents.
//!
//! This module provides shared functionality for agents that wrap external
//! command-line tools like `gemini` and `claude`.

#![allow(clippy::result_large_err)]

use crate::agent::{AgentError, ExecutionProfile, Payload};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::debug;

use super::cli_attachment::{
    TempAttachmentDir, format_prompt_with_attachments, process_attachments,
};

/// Common configuration for CLI-based agents.
///
/// This struct encapsulates shared settings that are common across
/// all CLI-based agents (e.g., GeminiAgent, ClaudeCodeAgent).
#[derive(Debug, Clone, Default)]
pub struct CliAgentConfig {
    /// Execution profile controlling agent behavior
    pub execution_profile: ExecutionProfile,
    /// Working directory for command execution
    pub working_dir: Option<PathBuf>,
    /// Environment variables to set for command execution
    pub env_vars: HashMap<String, String>,
    /// Additional CLI arguments to pass to command
    pub extra_args: Vec<String>,
    /// Directory for storing temporary attachment files
    pub attachment_dir: Option<PathBuf>,
    /// Whether to keep attachment files after execution (for debugging)
    pub keep_attachments: bool,
}

impl CliAgentConfig {
    /// Creates a new CliAgentConfig with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the execution profile.
    ///
    /// The execution profile controls the agent's behavior and can influence
    /// parameters like temperature, creativity, etc. (if supported by the CLI tool).
    pub fn with_execution_profile(mut self, profile: ExecutionProfile) -> Self {
        self.execution_profile = profile;
        self
    }

    /// Sets the working directory where the command will be executed.
    pub fn with_cwd(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }

    /// Alias for `with_cwd` using more explicit name.
    pub fn with_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }

    /// Sets a single environment variable for the command.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Sets multiple environment variables at once.
    pub fn with_envs(mut self, envs: HashMap<String, String>) -> Self {
        self.env_vars.extend(envs);
        self
    }

    /// Clears all environment variables.
    pub fn clear_env(mut self) -> Self {
        self.env_vars.clear();
        self
    }

    /// Adds a single CLI argument to pass to the command.
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_args.push(arg.into());
        self
    }

    /// Adds multiple CLI arguments at once.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.extra_args.extend(args);
        self
    }

    /// Sets the directory where attachment files will be written.
    pub fn with_attachment_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.attachment_dir = Some(path.into());
        self
    }

    /// Sets whether to keep temporary attachment files after execution.
    pub fn with_keep_attachments(mut self, keep: bool) -> Self {
        self.keep_attachments = keep;
        self
    }

    /// Applies configuration to a Command.
    ///
    /// Sets working directory and environment variables on the command.
    pub fn apply_to_command(&self, cmd: &mut Command) {
        // Set working directory if specified
        if let Some(dir) = &self.working_dir {
            debug!(
                target: "llm_toolkit::agent::cli_agent",
                "Setting working directory: {}", dir.display()
            );
            cmd.current_dir(dir);
        }

        // Apply environment variables
        for (key, value) in &self.env_vars {
            debug!(
                target: "llm_toolkit::agent::cli_agent",
                "Setting environment variable: {}={}", key, value
            );
            cmd.env(key, value);
        }
    }

    /// Processes payload attachments and returns the final prompt with attachment paths.
    ///
    /// If the payload has attachments:
    /// 1. Creates a temporary directory
    /// 2. Writes/copies attachments to temp files
    /// 3. Formats prompt with attachment file paths appended
    /// 4. Returns (formatted_prompt, Some(temp_dir_guard))
    ///
    /// If no attachments:
    /// - Returns (original_prompt, None)
    ///
    /// The TempAttachmentDir guard ensures cleanup on drop unless keep_attachments is true.
    pub(crate) async fn process_payload_attachments(
        &self,
        payload: &Payload,
    ) -> Result<(String, Option<TempAttachmentDir>), AgentError> {
        let text_intent = payload.to_text();

        if payload.has_attachments() {
            debug!(
                target: "llm_toolkit::agent::cli_agent",
                "Processing {} attachments", payload.attachments().len()
            );

            // Determine base directory for attachments
            let base_dir = self
                .attachment_dir
                .as_ref()
                .or(self.working_dir.as_ref())
                .cloned()
                .unwrap_or_else(std::env::temp_dir);

            // Create temp directory (will auto-cleanup on drop unless keep_attachments is true)
            let temp_dir =
                TempAttachmentDir::new(&base_dir, self.keep_attachments).map_err(|e| {
                    AgentError::Other(format!("Failed to create temp attachment directory: {}", e))
                })?;

            // Process attachments
            let attachments = payload.attachments();
            let attachment_paths = process_attachments(&attachments, temp_dir.path()).await?;

            debug!(
                target: "llm_toolkit::agent::cli_agent",
                "Processed {} attachments to temp files", attachment_paths.len()
            );

            // Format prompt with attachment paths
            let prompt = format_prompt_with_attachments(&text_intent, &attachment_paths);

            Ok((prompt, Some(temp_dir)))
        } else {
            Ok((text_intent.clone(), None))
        }
    }
}

/// Trait for CLI-based agents.
///
/// This trait provides common functionality for agents that execute
/// external command-line tools. It standardizes configuration,
/// attachment handling, and command execution patterns.
pub trait CliAgent {
    /// Returns the agent's CLI configuration.
    fn config(&self) -> &CliAgentConfig;

    /// Returns a mutable reference to the agent's CLI configuration.
    fn config_mut(&mut self) -> &mut CliAgentConfig;

    /// Returns the path to the CLI executable (if specified).
    fn cli_path(&self) -> Option<&Path>;

    /// Returns the name of the CLI command (e.g., "gemini", "claude").
    fn cli_command_name(&self) -> &str;

    /// Builds a Command for execution with the given prompt.
    ///
    /// This method should:
    /// 1. Create a Command with the appropriate executable
    /// 2. Apply configuration (working dir, env vars) via config.apply_to_command()
    /// 3. Add agent-specific arguments (model, etc.)
    /// 4. Add the prompt
    ///
    /// # Arguments
    /// * `prompt` - The final prompt text (may include attachment paths)
    fn build_command(&self, prompt: &str) -> Result<Command, AgentError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_agent_config_default() {
        let config = CliAgentConfig::default();
        assert!(matches!(
            config.execution_profile,
            ExecutionProfile::Balanced
        ));
        assert!(config.working_dir.is_none());
        assert!(config.env_vars.is_empty());
        assert!(config.extra_args.is_empty());
        assert!(config.attachment_dir.is_none());
        assert!(!config.keep_attachments);
    }

    #[test]
    fn test_cli_agent_config_builder() {
        let config = CliAgentConfig::new()
            .with_execution_profile(ExecutionProfile::Creative)
            .with_cwd("/project")
            .with_env("PATH", "/custom/path")
            .with_arg("--experimental")
            .with_attachment_dir("/tmp/attachments")
            .with_keep_attachments(true);

        assert!(matches!(
            config.execution_profile,
            ExecutionProfile::Creative
        ));
        assert_eq!(config.working_dir, Some(PathBuf::from("/project")));
        assert_eq!(
            config.env_vars.get("PATH"),
            Some(&"/custom/path".to_string())
        );
        assert_eq!(config.extra_args, vec!["--experimental"]);
        assert_eq!(
            config.attachment_dir,
            Some(PathBuf::from("/tmp/attachments"))
        );
        assert!(config.keep_attachments);
    }

    #[test]
    fn test_cli_agent_config_with_envs() {
        let mut env_map = HashMap::new();
        env_map.insert("KEY1".to_string(), "value1".to_string());
        env_map.insert("KEY2".to_string(), "value2".to_string());

        let config = CliAgentConfig::new().with_envs(env_map);

        assert_eq!(config.env_vars.len(), 2);
        assert_eq!(config.env_vars.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(config.env_vars.get("KEY2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_cli_agent_config_clear_env() {
        let config = CliAgentConfig::new()
            .with_env("KEY1", "value1")
            .with_env("KEY2", "value2")
            .clear_env();

        assert!(config.env_vars.is_empty());
    }

    #[test]
    fn test_cli_agent_config_with_args() {
        let config =
            CliAgentConfig::new().with_args(vec!["--flag1".to_string(), "--flag2".to_string()]);

        assert_eq!(config.extra_args.len(), 2);
        assert_eq!(config.extra_args[0], "--flag1");
        assert_eq!(config.extra_args[1], "--flag2");
    }
}
