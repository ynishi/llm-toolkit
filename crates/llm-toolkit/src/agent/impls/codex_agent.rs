//! CodexAgent - A universal agent implementation that wraps the Codex CLI.
//!
//! This agent can handle a wide variety of tasks by spawning the `codex` command
//! with prompts and configuration options.

use crate::agent::{Agent, AgentError, Payload};
use crate::models::OpenAIModel;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::{debug, error, info, instrument};

use super::cli_agent::{CliAgent, CliAgentConfig};

/// Type alias for backward compatibility.
/// Use [`OpenAIModel`] directly for new code.
#[deprecated(since = "0.59.0", note = "Use OpenAIModel directly")]
pub type CodexModel = OpenAIModel;

/// A general-purpose agent that executes tasks using the Codex CLI.
///
/// This agent wraps the `codex` command-line tool and can handle
/// coding, research, analysis, and other general tasks.
///
/// # Output
///
/// Returns the raw string output from Codex. For structured output,
/// you can parse this string using `serde_json` or other parsers.
///
/// # Example
///
/// ```rust,ignore
/// use llm_toolkit::agent::impls::CodexAgent;
/// use llm_toolkit::agent::Agent;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let agent = CodexAgent::new();
///
///     let result = agent.execute(
///         "Analyze the Rust ownership model and explain it in 3 bullet points".to_string().into()
///     ).await?;
///
///     println!("{}", result);
///     Ok(())
/// }
/// ```
pub struct CodexAgent {
    /// Path to the `codex` executable. If None, searches in PATH.
    codex_path: Option<PathBuf>,
    /// Model to use for generation
    model: Option<OpenAIModel>,
    /// Common CLI agent configuration
    config: CliAgentConfig,
    /// Sandbox mode (read-only, workspace-write, danger-full-access)
    sandbox: Option<String>,
    /// Approval policy (untrusted, on-failure, on-request, never)
    approval_policy: Option<String>,
    /// Enable web search
    enable_search: bool,
}

impl CodexAgent {
    /// Creates a new CodexAgent with default settings.
    ///
    /// By default:
    /// - Searches for `codex` in the system PATH
    /// - Uses default model
    /// - No working directory specified (uses current directory)
    /// - No additional environment variables
    /// - No extra CLI arguments
    pub fn new() -> Self {
        Self {
            codex_path: None,
            model: None,
            config: CliAgentConfig::new(),
            sandbox: None,
            approval_policy: None,
            enable_search: false,
        }
    }

    /// Creates a new CodexAgent with a custom path to the codex executable.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            codex_path: Some(path),
            model: None,
            config: CliAgentConfig::new(),
            sandbox: None,
            approval_policy: None,
            enable_search: false,
        }
    }

    /// Sets the model to use.
    pub fn with_model(mut self, model: OpenAIModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the model using a string identifier.
    ///
    /// Accepts: "gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1", "gpt-5-codex",
    /// "gpt-5-codex-mini", "gpt-5", or any custom model name.
    /// See [`OpenAIModel`] for all supported variants.
    pub fn with_model_str(mut self, model: &str) -> Self {
        self.model = Some(model.parse().unwrap_or(OpenAIModel::Gpt51Codex));
        self
    }

    /// Sets the execution profile.
    ///
    /// # Example
    /// ```rust,ignore
    /// use llm_toolkit::agent::ExecutionProfile;
    ///
    /// let agent = CodexAgent::new()
    ///     .with_execution_profile(ExecutionProfile::Creative);
    /// ```
    pub fn with_execution_profile(mut self, profile: crate::agent::ExecutionProfile) -> Self {
        self.config = self.config.with_execution_profile(profile);
        self
    }

    /// Sets the working directory where the codex command will be executed.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = CodexAgent::new()
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
    /// let agent = CodexAgent::new()
    ///     .with_directory("/path/to/project");
    /// ```
    pub fn with_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.config = self.config.with_directory(path);
        self
    }

    /// Sets a single environment variable for the codex command.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = CodexAgent::new()
    ///     .with_env("CODEX_API_KEY", "my-key");
    /// ```
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config = self.config.with_env(key, value);
        self
    }

    /// Sets multiple environment variables at once.
    pub fn with_envs(mut self, envs: std::collections::HashMap<String, String>) -> Self {
        self.config = self.config.with_envs(envs);
        self
    }

    /// Clears all environment variables.
    pub fn clear_env(mut self) -> Self {
        self.config = self.config.clear_env();
        self
    }

    /// Adds a single CLI argument to pass to the codex command.
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.config = self.config.with_arg(arg);
        self
    }

    /// Adds multiple CLI arguments at once.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.config = self.config.with_args(args);
        self
    }

    /// Sets the directory where attachment files will be written.
    ///
    /// If not specified, falls back to `working_dir` or system temp directory.
    pub fn with_attachment_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config = self.config.with_attachment_dir(path);
        self
    }

    /// Sets whether to keep temporary attachment files after execution.
    ///
    /// By default, temp files are deleted after each execution.
    /// Set to `true` to keep files for debugging purposes.
    pub fn with_keep_attachments(mut self, keep: bool) -> Self {
        self.config = self.config.with_keep_attachments(keep);
        self
    }

    /// Sets the sandbox mode.
    ///
    /// Options: "read-only", "workspace-write", "danger-full-access"
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = CodexAgent::new()
    ///     .with_sandbox("workspace-write");
    /// ```
    pub fn with_sandbox(mut self, mode: impl Into<String>) -> Self {
        self.sandbox = Some(mode.into());
        self
    }

    /// Sets the approval policy.
    ///
    /// Options: "untrusted", "on-failure", "on-request", "never"
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = CodexAgent::new()
    ///     .with_approval_policy("on-failure");
    /// ```
    pub fn with_approval_policy(mut self, policy: impl Into<String>) -> Self {
        self.approval_policy = Some(policy.into());
        self
    }

    /// Enables web search capability.
    ///
    /// # Example
    /// ```rust,ignore
    /// let agent = CodexAgent::new()
    ///     .with_search(true);
    /// ```
    pub fn with_search(mut self, enable: bool) -> Self {
        self.enable_search = enable;
        self
    }

    /// Convenience method for full automatic execution with workspace-write sandbox.
    ///
    /// Equivalent to: `.with_approval_policy("on-failure").with_sandbox("workspace-write")`
    pub fn full_auto(mut self) -> Self {
        self.approval_policy = Some("on-failure".to_string());
        self.sandbox = Some("workspace-write".to_string());
        self
    }

    /// Checks if the `codex` CLI is available in the system (static version).
    ///
    /// Returns `true` if the command exists in PATH, `false` otherwise.
    pub fn is_available() -> bool {
        #[cfg(unix)]
        let check_cmd = "which";
        #[cfg(windows)]
        let check_cmd = "where";

        std::process::Command::new(check_cmd)
            .arg("codex")
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
            .arg("codex")
            .output()
            .await
            .map_err(|e| AgentError::ProcessError {
                status_code: None,
                message: format!("Failed to check codex availability: {}", e),
                is_retryable: true,
                retry_after: None,
            })?;

        if output.status.success() {
            Ok(())
        } else {
            Err(AgentError::ExecutionFailed(
                "codex CLI not found in PATH. Please install Codex CLI.".to_string(),
            ))
        }
    }
}

impl Default for CodexAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl CliAgent for CodexAgent {
    fn config(&self) -> &CliAgentConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut CliAgentConfig {
        &mut self.config
    }

    fn cli_path(&self) -> Option<&Path> {
        self.codex_path.as_deref()
    }

    fn cli_command_name(&self) -> &str {
        "codex"
    }

    fn build_command(&self, prompt: &str) -> Result<Command, AgentError> {
        let cmd_name = self
            .codex_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "codex".to_string());

        let mut cmd = Command::new(cmd_name);

        // Apply common configuration (working dir, env vars)
        self.config.apply_to_command(&mut cmd);

        // Add working directory via -C flag if specified
        if let Some(dir) = &self.config.working_dir {
            cmd.arg("-C").arg(dir);
        }

        // Add model if specified
        if let Some(model) = &self.model {
            cmd.arg("-m").arg(model.as_api_id());
            debug!(
                target: "llm_toolkit::agent::codex",
                "Using model: {}", model.as_api_id()
            );
        }

        // Add sandbox mode if specified
        if let Some(sandbox) = &self.sandbox {
            cmd.arg("--sandbox").arg(sandbox);
            debug!(
                target: "llm_toolkit::agent::codex",
                "Using sandbox: {}", sandbox
            );
        }

        // Add approval policy if specified
        if let Some(policy) = &self.approval_policy {
            cmd.arg("-a").arg(policy);
            debug!(
                target: "llm_toolkit::agent::codex",
                "Using approval policy: {}", policy
            );
        }

        // Enable search if specified
        if self.enable_search {
            cmd.arg("--search");
            debug!(
                target: "llm_toolkit::agent::codex",
                "Web search enabled"
            );
        }

        // Add extra CLI arguments from config
        for arg in &self.config.extra_args {
            debug!(
                target: "llm_toolkit::agent::codex",
                "Adding extra argument: {}", arg
            );
            cmd.arg(arg);
        }

        // Add the prompt as positional argument (last)
        cmd.arg(prompt);

        Ok(cmd)
    }
}

#[async_trait]
impl Agent for CodexAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &Self::Expertise {
        &"General-purpose AI assistant powered by Codex CLI, capable of coding, analysis, \
         research, and autonomous task execution with sandbox support"
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
        // Note: Codex uses -i/--image for attachments, which will be added by build_command
        let (final_prompt, _temp_dir) = self.config.process_payload_attachments(&payload).await?;

        debug!(
            target: "llm_toolkit::agent::codex",
            "Building codex command with prompt length: {}", final_prompt.len()
        );

        let mut cmd = self.build_command(&final_prompt)?;

        // Add image attachments via -i flag if present
        if payload.has_attachments() {
            for attachment in payload.attachments() {
                if let Some(path) = attachment.file_name() {
                    cmd.arg("-i").arg(&path);
                    debug!(
                        target: "llm_toolkit::agent::codex",
                        "Adding image attachment: {}", path
                    );
                }
            }
        }

        debug!(
            target: "llm_toolkit::agent::codex",
            "Executing codex command: {:?}", cmd
        );

        let output = cmd.output().await.map_err(|e| {
            error!(
                target: "llm_toolkit::agent::codex",
                "Failed to execute codex command: {}", e
            );
            AgentError::ExecutionFailed(format!("Failed to execute codex command: {}", e))
        })?;

        if output.status.success() {
            let response = String::from_utf8(output.stdout).map_err(|e| {
                error!(
                    target: "llm_toolkit::agent::codex",
                    "Failed to parse stdout as UTF-8: {}", e
                );
                AgentError::ParseError {
                    message: format!("Failed to parse codex stdout: {}", e),
                    reason: crate::agent::error::ParseErrorReason::UnexpectedEof,
                }
            })?;

            info!(
                target: "llm_toolkit::agent::codex",
                "Codex command completed successfully, response length: {}", response.len()
            );
            Ok(response.trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!(
                target: "llm_toolkit::agent::codex",
                "Codex command failed with stderr: {}", stderr
            );
            Err(AgentError::ExecutionFailed(format!(
                "Codex command failed: {}",
                stderr
            )))
        }
    }

    fn name(&self) -> String {
        "CodexAgent".to_string()
    }

    async fn is_available(&self) -> Result<(), AgentError> {
        Self::check_available().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codex_agent_default() {
        let agent = CodexAgent::default();
        assert!(agent.codex_path.is_none());
        assert!(agent.model.is_none());
        assert!(agent.sandbox.is_none());
        assert!(agent.approval_policy.is_none());
        assert!(!agent.enable_search);
    }

    #[test]
    fn test_codex_agent_builder() {
        let agent = CodexAgent::new()
            .with_model(OpenAIModel::Gpt51Codex)
            .with_cwd("/project")
            .with_sandbox("workspace-write")
            .with_approval_policy("on-failure")
            .with_search(true);

        assert!(matches!(agent.model, Some(OpenAIModel::Gpt51Codex)));
        assert_eq!(agent.sandbox, Some("workspace-write".to_string()));
        assert_eq!(agent.approval_policy, Some("on-failure".to_string()));
        assert!(agent.enable_search);
    }

    #[test]
    fn test_codex_agent_full_auto() {
        let agent = CodexAgent::new().full_auto();

        assert_eq!(agent.approval_policy, Some("on-failure".to_string()));
        assert_eq!(agent.sandbox, Some("workspace-write".to_string()));
    }

    #[test]
    fn test_openai_model_custom() {
        let model = OpenAIModel::Custom("gpt-4".to_string());
        assert_eq!(model.as_api_id(), "gpt-4");
    }
}
