//! ClaudeCodeAgent - A universal agent implementation that wraps the Claude CLI.
//!
//! This agent can handle a wide variety of tasks by spawning the `claude` command
//! with the `-p` flag to pass prompts directly.

use crate::agent::{Agent, AgentError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::process::Command;

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
}

impl ClaudeCodeAgent {
    /// Creates a new ClaudeCodeAgent.
    ///
    /// By default, this will search for `claude` in the system PATH.
    pub fn new() -> Self {
        Self { claude_path: None }
    }

    /// Creates a new ClaudeCodeAgent with a custom path to the claude executable.
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            claude_path: Some(path),
        }
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
            .map_err(|e| {
                AgentError::ProcessError(format!("Failed to check claude availability: {}", e))
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

#[async_trait]
impl Agent for ClaudeCodeAgent {
    type Output = String;

    fn expertise(&self) -> &str {
        "A general-purpose AI agent capable of coding, research, analysis, \
         writing, and problem-solving across various domains. Can handle \
         complex multi-step tasks autonomously."
    }

    async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
        let claude_cmd = self
            .claude_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "claude".to_string());

        log::info!("🤖 ClaudeCodeAgent executing...");
        log::debug!("Intent length: {} chars", intent.len());
        log::trace!("Full intent: {}", intent);

        let output = Command::new(&claude_cmd)
            .arg("-p")
            .arg(&intent)
            .output()
            .await
            .map_err(|e| {
                log::error!("Failed to spawn claude process: {}", e);
                AgentError::ProcessError(format!(
                    "Failed to spawn claude process: {}. \
                     Make sure 'claude' is installed and in PATH.",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::error!("Claude command failed: {}", stderr);
            return Err(AgentError::ExecutionFailed(format!(
                "Claude command failed with status {}: {}",
                output.status, stderr
            )));
        }

        let stdout = String::from_utf8(output.stdout).map_err(|e| {
            log::error!("Failed to parse output: {}", e);
            AgentError::Other(format!("Failed to parse claude output as UTF-8: {}", e))
        })?;

        log::info!("✅ ClaudeCodeAgent completed");
        log::debug!("Output length: {} chars", stdout.len());
        log::trace!("Full output: {}", stdout);

        Ok(stdout)
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

    async fn execute(&self, intent: String) -> Result<Self::Output, AgentError> {
        log::info!(
            "📊 ClaudeCodeJsonAgent<{}> executing...",
            std::any::type_name::<T>()
        );

        let raw_output = self.inner.execute(intent).await?;

        log::debug!("Extracting JSON from raw output...");

        // Try to extract JSON from the output (might be wrapped in markdown, etc.)
        let json_str = crate::extract_json(&raw_output).map_err(|e| {
            log::error!("Failed to extract JSON: {}", e);
            AgentError::ParseError(format!(
                "Failed to extract JSON from claude output: {}. Raw output: {}",
                e, raw_output
            ))
        })?;

        log::debug!("Parsing JSON into {}...", std::any::type_name::<T>());

        let result = serde_json::from_str(&json_str).map_err(|e| {
            log::error!("Failed to parse JSON: {}", e);
            AgentError::ParseError(format!(
                "Failed to parse JSON: {}. Extracted JSON: {}",
                e, json_str
            ))
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
}
