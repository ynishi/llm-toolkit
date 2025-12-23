//! Model definitions for LLM providers.
//!
//! This module provides type-safe model identifiers for Anthropic Claude,
//! Google Gemini, and OpenAI models. Using enums prevents typos and ensures
//! only valid model names are used.
//!
//! # Design Philosophy
//!
//! - **Type Safety**: Enums prevent invalid model names at compile time
//! - **Flexibility**: `Custom` variant allows new models without code changes
//! - **Validation**: Custom models are validated by prefix on conversion
//! - **Dual Names**: Both API IDs and CLI shorthand names are supported
//!
//! # Future Direction
//!
//! This module will evolve to support capability-based model selection:
//! ```ignore
//! Model::query()
//!     .provider(Provider::Any)
//!     .tier(Tier::Fast)
//!     .with_capability(Cap::Vision)
//!     .max_budget_per_1k(0.01)
//!     .select()
//! ```

use std::fmt;

/// Error type for model-related operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Model name doesn't match expected prefix for the provider
    InvalidPrefix {
        model: String,
        expected_prefixes: &'static [&'static str],
    },
    /// Unknown model shorthand
    UnknownShorthand(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrefix {
                model,
                expected_prefixes,
            } => {
                write!(
                    f,
                    "Invalid model name '{}'. Expected prefix: {}",
                    model,
                    expected_prefixes.join(" or ")
                )
            }
            Self::UnknownShorthand(s) => write!(f, "Unknown model shorthand: {}", s),
        }
    }
}

impl std::error::Error for ModelError {}

// ============================================================================
// Anthropic Claude Models
// ============================================================================

/// Anthropic Claude model identifiers.
///
/// # Examples
///
/// ```
/// use llm_toolkit::models::ClaudeModel;
///
/// // Use predefined models
/// let model = ClaudeModel::Opus45;
/// assert_eq!(model.as_api_id(), "claude-opus-4-5-20251124");
///
/// // Parse from string (shorthand or full name)
/// let model: ClaudeModel = "opus".parse().unwrap();
/// assert_eq!(model, ClaudeModel::Opus45);
///
/// // Custom model (validated)
/// let model: ClaudeModel = "claude-future-model-2026".parse().unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClaudeModel {
    /// Claude Opus 4.5 - Most capable (November 2025)
    Opus45,
    /// Claude Sonnet 4.5 - Balanced performance (September 2025)
    Sonnet45,
    /// Claude Opus 4.1 - Enhanced agentic (August 2025)
    Opus41,
    /// Claude Opus 4 - Previous flagship (May 2025)
    Opus4,
    /// Claude Sonnet 4 - Previous balanced (May 2025)
    Sonnet4,
    /// Claude Haiku 3.5 - Fast and efficient
    Haiku35,
    /// Custom model (validated: must start with "claude-")
    Custom(String),
}

impl Default for ClaudeModel {
    fn default() -> Self {
        Self::Sonnet45
    }
}

impl ClaudeModel {
    /// Returns the full API model identifier.
    ///
    /// Use this when making API calls to Anthropic.
    pub fn as_api_id(&self) -> &str {
        match self {
            Self::Opus45 => "claude-opus-4-5-20251124",
            Self::Sonnet45 => "claude-sonnet-4-5-20250929",
            Self::Opus41 => "claude-opus-4-1-20250805",
            Self::Opus4 => "claude-opus-4-20250514",
            Self::Sonnet4 => "claude-sonnet-4-20250514",
            Self::Haiku35 => "claude-3-5-haiku-20241022",
            Self::Custom(s) => s,
        }
    }

    /// Returns the CLI shorthand name.
    ///
    /// Use this when invoking CLI tools like `claude`.
    pub fn as_cli_name(&self) -> &str {
        match self {
            Self::Opus45 => "claude-opus-4.5",
            Self::Sonnet45 => "claude-sonnet-4.5",
            Self::Opus41 => "claude-opus-4.1",
            Self::Opus4 => "claude-opus-4",
            Self::Sonnet4 => "claude-sonnet-4",
            Self::Haiku35 => "claude-haiku-3.5",
            Self::Custom(s) => s,
        }
    }

    /// Validates that a string is a valid Claude model identifier.
    fn validate_custom(s: &str) -> Result<(), ModelError> {
        if s.starts_with("claude-") {
            Ok(())
        } else {
            Err(ModelError::InvalidPrefix {
                model: s.to_string(),
                expected_prefixes: &["claude-"],
            })
        }
    }
}

impl std::str::FromStr for ClaudeModel {
    type Err = ModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // Opus 4.5 variants
            "opus" | "opus-4.5" | "opus45" | "claude-opus-4.5" | "claude-opus-4-5-20251124" => {
                Ok(Self::Opus45)
            }
            // Sonnet 4.5 variants
            "sonnet"
            | "sonnet-4.5"
            | "sonnet45"
            | "claude-sonnet-4.5"
            | "claude-sonnet-4-5-20250929" => Ok(Self::Sonnet45),
            // Opus 4.1 variants
            "opus-4.1" | "opus41" | "claude-opus-4.1" | "claude-opus-4-1-20250805" => {
                Ok(Self::Opus41)
            }
            // Opus 4 variants
            "opus-4" | "opus4" | "claude-opus-4" | "claude-opus-4-20250514" => Ok(Self::Opus4),
            // Sonnet 4 variants
            "sonnet-4" | "sonnet4" | "claude-sonnet-4" | "claude-sonnet-4-20250514" => {
                Ok(Self::Sonnet4)
            }
            // Haiku 3.5 variants
            "haiku"
            | "haiku-3.5"
            | "haiku35"
            | "claude-haiku-3.5"
            | "claude-3-5-haiku-20241022" => Ok(Self::Haiku35),
            // Custom (validated)
            _ => {
                Self::validate_custom(s)?;
                Ok(Self::Custom(s.to_string()))
            }
        }
    }
}

impl fmt::Display for ClaudeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_api_id())
    }
}

// ============================================================================
// Google Gemini Models
// ============================================================================

/// Google Gemini model identifiers.
///
/// # Examples
///
/// ```
/// use llm_toolkit::models::GeminiModel;
///
/// let model = GeminiModel::Flash3;
/// assert_eq!(model.as_api_id(), "gemini-3-flash");
///
/// let model: GeminiModel = "flash".parse().unwrap();
/// assert_eq!(model, GeminiModel::Flash25);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeminiModel {
    /// Gemini 3 Flash - Latest fast model (December 2025)
    Flash3,
    /// Gemini 3 Pro - Latest capable model (December 2025)
    Pro3,
    /// Gemini 2.5 Flash - Stable fast model
    Flash25,
    /// Gemini 2.5 Pro - Stable capable model
    Pro25,
    /// Gemini 2.0 Flash - Previous generation
    Flash20,
    /// Custom model (validated: must start with "gemini-")
    Custom(String),
}

impl Default for GeminiModel {
    fn default() -> Self {
        Self::Flash25
    }
}

impl GeminiModel {
    /// Returns the full API model identifier.
    pub fn as_api_id(&self) -> &str {
        match self {
            Self::Flash3 => "gemini-3-flash",
            Self::Pro3 => "gemini-3-pro",
            Self::Flash25 => "gemini-2.5-flash",
            Self::Pro25 => "gemini-2.5-pro",
            Self::Flash20 => "gemini-2.0-flash",
            Self::Custom(s) => s,
        }
    }

    /// Returns the CLI shorthand name.
    pub fn as_cli_name(&self) -> &str {
        match self {
            Self::Flash3 => "flash-3",
            Self::Pro3 => "pro-3",
            Self::Flash25 => "flash",
            Self::Pro25 => "pro",
            Self::Flash20 => "flash-2.0",
            Self::Custom(s) => s,
        }
    }

    fn validate_custom(s: &str) -> Result<(), ModelError> {
        if s.starts_with("gemini-") {
            Ok(())
        } else {
            Err(ModelError::InvalidPrefix {
                model: s.to_string(),
                expected_prefixes: &["gemini-"],
            })
        }
    }
}

impl std::str::FromStr for GeminiModel {
    type Err = ModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // Flash 3 variants
            "flash-3" | "flash3" | "gemini-3-flash" => Ok(Self::Flash3),
            // Pro 3 variants
            "pro-3" | "pro3" | "gemini-3-pro" => Ok(Self::Pro3),
            // Flash 2.5 variants (default "flash")
            "flash" | "flash-2.5" | "flash25" | "gemini-2.5-flash" => Ok(Self::Flash25),
            // Pro 2.5 variants (default "pro")
            "pro" | "pro-2.5" | "pro25" | "gemini-2.5-pro" => Ok(Self::Pro25),
            // Flash 2.0 variants
            "flash-2.0" | "flash20" | "flash-2" | "gemini-2.0-flash" => Ok(Self::Flash20),
            // Custom (validated)
            _ => {
                Self::validate_custom(s)?;
                Ok(Self::Custom(s.to_string()))
            }
        }
    }
}

impl fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_api_id())
    }
}

// ============================================================================
// OpenAI Models
// ============================================================================

/// OpenAI model identifiers.
///
/// # Examples
///
/// ```
/// use llm_toolkit::models::OpenAIModel;
///
/// let model = OpenAIModel::Gpt52;
/// assert_eq!(model.as_api_id(), "gpt-5.2");
///
/// let model: OpenAIModel = "4o".parse().unwrap();
/// assert_eq!(model, OpenAIModel::Gpt4o);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenAIModel {
    // GPT-5 Series
    /// GPT-5.2 - Latest flagship (December 2025)
    Gpt52,
    /// GPT-5.1 - Previous flagship (November 2025)
    Gpt51,
    /// GPT-5 - Original GPT-5 (August 2025)
    Gpt5,
    /// GPT-5 Mini - Cost-effective
    Gpt5Mini,
    /// GPT-5 Nano - Smallest
    Gpt5Nano,

    // GPT-5 Codex Series (Agentic coding)
    /// GPT-5.1 Codex - Optimized for agentic coding
    Gpt51Codex,
    /// GPT-5.1 Codex Mini - Cost-effective coding
    Gpt51CodexMini,
    /// GPT-5 Codex - Legacy agentic coding
    Gpt5Codex,
    /// GPT-5 Codex Mini - Legacy cost-effective coding
    Gpt5CodexMini,

    // GPT-4 Series (Legacy but still useful)
    /// GPT-4.1 - Improved instruction following
    Gpt41,
    /// GPT-4.1 Mini - Cost-effective
    Gpt41Mini,
    /// GPT-4o - Legacy, still useful for audio
    Gpt4o,
    /// GPT-4o Mini - Cost-effective legacy
    Gpt4oMini,

    // O-Series (Reasoning models)
    /// o3-pro - Extended reasoning
    O3Pro,
    /// o3 - Standard reasoning
    O3,
    /// o3-mini - Fast reasoning
    O3Mini,
    /// o1 - Previous reasoning model
    O1,
    /// o1-pro - Extended previous reasoning
    O1Pro,

    /// Custom model (validated: must start with "gpt-", "o1-", or "o3-")
    Custom(String),
}

impl Default for OpenAIModel {
    fn default() -> Self {
        Self::Gpt4o // Conservative default for compatibility
    }
}

impl OpenAIModel {
    /// Returns the full API model identifier.
    pub fn as_api_id(&self) -> &str {
        match self {
            // GPT-5 Series
            Self::Gpt52 => "gpt-5.2",
            Self::Gpt51 => "gpt-5.1",
            Self::Gpt5 => "gpt-5",
            Self::Gpt5Mini => "gpt-5-mini",
            Self::Gpt5Nano => "gpt-5-nano",
            // GPT-5 Codex
            Self::Gpt51Codex => "gpt-5.1-codex",
            Self::Gpt51CodexMini => "gpt-5.1-codex-mini",
            Self::Gpt5Codex => "gpt-5-codex",
            Self::Gpt5CodexMini => "gpt-5-codex-mini",
            // GPT-4 Series
            Self::Gpt41 => "gpt-4.1",
            Self::Gpt41Mini => "gpt-4.1-mini",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            // O-Series
            Self::O3Pro => "o3-pro",
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::O1 => "o1",
            Self::O1Pro => "o1-pro",
            // Custom
            Self::Custom(s) => s,
        }
    }

    /// Returns the CLI shorthand name.
    pub fn as_cli_name(&self) -> &str {
        self.as_api_id() // OpenAI uses same names for CLI
    }

    fn validate_custom(s: &str) -> Result<(), ModelError> {
        const VALID_PREFIXES: &[&str] = &["gpt-", "o1-", "o3-", "o4-"];
        if VALID_PREFIXES.iter().any(|p| s.starts_with(p)) {
            Ok(())
        } else {
            Err(ModelError::InvalidPrefix {
                model: s.to_string(),
                expected_prefixes: VALID_PREFIXES,
            })
        }
    }
}

impl std::str::FromStr for OpenAIModel {
    type Err = ModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // GPT-5.2
            "5.2" | "gpt-5.2" | "gpt52" => Ok(Self::Gpt52),
            // GPT-5.1
            "5.1" | "gpt-5.1" | "gpt51" => Ok(Self::Gpt51),
            // GPT-5
            "5" | "gpt-5" | "gpt5" => Ok(Self::Gpt5),
            // GPT-5 Mini/Nano
            "5-mini" | "gpt-5-mini" => Ok(Self::Gpt5Mini),
            "5-nano" | "gpt-5-nano" => Ok(Self::Gpt5Nano),
            // GPT-5.1 Codex
            "5.1-codex" | "gpt-5.1-codex" | "codex" => Ok(Self::Gpt51Codex),
            "5.1-codex-mini" | "gpt-5.1-codex-mini" | "codex-mini" => Ok(Self::Gpt51CodexMini),
            // GPT-5 Codex (Legacy)
            "5-codex" | "gpt-5-codex" => Ok(Self::Gpt5Codex),
            "5-codex-mini" | "gpt-5-codex-mini" => Ok(Self::Gpt5CodexMini),
            // GPT-4.1
            "4.1" | "gpt-4.1" | "gpt41" => Ok(Self::Gpt41),
            "4.1-mini" | "gpt-4.1-mini" => Ok(Self::Gpt41Mini),
            // GPT-4o
            "4o" | "gpt-4o" => Ok(Self::Gpt4o),
            "4o-mini" | "gpt-4o-mini" => Ok(Self::Gpt4oMini),
            // O-Series
            "o3-pro" => Ok(Self::O3Pro),
            "o3" => Ok(Self::O3),
            "o3-mini" => Ok(Self::O3Mini),
            "o1" => Ok(Self::O1),
            "o1-pro" => Ok(Self::O1Pro),
            // Custom (validated)
            _ => {
                Self::validate_custom(s)?;
                Ok(Self::Custom(s.to_string()))
            }
        }
    }
}

impl fmt::Display for OpenAIModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_api_id())
    }
}

// ============================================================================
// Provider-agnostic Model enum
// ============================================================================

/// Provider-agnostic model identifier.
///
/// Use this when you need to work with models from any provider.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Model {
    Claude(ClaudeModel),
    Gemini(GeminiModel),
    OpenAI(OpenAIModel),
}

impl Model {
    /// Returns the API model identifier.
    pub fn as_api_id(&self) -> &str {
        match self {
            Self::Claude(m) => m.as_api_id(),
            Self::Gemini(m) => m.as_api_id(),
            Self::OpenAI(m) => m.as_api_id(),
        }
    }

    /// Returns the CLI name.
    pub fn as_cli_name(&self) -> &str {
        match self {
            Self::Claude(m) => m.as_cli_name(),
            Self::Gemini(m) => m.as_cli_name(),
            Self::OpenAI(m) => m.as_cli_name(),
        }
    }
}

impl From<ClaudeModel> for Model {
    fn from(m: ClaudeModel) -> Self {
        Self::Claude(m)
    }
}

impl From<GeminiModel> for Model {
    fn from(m: GeminiModel) -> Self {
        Self::Gemini(m)
    }
}

impl From<OpenAIModel> for Model {
    fn from(m: OpenAIModel) -> Self {
        Self::OpenAI(m)
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_api_id())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    mod claude_model {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(ClaudeModel::default(), ClaudeModel::Sonnet45);
        }

        #[test]
        fn test_api_id() {
            assert_eq!(ClaudeModel::Opus45.as_api_id(), "claude-opus-4-5-20251124");
            assert_eq!(
                ClaudeModel::Sonnet45.as_api_id(),
                "claude-sonnet-4-5-20250929"
            );
            assert_eq!(
                ClaudeModel::Haiku35.as_api_id(),
                "claude-3-5-haiku-20241022"
            );
        }

        #[test]
        fn test_cli_name() {
            assert_eq!(ClaudeModel::Opus45.as_cli_name(), "claude-opus-4.5");
            assert_eq!(ClaudeModel::Sonnet4.as_cli_name(), "claude-sonnet-4");
        }

        #[test]
        fn test_parse_shorthand() {
            assert_eq!("opus".parse::<ClaudeModel>().unwrap(), ClaudeModel::Opus45);
            assert_eq!(
                "sonnet".parse::<ClaudeModel>().unwrap(),
                ClaudeModel::Sonnet45
            );
            assert_eq!(
                "haiku".parse::<ClaudeModel>().unwrap(),
                ClaudeModel::Haiku35
            );
        }

        #[test]
        fn test_parse_full_name() {
            assert_eq!(
                "claude-opus-4-5-20251124".parse::<ClaudeModel>().unwrap(),
                ClaudeModel::Opus45
            );
            assert_eq!(
                "claude-sonnet-4".parse::<ClaudeModel>().unwrap(),
                ClaudeModel::Sonnet4
            );
        }

        #[test]
        fn test_parse_custom_valid() {
            let model: ClaudeModel = "claude-future-model-2026".parse().unwrap();
            assert_eq!(
                model,
                ClaudeModel::Custom("claude-future-model-2026".to_string())
            );
        }

        #[test]
        fn test_parse_custom_invalid() {
            let result: Result<ClaudeModel, _> = "gpt-4o".parse();
            assert!(result.is_err());
        }
    }

    mod gemini_model {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(GeminiModel::default(), GeminiModel::Flash25);
        }

        #[test]
        fn test_parse() {
            assert_eq!(
                "flash".parse::<GeminiModel>().unwrap(),
                GeminiModel::Flash25
            );
            assert_eq!("pro".parse::<GeminiModel>().unwrap(), GeminiModel::Pro25);
            assert_eq!(
                "flash-3".parse::<GeminiModel>().unwrap(),
                GeminiModel::Flash3
            );
        }

        #[test]
        fn test_custom_invalid() {
            let result: Result<GeminiModel, _> = "claude-opus".parse();
            assert!(result.is_err());
        }
    }

    mod openai_model {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(OpenAIModel::default(), OpenAIModel::Gpt4o);
        }

        #[test]
        fn test_parse() {
            assert_eq!("4o".parse::<OpenAIModel>().unwrap(), OpenAIModel::Gpt4o);
            assert_eq!(
                "gpt-5.2".parse::<OpenAIModel>().unwrap(),
                OpenAIModel::Gpt52
            );
            assert_eq!("o3".parse::<OpenAIModel>().unwrap(), OpenAIModel::O3);
            assert_eq!(
                "codex".parse::<OpenAIModel>().unwrap(),
                OpenAIModel::Gpt51Codex
            );
        }

        #[test]
        fn test_o_series_validation() {
            // o-series should be valid custom
            let model: OpenAIModel = "o4-mini".parse().unwrap();
            assert_eq!(model, OpenAIModel::Custom("o4-mini".to_string()));
        }

        #[test]
        fn test_custom_invalid() {
            let result: Result<OpenAIModel, _> = "gemini-pro".parse();
            assert!(result.is_err());
        }
    }
}
