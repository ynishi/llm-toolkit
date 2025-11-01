//! # Observability
//!
//! Provides a simple, configurable interface for initializing and managing observability
//! (tracing and logging) for the `llm-toolkit`.
//!
//! This module is designed to make it easy for developers to get detailed insights into
//! the execution of their LLM workflows with minimal setup.

use tracing::Level;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Configuration for initializing the observability system.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    /// The maximum log level to capture.
    pub level: Level,
    /// The target for the logs.
    pub target: LogTarget,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            target: LogTarget::default(),
        }
    }
}

/// Defines the output target for logs.
#[derive(Debug, Clone, Default)]
pub enum LogTarget {
    /// Log to the console (stdout).
    #[default]
    Console,
    /// Log to a file.
    File(String),
}

/// Initializes the global tracing subscriber.
///
/// This function should be called once at the beginning of your application's main function.
///
/// # Panics
///
/// This function will panic if it is called more than once, or if another tracing
/// subscriber has already been set.
pub fn init(config: ObservabilityConfig) -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::from_default_env()
        .add_directive(format!("llm_toolkit={}", config.level).parse()?)
        .add_directive(format!("llm_toolkit_macros={}", config.level).parse()?);

    let subscriber = tracing_subscriber::registry().with(filter);

    match config.target {
        LogTarget::Console => {
            let layer = fmt::layer().with_writer(std::io::stdout);
            subscriber.with(layer).init();
        }
        LogTarget::File(path) => {
            let file = std::fs::File::create(path)?;
            let layer = fmt::layer().with_writer(file);
            subscriber.with(layer).init();
        }
    };

    Ok(())
}
