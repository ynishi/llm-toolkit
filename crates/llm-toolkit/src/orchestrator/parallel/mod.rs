//! Parallel orchestrator implementation for concurrent workflow execution.
//!
//! This module provides a parallel execution engine that analyzes workflow dependencies
//! and executes independent steps concurrently in "waves" for optimal performance.

pub mod config;
pub mod dependency_analysis;
pub mod dependency_graph;
pub mod execution_state;

pub use config::ParallelOrchestratorConfig;
pub use dependency_analysis::{build_dependency_graph, extract_template_variables};
pub use dependency_graph::DependencyGraph;
pub use execution_state::{ExecutionStateManager, StepFailure, StepState};
