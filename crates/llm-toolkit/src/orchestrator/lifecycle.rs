#[cfg(feature = "agent")]
use async_trait::async_trait;

#[cfg(feature = "agent")]
use super::{error::OrchestratorError, strategy::StrategyMap};

/// Shared lifecycle operations for orchestrators that work with `StrategyMap`.
///
/// This trait makes it easy to manage execution plans consistently across the
/// sequential `Orchestrator` and the parallel `ParallelOrchestrator`.
#[cfg(feature = "agent")]
#[async_trait]
pub trait StrategyLifecycle {
    /// Injects a pre-built strategy map, bypassing automatic generation.
    fn set_strategy_map(&mut self, strategy: StrategyMap);

    /// Returns the currently active strategy map, if any.
    fn strategy_map(&self) -> Option<&StrategyMap>;

    /// Generates a strategy map for the given task without executing it.
    async fn generate_strategy_only(
        &mut self,
        task: &str,
    ) -> Result<StrategyMap, OrchestratorError>;
}
