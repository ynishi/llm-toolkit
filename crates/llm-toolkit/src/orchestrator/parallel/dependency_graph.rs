//! Dependency graph representation for parallel execution.
//!
//! This module provides a directed acyclic graph (DAG) structure to represent
//! dependencies between workflow steps, enabling parallel execution of independent steps.

use std::collections::{HashMap, HashSet};

/// A directed acyclic graph representing step dependencies.
///
/// The graph maintains both forward edges (dependencies) and reverse edges (dependents)
/// for efficient traversal in both directions. This is crucial for:
/// - Finding ready-to-execute steps (zero dependencies)
/// - Propagating failures to dependent steps
///
/// # Examples
///
/// ```ignore
/// use llm_toolkit::orchestrator::parallel::DependencyGraph;
///
/// let mut graph = DependencyGraph::new();
/// graph.add_node("step_1");
/// graph.add_node("step_2");
/// graph.add_dependency("step_2", "step_1"); // step_2 depends on step_1
///
/// assert!(graph.get_dependencies("step_2").contains("step_1"));
/// assert!(graph.get_dependents("step_1").contains("step_2"));
/// ```
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Forward edges: step_id -> set of step_ids it depends on
    nodes: HashMap<String, HashSet<String>>,
    /// Reverse edges: step_id -> set of step_ids that depend on it
    reverse_edges: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    /// Creates a new empty dependency graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            reverse_edges: HashMap::new(),
        }
    }

    /// Adds a node to the graph without any dependencies.
    ///
    /// If the node already exists, this is a no-op.
    pub fn add_node(&mut self, step_id: &str) {
        self.nodes.entry(step_id.to_string()).or_default();
        self.reverse_edges.entry(step_id.to_string()).or_default();
    }

    /// Adds a dependency edge: `step_id` depends on `depends_on`.
    ///
    /// This automatically adds both nodes if they don't exist and maintains
    /// both forward and reverse edges.
    ///
    /// # Arguments
    ///
    /// * `step_id` - The step that has a dependency
    /// * `depends_on` - The step that must complete first
    pub fn add_dependency(&mut self, step_id: &str, depends_on: &str) {
        // Forward edge: step_id depends on depends_on
        self.nodes
            .entry(step_id.to_string())
            .or_default()
            .insert(depends_on.to_string());

        // Ensure depends_on exists as a node
        self.nodes.entry(depends_on.to_string()).or_default();

        // Reverse edge: depends_on is depended upon by step_id
        self.reverse_edges
            .entry(depends_on.to_string())
            .or_default()
            .insert(step_id.to_string());

        // Ensure step_id has a reverse_edges entry
        self.reverse_edges.entry(step_id.to_string()).or_default();
    }

    /// Returns the set of step IDs that the given step depends on.
    ///
    /// Returns an empty set if the step has no dependencies.
    pub fn get_dependencies(&self, step_id: &str) -> HashSet<String> {
        self.nodes
            .get(step_id)
            .cloned()
            .unwrap_or_else(HashSet::new)
    }

    /// Returns the set of step IDs that depend on the given step.
    ///
    /// Returns an empty set if no steps depend on this step.
    pub fn get_dependents(&self, step_id: &str) -> HashSet<String> {
        self.reverse_edges
            .get(step_id)
            .cloned()
            .unwrap_or_else(HashSet::new)
    }

    /// Returns the total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns all step IDs that have zero dependencies.
    ///
    /// These steps can be executed immediately in the first wave.
    pub fn get_zero_dependency_steps(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, deps)| deps.is_empty())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Detects if the graph contains any cycles using depth-first search.
    ///
    /// Returns `true` if a cycle is detected, `false` otherwise.
    /// A cycle would make parallel execution impossible as steps would
    /// wait for each other indefinitely.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in self.nodes.keys() {
            if self.has_cycle_dfs(node, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    /// DFS helper for cycle detection.
    ///
    /// Uses a recursion stack to detect back edges, which indicate cycles.
    fn has_cycle_dfs(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(node) {
            return true; // Found cycle (back edge)
        }

        if visited.contains(node) {
            return false; // Already processed this subtree
        }

        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        // Visit all dependencies
        if let Some(deps) = self.nodes.get(node) {
            for dep in deps {
                if self.has_cycle_dfs(dep, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph_is_empty() {
        let graph = DependencyGraph::new();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");

        assert_eq!(graph.node_count(), 1);
        assert!(graph.get_dependencies("step_1").is_empty());
    }

    #[test]
    fn test_add_dependency() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_dependency("step_2", "step_1");

        let deps = graph.get_dependencies("step_2");
        assert_eq!(deps.len(), 1);
        assert!(deps.contains("step_1"));
    }

    #[test]
    fn test_add_dependency_creates_nodes() {
        let mut graph = DependencyGraph::new();
        // Don't add nodes explicitly - add_dependency should create them
        graph.add_dependency("step_2", "step_1");

        assert_eq!(graph.node_count(), 2);
        assert!(graph.get_dependencies("step_2").contains("step_1"));
    }

    #[test]
    fn test_get_dependents() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_dependency("step_2", "step_1");

        let dependents = graph.get_dependents("step_1");
        assert_eq!(dependents.len(), 1);
        assert!(dependents.contains("step_2"));
    }

    #[test]
    fn test_multiple_dependents() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_node("step_3");
        graph.add_dependency("step_2", "step_1");
        graph.add_dependency("step_3", "step_1");

        let dependents = graph.get_dependents("step_1");
        assert_eq!(dependents.len(), 2);
        assert!(dependents.contains("step_2"));
        assert!(dependents.contains("step_3"));
    }

    #[test]
    fn test_cycle_detection_simple_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_dependency("step_1", "step_2");
        graph.add_dependency("step_2", "step_1");

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_no_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_node("step_3");
        graph.add_dependency("step_2", "step_1");
        graph.add_dependency("step_3", "step_2");

        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_self_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_dependency("step_1", "step_1");

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_complex() {
        let mut graph = DependencyGraph::new();
        // Create a more complex graph:
        // step_1 -> step_2 -> step_4
        //       \-> step_3 -> step_4
        // No cycle
        graph.add_dependency("step_2", "step_1");
        graph.add_dependency("step_3", "step_1");
        graph.add_dependency("step_4", "step_2");
        graph.add_dependency("step_4", "step_3");

        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_complex_with_cycle() {
        let mut graph = DependencyGraph::new();
        // Add a cycle: step_4 -> step_1
        graph.add_dependency("step_2", "step_1");
        graph.add_dependency("step_3", "step_1");
        graph.add_dependency("step_4", "step_2");
        graph.add_dependency("step_4", "step_3");
        graph.add_dependency("step_1", "step_4"); // Creates cycle

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_get_zero_dependency_steps() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_node("step_3");
        graph.add_dependency("step_2", "step_1");
        graph.add_dependency("step_3", "step_1");

        let zero_deps = graph.get_zero_dependency_steps();
        assert_eq!(zero_deps.len(), 1);
        assert!(zero_deps.contains(&"step_1".to_string()));
    }

    #[test]
    fn test_get_zero_dependency_steps_multiple() {
        let mut graph = DependencyGraph::new();
        graph.add_node("step_1");
        graph.add_node("step_2");
        graph.add_node("step_3");
        graph.add_dependency("step_3", "step_1");

        let zero_deps = graph.get_zero_dependency_steps();
        assert_eq!(zero_deps.len(), 2);
        assert!(zero_deps.contains(&"step_1".to_string()));
        assert!(zero_deps.contains(&"step_2".to_string()));
    }

    #[test]
    fn test_get_dependencies_nonexistent() {
        let graph = DependencyGraph::new();
        let deps = graph.get_dependencies("nonexistent");
        assert!(deps.is_empty());
    }

    #[test]
    fn test_get_dependents_nonexistent() {
        let graph = DependencyGraph::new();
        let dependents = graph.get_dependents("nonexistent");
        assert!(dependents.is_empty());
    }
}
