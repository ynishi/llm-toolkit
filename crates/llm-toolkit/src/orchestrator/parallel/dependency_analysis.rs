//! Dependency analysis for parallel orchestrator.
//!
//! This module provides tools for analyzing template dependencies and building
//! execution dependency graphs from strategy maps.

use crate::orchestrator::{OrchestratorError, StrategyMap};
use std::collections::{HashMap, HashSet};

use super::DependencyGraph;

/// Extracts top-level variables from a Jinja2 template using regex.
///
/// This function uses regular expressions to find all variable references
/// in the template. Only top-level variable names are extracted (before any dot notation).
///
/// # Arguments
///
/// * `template` - A Jinja2 template string
///
/// # Returns
///
/// A set of variable names found in the template
///
/// # Examples
///
/// ```ignore
/// let vars = extract_template_variables("Process {{ step_1_output }}").unwrap();
/// assert!(vars.contains("step_1_output"));
/// ```
pub fn extract_template_variables(template: &str) -> Result<HashSet<String>, OrchestratorError> {
    use regex::Regex;

    let mut variables = HashSet::new();

    // Match {{ variable }} patterns
    let var_re = Regex::new(r"\{\{[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)").map_err(|e| {
        OrchestratorError::ExecutionFailed(format!("Regex compilation failed: {}", e))
    })?;

    for cap in var_re.captures_iter(template) {
        if let Some(m) = cap.get(1) {
            let full_var = m.as_str();
            // Extract top-level variable (before any dot)
            let top_level = full_var.split('.').next().unwrap_or(full_var);
            variables.insert(top_level.to_string());
        }
    }

    // Match {% if variable %}, {% for variable %}, etc.
    let control_re = Regex::new(r"\{%[ \t]+(?:if|for|elif)[ \t]+([a-zA-Z_][a-zA-Z0-9_]*)")
        .map_err(|e| {
            OrchestratorError::ExecutionFailed(format!("Regex compilation failed: {}", e))
        })?;

    for cap in control_re.captures_iter(template) {
        if let Some(m) = cap.get(1) {
            let full_var = m.as_str();
            let top_level = full_var.split('.').next().unwrap_or(full_var);
            variables.insert(top_level.to_string());
        }
    }

    Ok(variables)
}

/// Builds a dependency graph from a strategy map.
///
/// This function analyzes all steps in the strategy map, extracts variables
/// from their intent templates, and constructs a dependency graph showing
/// which steps depend on which other steps.
///
/// # Arguments
///
/// * `strategy` - The strategy map to analyze
///
/// # Returns
///
/// A dependency graph representing step dependencies
///
/// # Errors
///
/// Returns an error if:
/// - Template parsing fails
/// - A cycle is detected in the dependency graph
///
/// # Examples
///
/// ```ignore
/// let graph = build_dependency_graph(&strategy_map)?;
/// let ready_steps = graph.get_zero_dependency_steps();
/// ```
pub fn build_dependency_graph(
    strategy: &StrategyMap,
) -> Result<DependencyGraph, OrchestratorError> {
    let mut graph = DependencyGraph::new();

    // Build output lookup: variable_name -> step_id
    let mut output_lookup: HashMap<String, String> = HashMap::new();
    for step in &strategy.steps {
        // Register default output key: {step_id}_output
        output_lookup.insert(format!("{}_output", step.step_id), step.step_id.clone());

        // Register custom output key if exists
        if let Some(ref output_key) = step.output_key {
            output_lookup.insert(output_key.clone(), step.step_id.clone());
        }

        // Add node to graph
        graph.add_node(&step.step_id);
    }

    // Analyze dependencies for each step
    for step in &strategy.steps {
        let variables = extract_template_variables(&step.intent_template)?;

        for var in variables {
            // Skip built-in variables like "task" and "previous_output"
            if var == "task" || var == "previous_output" {
                continue;
            }

            // Find which step produces this variable
            if let Some(producer_step_id) = output_lookup.get(&var) {
                // Don't add self-dependency
                if producer_step_id != &step.step_id {
                    graph.add_dependency(&step.step_id, producer_step_id);
                }
            }
        }
    }

    // Detect cycles
    if graph.has_cycle() {
        return Err(OrchestratorError::ExecutionFailed(
            "Cyclic dependency detected in strategy map".to_string(),
        ));
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::StrategyStep;

    #[test]
    fn test_extract_variables_simple() {
        let template = "Process {{ previous_output }}";
        let vars = extract_template_variables(template).unwrap();

        assert_eq!(vars.len(), 1);
        assert!(vars.contains("previous_output"));
    }

    #[test]
    fn test_extract_variables_multiple() {
        let template = "Use {{ step_1_output }} and {{ step_2_output }}";
        let vars = extract_template_variables(template).unwrap();

        assert_eq!(vars.len(), 2);
        assert!(vars.contains("step_1_output"));
        assert!(vars.contains("step_2_output"));
    }

    #[test]
    fn test_extract_variables_dot_notation() {
        let template = "Get {{ step_1_output.field }}";
        let vars = extract_template_variables(template).unwrap();

        // Should extract top-level variable only
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("step_1_output"));
    }

    #[test]
    fn test_extract_variables_in_filter() {
        let template = "{{ step_1_output | upper }}";
        let vars = extract_template_variables(template).unwrap();

        assert_eq!(vars.len(), 1);
        assert!(vars.contains("step_1_output"));
    }

    #[test]
    fn test_extract_variables_in_condition() {
        let template = "{% if step_1_output %}Use {{ step_2_output }}{% endif %}";
        let vars = extract_template_variables(template).unwrap();

        assert_eq!(vars.len(), 2);
        assert!(vars.contains("step_1_output"));
        assert!(vars.contains("step_2_output"));
    }

    #[test]
    fn test_extract_variables_no_duplicates() {
        let template = "{{ data }} and {{ data }} again";
        let vars = extract_template_variables(template).unwrap();

        assert_eq!(vars.len(), 1);
        assert!(vars.contains("data"));
    }

    #[test]
    fn test_build_dependency_graph_simple() {
        let mut strategy = StrategyMap::new("Test".to_string());
        strategy.add_step(StrategyStep::new(
            "step_1".to_string(),
            "First".to_string(),
            "Agent1".to_string(),
            "Do {{ task }}".to_string(),
            "Output 1".to_string(),
        ));
        strategy.add_step(StrategyStep::new(
            "step_2".to_string(),
            "Second".to_string(),
            "Agent2".to_string(),
            "Process {{ step_1_output }}".to_string(),
            "Output 2".to_string(),
        ));

        let graph = build_dependency_graph(&strategy).unwrap();

        // step_2 depends on step_1
        assert!(graph.get_dependencies("step_2").contains("step_1"));
        // step_1 has no dependencies
        assert!(graph.get_dependencies("step_1").is_empty());
    }

    #[test]
    fn test_build_dependency_graph_with_output_key() {
        let mut strategy = StrategyMap::new("Test".to_string());
        let mut step1 = StrategyStep::new(
            "step_1".to_string(),
            "First".to_string(),
            "Agent1".to_string(),
            "Do {{ task }}".to_string(),
            "Output 1".to_string(),
        );
        step1.output_key = Some("custom_output".to_string());
        strategy.add_step(step1);

        strategy.add_step(StrategyStep::new(
            "step_2".to_string(),
            "Second".to_string(),
            "Agent2".to_string(),
            "Process {{ custom_output }}".to_string(),
            "Output 2".to_string(),
        ));

        let graph = build_dependency_graph(&strategy).unwrap();

        // step_2 depends on step_1 (via custom_output)
        assert!(graph.get_dependencies("step_2").contains("step_1"));
    }

    #[test]
    fn test_build_dependency_graph_detects_cycle() {
        let mut strategy = StrategyMap::new("Test".to_string());
        strategy.add_step(StrategyStep::new(
            "step_1".to_string(),
            "First".to_string(),
            "Agent1".to_string(),
            "Do {{ step_2_output }}".to_string(),
            "Output 1".to_string(),
        ));
        strategy.add_step(StrategyStep::new(
            "step_2".to_string(),
            "Second".to_string(),
            "Agent2".to_string(),
            "Process {{ step_1_output }}".to_string(),
            "Output 2".to_string(),
        ));

        let result = build_dependency_graph(&strategy);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_dependency_graph_ignores_builtins() {
        let mut strategy = StrategyMap::new("Test".to_string());
        strategy.add_step(StrategyStep::new(
            "step_1".to_string(),
            "First".to_string(),
            "Agent1".to_string(),
            "Do {{ task }} with {{ previous_output }}".to_string(),
            "Output 1".to_string(),
        ));

        let graph = build_dependency_graph(&strategy).unwrap();

        // step_1 should have no dependencies (task and previous_output are built-ins)
        assert!(graph.get_dependencies("step_1").is_empty());
    }

    #[test]
    fn test_build_dependency_graph_diamond() {
        let mut strategy = StrategyMap::new("Test".to_string());
        strategy.add_step(StrategyStep::new(
            "step_1".to_string(),
            "Root".to_string(),
            "Agent1".to_string(),
            "{{ task }}".to_string(),
            "Output 1".to_string(),
        ));
        strategy.add_step(StrategyStep::new(
            "step_2".to_string(),
            "Left".to_string(),
            "Agent2".to_string(),
            "{{ step_1_output }}".to_string(),
            "Output 2".to_string(),
        ));
        strategy.add_step(StrategyStep::new(
            "step_3".to_string(),
            "Right".to_string(),
            "Agent3".to_string(),
            "{{ step_1_output }}".to_string(),
            "Output 3".to_string(),
        ));
        strategy.add_step(StrategyStep::new(
            "step_4".to_string(),
            "Merge".to_string(),
            "Agent4".to_string(),
            "{{ step_2_output }} and {{ step_3_output }}".to_string(),
            "Output 4".to_string(),
        ));

        let graph = build_dependency_graph(&strategy).unwrap();

        // Verify diamond structure
        assert!(graph.get_dependencies("step_1").is_empty());
        assert!(graph.get_dependencies("step_2").contains("step_1"));
        assert!(graph.get_dependencies("step_3").contains("step_1"));
        assert!(graph.get_dependencies("step_4").contains("step_2"));
        assert!(graph.get_dependencies("step_4").contains("step_3"));
    }
}
