//! Dependency analysis for parallel orchestrator.
//!
//! This module provides tools for analyzing template dependencies and building
//! execution dependency graphs from strategy maps.

use crate::orchestrator::{OrchestratorError, StrategyMap};
use minijinja::machinery::{ast, parse};
use std::collections::{HashMap, HashSet};

use super::DependencyGraph;

/// Extracts top-level variables from a Jinja2 template using AST parsing.
///
/// This function parses the template into an AST and traverses it to find all
/// variable references. Only top-level variable names are extracted (before any dot notation).
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
    use minijinja::machinery::WhitespaceConfig;
    use minijinja::syntax::SyntaxConfig;

    let ast = parse(
        template,
        "template",
        SyntaxConfig,
        WhitespaceConfig::default(),
    )
    .map_err(|e| OrchestratorError::ExecutionFailed(format!("Template parse error: {}", e)))?;

    let mut variables = HashSet::new();

    // The parse function returns a Stmt which wraps a Template
    // We need to extract the Template from it
    match &ast {
        ast::Stmt::Template(template) => {
            for stmt in &template.children {
                extract_vars_from_stmt(stmt, &mut variables);
            }
        }
        _ => {
            // Fallback: treat the whole thing as a statement
            extract_vars_from_stmt(&ast, &mut variables);
        }
    }

    Ok(variables)
}

/// Recursively extracts variables from an AST statement.
fn extract_vars_from_stmt(stmt: &ast::Stmt<'_>, vars: &mut HashSet<String>) {
    match stmt {
        ast::Stmt::Template(template) => {
            for child in &template.children {
                extract_vars_from_stmt(child, vars);
            }
        }
        ast::Stmt::EmitExpr(emit) => extract_vars_from_expr(&emit.expr, vars),
        ast::Stmt::EmitRaw(_) => {}
        ast::Stmt::ForLoop(for_loop) => {
            extract_vars_from_expr(&for_loop.iter, vars);
            if let Some(filter_expr) = &for_loop.filter_expr {
                extract_vars_from_expr(filter_expr, vars);
            }
            for stmt in &for_loop.body {
                extract_vars_from_stmt(stmt, vars);
            }
            for stmt in &for_loop.else_body {
                extract_vars_from_stmt(stmt, vars);
            }
        }
        ast::Stmt::IfCond(if_cond) => {
            extract_vars_from_expr(&if_cond.expr, vars);
            for stmt in &if_cond.true_body {
                extract_vars_from_stmt(stmt, vars);
            }
            for stmt in &if_cond.false_body {
                extract_vars_from_stmt(stmt, vars);
            }
        }
        ast::Stmt::WithBlock(with_block) => {
            for (_, value_expr) in &with_block.assignments {
                extract_vars_from_expr(value_expr, vars);
            }
            for stmt in &with_block.body {
                extract_vars_from_stmt(stmt, vars);
            }
        }
        ast::Stmt::Set(set_stmt) => {
            extract_vars_from_expr(&set_stmt.expr, vars);
        }
        ast::Stmt::SetBlock(set_block) => {
            if let Some(filter_expr) = &set_block.filter {
                extract_vars_from_expr(filter_expr, vars);
            }
            for stmt in &set_block.body {
                extract_vars_from_stmt(stmt, vars);
            }
        }
        ast::Stmt::AutoEscape(auto_escape) => {
            extract_vars_from_expr(&auto_escape.enabled, vars);
            for stmt in &auto_escape.body {
                extract_vars_from_stmt(stmt, vars);
            }
        }
        ast::Stmt::FilterBlock(filter_block) => {
            extract_vars_from_expr(&filter_block.filter, vars);
            for stmt in &filter_block.body {
                extract_vars_from_stmt(stmt, vars);
            }
        }
        ast::Stmt::Do(do_stmt) => extract_vars_from_call(&do_stmt.call, vars),
        _ => {}
    }
}

/// Recursively extracts variables from an AST expression.
fn extract_vars_from_expr(expr: &ast::Expr<'_>, vars: &mut HashSet<String>) {
    match expr {
        ast::Expr::Var(var) => {
            // Extract top-level variable name (before any dots)
            let name = var.id.split('.').next().unwrap_or(var.id);
            vars.insert(name.to_string());
        }
        ast::Expr::Const(_) => {}
        ast::Expr::GetAttr(get_attr) => {
            extract_vars_from_expr(&get_attr.expr, vars);
        }
        ast::Expr::GetItem(get_item) => {
            extract_vars_from_expr(&get_item.expr, vars);
            extract_vars_from_expr(&get_item.subscript_expr, vars);
        }
        ast::Expr::Filter(filter) => {
            if let Some(ref expr) = filter.expr {
                extract_vars_from_expr(expr, vars);
            }
            for arg in &filter.args {
                extract_vars_from_call_arg(arg, vars);
            }
        }
        ast::Expr::Test(test) => {
            extract_vars_from_expr(&test.expr, vars);
            for arg in &test.args {
                extract_vars_from_call_arg(arg, vars);
            }
        }
        ast::Expr::BinOp(bin_op) => {
            extract_vars_from_expr(&bin_op.left, vars);
            extract_vars_from_expr(&bin_op.right, vars);
        }
        ast::Expr::UnaryOp(unary_op) => {
            extract_vars_from_expr(&unary_op.expr, vars);
        }
        ast::Expr::Call(call) => extract_vars_from_call(call, vars),
        ast::Expr::IfExpr(if_expr) => {
            extract_vars_from_expr(&if_expr.test_expr, vars);
            extract_vars_from_expr(&if_expr.true_expr, vars);
            if let Some(false_expr) = &if_expr.false_expr {
                extract_vars_from_expr(false_expr, vars);
            }
        }
        ast::Expr::List(list) => {
            for item in &list.items {
                extract_vars_from_expr(item, vars);
            }
        }
        ast::Expr::Map(map) => {
            for key in &map.keys {
                extract_vars_from_expr(key, vars);
            }
            for value in &map.values {
                extract_vars_from_expr(value, vars);
            }
        }
        ast::Expr::Slice(slice) => {
            extract_vars_from_expr(&slice.expr, vars);
            if let Some(start) = &slice.start {
                extract_vars_from_expr(start, vars);
            }
            if let Some(stop) = &slice.stop {
                extract_vars_from_expr(stop, vars);
            }
            if let Some(step) = &slice.step {
                extract_vars_from_expr(step, vars);
            }
        }
    }
}

fn extract_vars_from_call(call: &ast::Call<'_>, vars: &mut HashSet<String>) {
    extract_vars_from_expr(&call.expr, vars);
    for arg in &call.args {
        extract_vars_from_call_arg(arg, vars);
    }
}

fn extract_vars_from_call_arg(arg: &ast::CallArg<'_>, vars: &mut HashSet<String>) {
    match arg {
        ast::CallArg::Pos(expr)
        | ast::CallArg::PosSplat(expr)
        | ast::CallArg::Kwarg(_, expr)
        | ast::CallArg::KwargSplat(expr) => extract_vars_from_expr(expr, vars),
    }
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
            // Skip built-in variables like "task"
            if var == "task" {
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
    fn test_extract_variables_in_filter_with_args() {
        // Test complex case: filters with arguments that are also variables
        let template = "{{ my_var | filter(other_var) | another_filter(third_var, 'constant') }}";
        let vars = extract_template_variables(template).unwrap();

        // Should extract all three variables
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("my_var"));
        assert!(vars.contains("other_var"));
        assert!(vars.contains("third_var"));
    }

    #[test]
    fn test_extract_variables_complex_expressions() {
        // Test complex expressions with nested filters, conditionals, and operations
        let template = r#"
            {% if condition_var %}
                {{ data.field | process(param_var) }}
                {{ result_var + offset_var }}
            {% endif %}
        "#;
        let vars = extract_template_variables(template).unwrap();

        // Should extract top-level variables from all parts
        assert_eq!(vars.len(), 5);
        assert!(vars.contains("condition_var"));
        assert!(vars.contains("data"));
        assert!(vars.contains("param_var"));
        assert!(vars.contains("result_var"));
        assert!(vars.contains("offset_var"));
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
            "Do {{ task }}".to_string(),
            "Output 1".to_string(),
        ));

        let graph = build_dependency_graph(&strategy).unwrap();

        // step_1 should have no dependencies (task is a built-in)
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
