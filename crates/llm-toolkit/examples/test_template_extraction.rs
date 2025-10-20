use regex::Regex;
use std::collections::HashSet;

fn extract_template_variables(template: &str) -> HashSet<String> {
    let mut variables = HashSet::new();

    // Match {{ variable }} patterns
    let var_re = Regex::new(r"\{\{[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)").unwrap();

    for cap in var_re.captures_iter(template) {
        if let Some(m) = cap.get(1) {
            let full_var = m.as_str();
            // Extract top-level variable (before any dot)
            let top_level = full_var.split('.').next().unwrap_or(full_var);
            variables.insert(top_level.to_string());
        }
    }

    variables
}

fn main() {
    println!("=== Template Variable Extraction Test ===\n");

    let test_cases = vec![
        "Process {{ step_1_output }}",
        "Use {{ step_1_output }} and {{ step_2_output }}",
        "Get {{ step_1_output.field }}",
        "Do {{ task }}",
        "{{ data }} and {{ data }} again",
    ];

    for template in test_cases {
        let vars = extract_template_variables(template);
        println!("Template: '{}'", template);
        println!("Variables: {:?}", vars);
        println!();
    }
}
