//! Example demonstrating the use of external template files with the ToPrompt derive macro.
//!
//! This example shows how to:
//! - Load templates from external files using `template_file`
//! - Enable template validation with the `validate` flag
//! - Use Jinja2 syntax in external templates

use llm_toolkit::ToPrompt;
use serde::Serialize;
use std::fs;
use std::path::Path;

// First, let's create a template file for this example
fn setup_template_files() {
    let template_dir = Path::new("examples/templates");
    if !template_dir.exists() {
        fs::create_dir_all(template_dir).expect("Failed to create templates directory");
    }

    // Create a user profile template
    let profile_template = r#"=== User Profile ===
Name: {{ name }}
Email: {{ email }}
Role: {{ role }}
Years of Experience: {{ years_experience }}

Bio:
{{ bio }}

This profile is for {{ name }}, who works as a {{ role }}."#;

    fs::write("examples/templates/profile.jinja", profile_template)
        .expect("Failed to write profile template");

    // Create a project template
    let project_template = r#"## Project: {{ name }}

**Description:** {{ description }}

**Status:** {{ status }}
**Priority:** {{ priority }}

### Team Members:
{% for member in team_members %}
- {{ member }}
{% endfor %}

Project "{{ name }}" is currently {{ status }} with {{ priority }} priority."#;

    fs::write("examples/templates/project.jinja", project_template)
        .expect("Failed to write project template");
}

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(template_file = "examples/templates/profile.jinja")]
struct UserProfile {
    name: String,
    email: String,
    role: String,
    years_experience: u32,
    bio: String,
}

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(template_file = "examples/templates/project2.jinja", validate = true)]
struct Project {
    name: String,
    description: String,
    status: String,
    priority: String,
}

fn main() {
    // Setup template files
    setup_template_files();

    // Example 1: User Profile with external template
    let user = UserProfile {
        name: "Alice Johnson".to_string(),
        email: "alice@example.com".to_string(),
        role: "Senior Software Engineer".to_string(),
        years_experience: 8,
        bio: "Passionate about building scalable systems and mentoring junior developers."
            .to_string(),
    };

    println!("User Profile Prompt:");
    println!("{}\n", user.to_prompt());

    // Example 2: Project with external template and validation
    let project = Project {
        name: "AI Integration Platform".to_string(),
        description:
            "A comprehensive platform for integrating various AI models into business workflows"
                .to_string(),
        status: "In Progress".to_string(),
        priority: "High".to_string(),
    };

    println!("Project Prompt:");
    println!("{}\n", project.to_prompt());

    // Example 3: Using to_prompt_parts for multimodal content
    let parts = user.to_prompt_parts();
    println!("User Profile Parts Count: {}", parts.len());
    for (i, part) in parts.iter().enumerate() {
        println!("Part {}: {:?}", i + 1, part);
    }

    // Clean up (optional - you might want to keep the templates)
    // fs::remove_dir_all("examples/templates").ok();
}
