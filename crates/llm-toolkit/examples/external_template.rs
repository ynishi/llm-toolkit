//! Example demonstrating the use of external template files with the ToPrompt derive macro.
//!
//! This example shows how to:
//! - Load templates from external files using `template_file`
//! - Enable template validation with the `validate` flag
//! - Use Jinja2 syntax in external templates
//!
//! Note: Template files are located in `examples/templates/` directory.

use llm_toolkit::ToPrompt;
use serde::Serialize;

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
}
