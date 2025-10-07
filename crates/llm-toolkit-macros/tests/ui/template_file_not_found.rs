// Test that template_file validation catches missing files at compile time

use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(Serialize, ToPrompt)]
#[prompt(template_file = "templates/does_not_exist.jinja")]
struct MissingTemplateTest {
    name: String,
}

fn main() {}
