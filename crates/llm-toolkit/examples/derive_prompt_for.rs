use llm_toolkit::{ToPrompt, ToPromptFor};
use serde::Serialize;

// The target struct for the prompt. It doesn't need any special derives.
#[derive(Serialize)]
struct Agent {
    name: String,
    role: String,
}

#[derive(ToPrompt, ToPromptFor, Serialize, Default)]
// This first `prompt` attribute enables mode-based generation for `ToPrompt`
#[prompt(mode = "full")]
// This second `prompt_for` attribute defines the template for `ToPromptFor<Agent>`
#[prompt_for(
    target = "Agent",
    template = r#"\nHello, {{ target.name }}. As a {{ target.role }}, you can use the following tool.\n\n### Tool Schema\n{self:schema_only}\n\n### Tool Example\n{self:example_only}\n\nThe tool's name is '{{ self.name }}'.\n"#
)]
/// A tool that can be used by an agent.
struct Tool {
    /// The name of the tool.
    #[prompt(example = "file_writer")]
    name: String,
    /// A description of what the tool does.
    #[prompt(example = "Writes content to a file.")]
    description: String,
    /// The version of the tool.
    #[serde(default)]
    version: u32,
}

fn main() {
    let agent = Agent {
        name: "Yui".to_string(),
        role: "Pro Engineer".to_string(),
    };

    let tool = Tool {
        name: "file_writer_tool".to_string(),
        ..Default::default()
    };

    println!("--- Calling to_prompt_for (default 'full' mode) ---");
    let prompt = tool.to_prompt_for(&agent);
    println!("{}", prompt);

    // Note: The `to_prompt_for_with_mode`'s `mode` argument affects `{self}` placeholders,
    // but not `{self:schema_only}` or other hardcoded-mode placeholders.
    // Since our template doesn't use a plain `{self}`, the mode argument won't change the output here.
    println!("--- Calling to_prompt_for_with_mode(\"schema_only\") ---");
    let prompt_schema = tool.to_prompt_for_with_mode(&agent, "schema_only");
    println!("{}", prompt_schema);
}
