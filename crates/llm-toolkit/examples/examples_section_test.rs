use llm_toolkit::{ToPrompt, examples_section};
use serde::Serialize;

#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "full")]
/// Represents a user of the system.
struct User {
    /// A unique identifier for the user.
    #[prompt(example = "user-12345")]
    id: String,

    /// The user's full name.
    #[prompt(example = "Taro Yamada")]
    name: String,

    /// The user's age.
    #[serde(default)]
    age: u8,
}

#[derive(ToPrompt, Default, Serialize)]
#[prompt(mode = "full")]
/// Defines a concept for image generation.
struct Concept {
    /// The main idea for the art to be generated.
    #[prompt(example = "a cinematic, dynamic shot of a futuristic city at night")]
    prompt: String,

    /// Elements to exclude from the generation.
    #[serde(default)]
    negative_prompt: Option<String>,

    /// The style of the generation.
    #[prompt(example = "anime")]
    style: String,
}

fn main() {
    let examples = examples_section!(User, Concept);
    println!("{}", examples);
}
