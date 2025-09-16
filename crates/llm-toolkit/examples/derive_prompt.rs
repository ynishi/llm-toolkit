use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(ToPrompt, Serialize)]
#[prompt(template = "User {{name}} is a {{role}}.")]
struct UserProfile {
    name: &'static str,
    role: &'static str,
}

fn main() {
    let user = UserProfile {
        name: "Yui",
        role: "World-Class Pro Engineer",
    };

    println!("{}", user.to_prompt());
}
