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

    let output = user.to_prompt();
    println!("{}", output);

    // E2E Test Assertions
    let expected = "User Yui is a World-Class Pro Engineer.";
    if output != expected {
        eprintln!("❌ ASSERTION FAILED:");
        eprintln!("  Expected: {}", expected);
        eprintln!("  Got:      {}", output);
        std::process::exit(1);
    }

    println!("✅ E2E Test: PASSED");
}
