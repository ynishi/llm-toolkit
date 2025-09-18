use llm_toolkit::ToPrompt;

/// Represents different user intents for a chatbot
#[derive(ToPrompt, Debug)]
pub enum UserIntent {
    /// User wants to greet or say hello
    Greeting,
    /// User is asking for help or assistance
    Help,
    /// User wants to know the current weather
    WeatherQuery,
    /// User wants to set a reminder for later
    SetReminder,
    /// User is saying goodbye
    Farewell,
}

/// Task priority levels with advanced prompt attributes
#[derive(ToPrompt, Debug)]
pub enum Priority {
    /// Urgent tasks that need immediate attention
    Critical,
    #[prompt("High priority - important but can wait a bit")]
    High,
    /// Regular priority tasks
    Medium,
    /// Tasks that can be done when time permits
    Low,
    #[prompt(skip)]
    Deprecated, // This variant will be excluded from prompts
}

/// Example enum showing all prompt attribute features
#[derive(ToPrompt, Debug)]
pub enum FeatureDemo {
    /// This uses a doc comment for description
    WithDocComment,

    #[prompt("This overrides any doc comment with a custom description")]
    WithCustomDescription,

    #[prompt(skip)]
    /// Even though this has a doc comment, it will be skipped
    SkippedVariant,

    // This variant has no doc comment, so only the name will be shown
    NoDescription,
}

fn main() {
    // Create an instance and demonstrate the generated prompt
    let intent = UserIntent::Greeting;
    println!("UserIntent prompt:");
    println!("{}", intent.to_prompt());
    println!("\n---\n");

    let priority = Priority::High;
    println!("Priority prompt (with custom description and skip):");
    println!("{}", priority.to_prompt());
    println!("\n---\n");

    let demo = FeatureDemo::WithDocComment;
    println!("FeatureDemo prompt (showing all attribute features):");
    println!("{}", demo.to_prompt());
}
