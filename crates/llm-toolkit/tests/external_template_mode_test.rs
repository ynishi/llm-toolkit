use llm_toolkit::ToPrompt;
use serde::Serialize;

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(mode = "full")]
struct TaskDescription {
    description: String,
}

impl TaskDescription {
    fn to_prompt_with_mode(&self, mode: &str) -> String {
        match mode {
            "brief" => format!(
                "Brief: {}",
                &self.description[..30.min(self.description.len())]
            ),
            "full" => format!("Full description: {}", self.description),
            _ => self.description.clone(),
        }
    }
}

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(mode = "full")]
struct TaskDetails {
    details: String,
}

impl TaskDetails {
    fn to_prompt_with_mode(&self, mode: &str) -> String {
        match mode {
            "full" => format!("Complete details:\n{}", self.details),
            _ => self.details.clone(),
        }
    }
}

#[derive(Debug, Serialize, ToPrompt)]
#[prompt(template_file = "tests/templates/with_mode.jinja")]
struct TaskConfig {
    title: String,
    description: TaskDescription,
    details: TaskDetails,
}

#[test]
fn test_external_template_with_mode() {
    let task = TaskConfig {
        title: "Important Task".to_string(),
        description: TaskDescription {
            description: "This is a very long description that should be shortened in brief mode"
                .to_string(),
        },
        details: TaskDetails {
            details: "Here are all the nitty-gritty details of the task".to_string(),
        },
    };

    let prompt = task.to_prompt();
    println!("Generated prompt:\n{}", prompt);
    assert!(prompt.contains("Important Task"));
    assert!(prompt.contains("Brief:"));
    assert!(prompt.contains("Complete details:"));
}
