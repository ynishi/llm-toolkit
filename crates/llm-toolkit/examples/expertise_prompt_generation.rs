//! Demonstrate prompt generation with different contexts

use llm_toolkit::agent::expertise::{Expertise, KnowledgeFragment, RenderContext, WeightedFragment};
use llm_toolkit::context::{ContextProfile, Priority, TaskHealth};

fn main() {
    println!("=== Context-Aware Prompt Generation ===\n");

    // Create an expertise with context-conditional fragments
    let expertise = Expertise::new("adaptive-assistant", "1.0.0")
        .with_tag("adaptive")
        .with_tag("context-aware")
        // Always active: Basic guidelines
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Be helpful, concise, and accurate in all responses.".to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Always),
        )
        // On Track: Speed mode
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Optimize for speed and efficiency".to_string(),
                steps: vec![
                    "Provide direct, concise answers".to_string(),
                    "Skip verbose explanations unless asked".to_string(),
                    "Use shortcuts and best practices".to_string(),
                ],
            })
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::OnTrack),
            }),
        )
        // At Risk: Careful mode
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Proceed with caution and verification".to_string(),
                steps: vec![
                    "Double-check all assumptions".to_string(),
                    "Ask clarifying questions".to_string(),
                    "Verify requirements before proceeding".to_string(),
                    "Explain reasoning for each decision".to_string(),
                ],
            })
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::AtRisk),
            }),
        )
        // Off Track: Stop mode
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "STOP. The current approach is not working. Reassess the problem, consult with the user, and propose a different strategy before continuing.".to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::OffTrack),
            }),
        )
        // Beginner user state
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Provide detailed explanations and educational context. Avoid jargon.".to_string(),
            ))
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["beginner".to_string()],
                task_health: None,
            }),
        )
        // Debug task type
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Systematic debugging approach".to_string(),
                steps: vec![
                    "Reproduce the issue".to_string(),
                    "Isolate the root cause".to_string(),
                    "Propose minimal fix".to_string(),
                    "Verify fix doesn't break other functionality".to_string(),
                ],
            })
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec!["debug".to_string()],
                user_states: vec![],
                task_health: None,
            }),
        );

    // Test different contexts
    let scenarios = vec![
        ("Default (no context)", RenderContext::new()),
        (
            "On Track",
            RenderContext::new().with_task_health(TaskHealth::OnTrack),
        ),
        (
            "At Risk",
            RenderContext::new().with_task_health(TaskHealth::AtRisk),
        ),
        (
            "Off Track",
            RenderContext::new().with_task_health(TaskHealth::OffTrack),
        ),
        (
            "Beginner User",
            RenderContext::new().with_user_state("beginner"),
        ),
        ("Debug Task", RenderContext::new().with_task_type("debug")),
        (
            "Debug + At Risk",
            RenderContext::new()
                .with_task_type("debug")
                .with_task_health(TaskHealth::AtRisk),
        ),
    ];

    for (name, context) in scenarios {
        println!("### Scenario: {}\n", name);
        let prompt = expertise.to_prompt_with_context(&context);
        println!("{}", prompt);
        println!("---\n");
    }

    println!("\n=== Visualization ===\n");
    println!("{}", expertise.to_tree());
}
