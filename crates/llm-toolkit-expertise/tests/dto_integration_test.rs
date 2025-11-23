//! DTO Integration Pattern Tests
//!
//! Tests for ContextualPrompt integration with ToPrompt-based DTO pattern.
//! These tests demonstrate how ContextualPrompt works in real-world scenarios.

#![cfg(feature = "integration")]

use llm_toolkit_expertise::{
    context::{ContextProfile, Priority, TaskHealth},
    fragment::KnowledgeFragment,
    render::{ContextualPrompt, RenderContext},
    types::{Expertise, WeightedFragment},
};
use llm_toolkit::ToPrompt;
use serde::Serialize;

// ============================================================================
// DTO Pattern Examples
// ============================================================================

/// Example DTO for agent request with contextual expertise
#[derive(Serialize, ToPrompt)]
#[prompt(template = r#"# Expert Knowledge

{{expertise}}

# Task

{{task}}

# Additional Context

User Level: {{user_level}}
"#)]
struct AgentRequestDto {
    expertise: String,  // We'll use ContextualPrompt.to_prompt() result
    task: String,
    user_level: String,
}

#[test]
fn test_dto_with_contextual_prompt() {
    // Create expertise with conditional fragments
    let expertise = Expertise::new("rust-tutor", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "You are a Rust programming tutor".to_string(),
            ))
            .with_priority(Priority::High)
            .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Provide detailed explanations with examples".to_string(),
            ))
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["beginner".to_string()],
                task_health: None,
            }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Focus on advanced patterns and optimizations".to_string(),
            ))
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["expert".to_string()],
                task_health: None,
            }),
        );

    // Beginner context
    let beginner_context = RenderContext::new().with_user_state("beginner");
    let beginner_expertise = ContextualPrompt::from_expertise(&expertise, beginner_context).to_prompt();

    let beginner_dto = AgentRequestDto {
        expertise: beginner_expertise,
        task: "Explain ownership and borrowing".to_string(),
        user_level: "beginner".to_string(),
    };

    let beginner_prompt = beginner_dto.to_prompt();

    // Should contain base + beginner-specific
    assert!(beginner_prompt.contains("Rust programming tutor"));
    assert!(beginner_prompt.contains("detailed explanations with examples"));
    assert!(!beginner_prompt.contains("advanced patterns"));
    assert!(beginner_prompt.contains("Explain ownership and borrowing"));

    // Expert context
    let expert_context = RenderContext::new().with_user_state("expert");
    let expert_expertise = ContextualPrompt::from_expertise(&expertise, expert_context).to_prompt();

    let expert_dto = AgentRequestDto {
        expertise: expert_expertise,
        task: "Optimize this async code".to_string(),
        user_level: "expert".to_string(),
    };

    let expert_prompt = expert_dto.to_prompt();

    // Should contain base + expert-specific
    assert!(expert_prompt.contains("Rust programming tutor"));
    assert!(!expert_prompt.contains("detailed explanations with examples"));
    assert!(expert_prompt.contains("advanced patterns"));
    assert!(expert_prompt.contains("Optimize this async code"));
}

/// Multi-agent request DTO
#[derive(Serialize, ToPrompt)]
#[prompt(template = r#"# Multi-Agent System

## Primary Agent
{{primary_expertise}}

## Secondary Agent
{{secondary_expertise}}

## Workflow
{{workflow}}
"#)]
struct MultiAgentDto {
    primary_expertise: String,
    secondary_expertise: String,
    workflow: String,
}

#[test]
fn test_multi_agent_dto_with_different_contexts() {
    // Primary agent: Security reviewer (AtRisk context)
    let security_expert = Expertise::new("security-reviewer", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Security review specialist".to_string(),
            ))
            .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "CRITICAL: Extra vigilance required".to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::AtRisk),
            }),
        );

    // Secondary agent: Code formatter (always OnTrack)
    let formatter_expert = Expertise::new("formatter", "1.0").with_fragment(
        WeightedFragment::new(KnowledgeFragment::Text("Format code consistently".to_string()))
            .with_context(ContextProfile::Always),
    );

    // Different contexts for each agent
    let at_risk_context = RenderContext::new().with_task_health(TaskHealth::AtRisk);
    let normal_context = RenderContext::new();

    let dto = MultiAgentDto {
        primary_expertise: ContextualPrompt::from_expertise(&security_expert, at_risk_context)
            .to_prompt(),
        secondary_expertise: ContextualPrompt::from_expertise(&formatter_expert, normal_context)
            .to_prompt(),
        workflow: "1. Security review\n2. Format code".to_string(),
    };

    let prompt = dto.to_prompt();

    // Primary should have critical alert
    assert!(prompt.contains("Security review specialist"));
    assert!(prompt.contains("CRITICAL: Extra vigilance required"));

    // Secondary should be normal
    assert!(prompt.contains("Format code consistently"));
}

// ============================================================================
// Real-world Scenario Tests
// ============================================================================

#[test]
fn test_adaptive_debugging_assistant() {
    let debugger = Expertise::new("debugger", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Standard debugging procedure".to_string(),
                steps: vec![
                    "Reproduce the issue".to_string(),
                    "Isolate the cause".to_string(),
                    "Fix and verify".to_string(),
                ],
            })
            .with_priority(Priority::Normal)
            .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "⚠️ SLOW DOWN. The task is at risk. Ask clarifying questions before proceeding."
                    .to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::AtRisk),
            }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "🚫 STOP. Reassess the entire approach. Consider alternative solutions."
                    .to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::OffTrack),
            }),
        );

    // Scenario 1: OnTrack - normal debugging
    let on_track = ContextualPrompt::from_expertise(&debugger, RenderContext::new().with_task_health(TaskHealth::OnTrack))
        .to_prompt();

    assert!(on_track.contains("Standard debugging procedure"));
    assert!(!on_track.contains("SLOW DOWN"));
    assert!(!on_track.contains("STOP"));

    // Scenario 2: AtRisk - cautious mode
    let at_risk = ContextualPrompt::from_expertise(&debugger, RenderContext::new().with_task_health(TaskHealth::AtRisk))
        .to_prompt();

    assert!(at_risk.contains("Standard debugging procedure"));
    assert!(at_risk.contains("SLOW DOWN"));
    assert!(!at_risk.contains("STOP"));

    // Scenario 3: OffTrack - intervention mode
    let off_track = ContextualPrompt::from_expertise(
        &debugger,
        RenderContext::new().with_task_health(TaskHealth::OffTrack),
    )
    .to_prompt();

    assert!(off_track.contains("Standard debugging procedure"));
    assert!(!off_track.contains("SLOW DOWN"));
    assert!(off_track.contains("STOP"));

    // The prompts should be different
    assert_ne!(on_track, at_risk);
    assert_ne!(at_risk, off_track);
}

#[test]
fn test_context_aware_code_reviewer() {
    let reviewer = Expertise::new("code-reviewer", "2.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Review code for correctness and style".to_string(),
            ))
            .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Security-focused review checklist".to_string(),
                steps: vec![
                    "Check for SQL injection vulnerabilities".to_string(),
                    "Verify input validation".to_string(),
                    "Review authentication logic".to_string(),
                ],
            })
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec!["security-review".to_string()],
                user_states: vec![],
                task_health: None,
            }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Explain issues clearly with examples for learning".to_string(),
            ))
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["beginner".to_string()],
                task_health: None,
            }),
        );

    // Combined context: Security review by a beginner
    let combined_context = RenderContext::new()
        .with_task_type("security-review")
        .with_user_state("beginner");

    let combined_prompt =
        ContextualPrompt::from_expertise(&reviewer, combined_context).to_prompt();

    // Should have all three fragments
    assert!(combined_prompt.contains("Review code for correctness"));
    assert!(combined_prompt.contains("Security-focused review checklist"));
    assert!(combined_prompt.contains("SQL injection"));
    assert!(combined_prompt.contains("Explain issues clearly"));
}

#[test]
fn test_priority_ordering_in_critical_situations() {
    let expert = Expertise::new("emergency-responder", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Reference docs".to_string()))
                .with_priority(Priority::Low),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Best practices".to_string()))
                .with_priority(Priority::Normal),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Important guideline".to_string()))
                .with_priority(Priority::High),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "MUST DO FIRST".to_string(),
            ))
            .with_priority(Priority::Critical),
        );

    let prompt = ContextualPrompt::from_expertise(&expert, RenderContext::new()).to_prompt();

    // Extract positions
    let critical_pos = prompt.find("MUST DO FIRST").unwrap();
    let high_pos = prompt.find("Important guideline").unwrap();
    let normal_pos = prompt.find("Best practices").unwrap();
    let low_pos = prompt.find("Reference docs").unwrap();

    // Verify correct ordering
    assert!(critical_pos < high_pos, "Critical should come before High");
    assert!(high_pos < normal_pos, "High should come before Normal");
    assert!(
        normal_pos < low_pos,
        "Normal should come before Low"
    );
}
