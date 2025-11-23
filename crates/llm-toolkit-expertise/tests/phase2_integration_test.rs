//! Phase 2 Integration Tests
//!
//! Comprehensive tests for context-aware prompt rendering and DTO integration.

use llm_toolkit_expertise::{
    context::{ContextProfile, Priority, TaskHealth},
    fragment::KnowledgeFragment,
    render::{ContextualPrompt, RenderContext},
    types::{Expertise, WeightedFragment},
};

// ============================================================================
// Context Filtering Tests
// ============================================================================

#[test]
fn test_context_filtering_by_task_type() {
    // Create expertise with conditional fragments
    let expertise = Expertise::new("rust-reviewer", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Always run cargo check".to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Check for security vulnerabilities".to_string(),
                steps: vec![
                    "Scan for unsafe code".to_string(),
                    "Verify input validation".to_string(),
                ],
            })
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec!["security-review".to_string()],
                user_states: vec![],
                task_health: None,
            }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Consider code readability".to_string(),
            ))
            .with_priority(Priority::Normal)
            .with_context(ContextProfile::Conditional {
                task_types: vec!["code-review".to_string()],
                user_states: vec![],
                task_health: None,
            }),
        );

    // Context 1: Security review
    let security_context = RenderContext::new().with_task_type("security-review");
    let security_prompt = expertise.to_prompt_with_render_context(&security_context);

    // Should include: Always fragment + security fragment
    assert!(security_prompt.contains("Always run cargo check"));
    assert!(security_prompt.contains("Check for security vulnerabilities"));
    // Should NOT include: code-review fragment
    assert!(!security_prompt.contains("Consider code readability"));

    // Context 2: Code review
    let code_context = RenderContext::new().with_task_type("code-review");
    let code_prompt = expertise.to_prompt_with_render_context(&code_context);

    // Should include: Always fragment + code-review fragment
    assert!(code_prompt.contains("Always run cargo check"));
    assert!(code_prompt.contains("Consider code readability"));
    // Should NOT include: security fragment
    assert!(!code_prompt.contains("Check for security vulnerabilities"));
}

#[test]
fn test_context_filtering_by_task_health() {
    let expertise = Expertise::new("debugger", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Standard debugging steps".to_string(),
            ))
            .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "SLOW DOWN. Ask clarifying questions.".to_string(),
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
                "STOP. Reassess the approach.".to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::OffTrack),
            }),
        );

    // OnTrack: Only standard fragment
    let on_track_prompt = expertise
        .to_prompt_with_render_context(&RenderContext::new().with_task_health(TaskHealth::OnTrack));

    assert!(on_track_prompt.contains("Standard debugging steps"));
    assert!(!on_track_prompt.contains("SLOW DOWN"));
    assert!(!on_track_prompt.contains("STOP"));

    // AtRisk: Standard + AtRisk fragment
    let at_risk_prompt = expertise
        .to_prompt_with_render_context(&RenderContext::new().with_task_health(TaskHealth::AtRisk));

    assert!(at_risk_prompt.contains("Standard debugging steps"));
    assert!(at_risk_prompt.contains("SLOW DOWN"));
    assert!(!at_risk_prompt.contains("STOP"));

    // OffTrack: Standard + OffTrack fragment
    let off_track_prompt = expertise.to_prompt_with_render_context(
        &RenderContext::new().with_task_health(TaskHealth::OffTrack),
    );

    assert!(off_track_prompt.contains("Standard debugging steps"));
    assert!(!off_track_prompt.contains("SLOW DOWN"));
    assert!(off_track_prompt.contains("STOP"));
}

#[test]
fn test_context_filtering_by_user_state() {
    let expertise = Expertise::new("tutor", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Basic concepts".to_string()))
                .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Detailed explanations with examples".to_string(),
            ))
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["beginner".to_string()],
                task_health: None,
            }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Advanced optimizations".to_string(),
            ))
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["expert".to_string()],
                task_health: None,
            }),
        );

    // Beginner context
    let beginner_prompt =
        expertise.to_prompt_with_render_context(&RenderContext::new().with_user_state("beginner"));

    assert!(beginner_prompt.contains("Basic concepts"));
    assert!(beginner_prompt.contains("Detailed explanations"));
    assert!(!beginner_prompt.contains("Advanced optimizations"));

    // Expert context
    let expert_prompt =
        expertise.to_prompt_with_render_context(&RenderContext::new().with_user_state("expert"));

    assert!(expert_prompt.contains("Basic concepts"));
    assert!(!expert_prompt.contains("Detailed explanations"));
    assert!(expert_prompt.contains("Advanced optimizations"));
}

#[test]
fn test_multiple_user_states() {
    let expertise = Expertise::new("helper", "1.0").with_fragment(
        WeightedFragment::new(KnowledgeFragment::Text("Provide extra help".to_string()))
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["beginner".to_string(), "confused".to_string()],
                task_health: None,
            }),
    );

    // Should match if ANY user_state matches
    let context1 = RenderContext::new()
        .with_user_state("beginner")
        .with_user_state("focused");

    let prompt1 = expertise.to_prompt_with_render_context(&context1);
    assert!(prompt1.contains("Provide extra help"));

    // No match
    let context2 = RenderContext::new()
        .with_user_state("expert")
        .with_user_state("focused");

    let prompt2 = expertise.to_prompt_with_render_context(&context2);
    assert!(!prompt2.contains("Provide extra help"));
}

#[test]
fn test_combined_context_conditions() {
    let expertise = Expertise::new("advanced-tutor", "1.0").with_fragment(
        WeightedFragment::new(KnowledgeFragment::Text(
            "Critical security guidance for beginners".to_string(),
        ))
        .with_priority(Priority::Critical)
        .with_context(ContextProfile::Conditional {
            task_types: vec!["security-review".to_string()],
            user_states: vec!["beginner".to_string()],
            task_health: Some(TaskHealth::AtRisk),
        }),
    );

    // All conditions match
    let matching_context = RenderContext::new()
        .with_task_type("security-review")
        .with_user_state("beginner")
        .with_task_health(TaskHealth::AtRisk);

    let matching_prompt = expertise.to_prompt_with_render_context(&matching_context);
    assert!(matching_prompt.contains("Critical security guidance"));

    // Missing one condition (wrong task_health)
    let partial_context = RenderContext::new()
        .with_task_type("security-review")
        .with_user_state("beginner")
        .with_task_health(TaskHealth::OnTrack);

    let partial_prompt = expertise.to_prompt_with_render_context(&partial_context);
    assert!(!partial_prompt.contains("Critical security guidance"));
}

// ============================================================================
// Priority Ordering Tests
// ============================================================================

#[test]
fn test_priority_based_ordering() {
    let expertise = Expertise::new("ordered-expert", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Low priority info".to_string()))
                .with_priority(Priority::Low),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Critical requirement".to_string()))
                .with_priority(Priority::Critical),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Normal guidance".to_string()))
                .with_priority(Priority::Normal),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("High priority task".to_string()))
                .with_priority(Priority::High),
        );

    let prompt = expertise.to_prompt_with_render_context(&RenderContext::new());

    // Find positions of each priority section
    let critical_pos = prompt.find("Critical requirement").unwrap();
    let high_pos = prompt.find("High priority task").unwrap();
    let normal_pos = prompt.find("Normal guidance").unwrap();
    let low_pos = prompt.find("Low priority info").unwrap();

    // Verify order: Critical < High < Normal < Low
    assert!(critical_pos < high_pos);
    assert!(high_pos < normal_pos);
    assert!(normal_pos < low_pos);
}

#[test]
fn test_priority_headers_in_output() {
    let expertise = Expertise::new("test", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Critical".to_string()))
                .with_priority(Priority::Critical),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("High".to_string()))
                .with_priority(Priority::High),
        );

    let prompt = expertise.to_prompt_with_render_context(&RenderContext::new());

    // Should have priority headers
    assert!(prompt.contains("Priority: CRITICAL"));
    assert!(prompt.contains("Priority: HIGH"));
}

// ============================================================================
// ContextualPrompt Wrapper Tests
// ============================================================================

#[test]
fn test_contextual_prompt_builder_pattern() {
    let expertise = Expertise::new("test", "1.0").with_fragment(
        WeightedFragment::new(KnowledgeFragment::Text("Content".to_string())).with_context(
            ContextProfile::Conditional {
                task_types: vec!["review".to_string()],
                user_states: vec!["beginner".to_string()],
                task_health: Some(TaskHealth::AtRisk),
            },
        ),
    );

    // Build context using fluent API
    let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
        .with_task_type("review")
        .with_user_state("beginner")
        .with_task_health(TaskHealth::AtRisk)
        .to_prompt();

    assert!(prompt.contains("Content"));
}

#[test]
fn test_contextual_prompt_direct_usage() {
    let expertise = Expertise::new("direct-test", "1.0").with_fragment(WeightedFragment::new(
        KnowledgeFragment::Text("Test content".to_string()),
    ));

    let context = RenderContext::new()
        .with_task_type("test")
        .with_user_state("user");

    let contextual = ContextualPrompt::from_expertise(&expertise, context);
    let prompt = contextual.to_prompt();

    assert!(prompt.contains("Test content"));
    assert!(prompt.contains("direct-test"));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_context_matches_always_only() {
    let expertise = Expertise::new("test", "1.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Always visible".to_string()))
                .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Conditional".to_string())).with_context(
                ContextProfile::Conditional {
                    task_types: vec!["specific".to_string()],
                    user_states: vec![],
                    task_health: None,
                },
            ),
        );

    let empty_context = RenderContext::new();
    let prompt = expertise.to_prompt_with_render_context(&empty_context);

    assert!(prompt.contains("Always visible"));
    assert!(!prompt.contains("Conditional"));
}

#[test]
fn test_no_matching_fragments() {
    let expertise = Expertise::new("test", "1.0").with_fragment(
        WeightedFragment::new(KnowledgeFragment::Text("Conditional only".to_string()))
            .with_context(ContextProfile::Conditional {
                task_types: vec!["specific-type".to_string()],
                user_states: vec![],
                task_health: None,
            }),
    );

    let wrong_context = RenderContext::new().with_task_type("different-type");
    let prompt = expertise.to_prompt_with_render_context(&wrong_context);

    // Should only have header, no fragment content
    assert!(prompt.contains("Expertise: test"));
    assert!(!prompt.contains("Conditional only"));
}

#[test]
fn test_backward_compatibility_with_legacy_to_prompt() {
    let expertise = Expertise::new("legacy-test", "1.0").with_fragment(WeightedFragment::new(
        KnowledgeFragment::Text("Content".to_string()),
    ));

    // Old API should still work
    let old_prompt = expertise.to_prompt();

    // New API with empty context should produce same result
    let new_prompt = expertise.to_prompt_with_render_context(&RenderContext::new());

    assert_eq!(old_prompt, new_prompt);
}
