//! Integration tests for llm-toolkit expertise module

use llm_toolkit::agent::expertise::{
    Anchor, Expertise, KnowledgeFragment, RenderContext, WeightedFragment,
};
use llm_toolkit::context::{ContextProfile, Priority, TaskHealth};

#[test]
fn test_complete_expertise_workflow() {
    // Create a complete expertise
    let expertise = Expertise::new("test-expertise", "1.0.0")
        .with_tag("test")
        .with_tag("integration")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Critical rule".to_string()))
                .with_priority(Priority::Critical),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Logic {
                instruction: "Test instruction".to_string(),
                steps: vec!["Step 1".to_string(), "Step 2".to_string()],
            })
            .with_priority(Priority::High),
        );

    // Test serialization roundtrip
    let json = serde_json::to_string(&expertise).expect("Serialization failed");
    let deserialized: Expertise = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(deserialized.id, "test-expertise");
    assert_eq!(deserialized.version, "1.0.0");
    assert_eq!(deserialized.tags.len(), 2);
    assert_eq!(deserialized.content.len(), 2);
}

#[test]
fn test_prompt_generation_with_all_fragment_types() {
    let expertise = Expertise::new("comprehensive", "1.0.0")
        .with_fragment(WeightedFragment::new(KnowledgeFragment::Logic {
            instruction: "Logic test".to_string(),
            steps: vec!["Step A".to_string()],
        }))
        .with_fragment(WeightedFragment::new(KnowledgeFragment::Guideline {
            rule: "Guideline test".to_string(),
            anchors: vec![Anchor {
                context: "Context".to_string(),
                positive: "Good".to_string(),
                negative: "Bad".to_string(),
                reason: "Because".to_string(),
            }],
        }))
        .with_fragment(WeightedFragment::new(KnowledgeFragment::QualityStandard {
            criteria: vec!["Criterion 1".to_string()],
            passing_grade: "Pass".to_string(),
        }))
        .with_fragment(WeightedFragment::new(KnowledgeFragment::Text(
            "Plain text".to_string(),
        )))
        .with_fragment(WeightedFragment::new(KnowledgeFragment::ToolDefinition(
            serde_json::json!({
                "name": "test_tool",
                "description": "A test tool"
            }),
        )));

    let prompt = expertise.to_prompt();

    // Verify all fragment types are included
    assert!(prompt.contains("Logic test"));
    assert!(prompt.contains("Guideline test"));
    assert!(prompt.contains("Quality Standard"));
    assert!(prompt.contains("Plain text"));
    assert!(prompt.contains("test_tool"));
}

#[test]
fn test_context_filtering_combinations() {
    let expertise = Expertise::new("context-test", "1.0.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Always visible".to_string()))
                .with_context(ContextProfile::Always),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Debug only".to_string())).with_context(
                ContextProfile::Conditional {
                    task_types: vec!["Debug".to_string()],
                    user_states: vec![],
                    task_health: None,
                },
            ),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Beginner only".to_string()))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec!["Beginner".to_string()],
                    task_health: None,
                }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("At risk only".to_string()))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec![],
                    task_health: Some(TaskHealth::AtRisk),
                }),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Debug + Beginner".to_string()))
                .with_context(ContextProfile::Conditional {
                    task_types: vec!["Debug".to_string()],
                    user_states: vec!["Beginner".to_string()],
                    task_health: None,
                }),
        );

    // Test 1: No context
    let prompt1 = expertise.to_prompt_with_context(&RenderContext::new());
    assert!(prompt1.contains("Always visible"));
    assert!(!prompt1.contains("Debug only"));
    assert!(!prompt1.contains("Beginner only"));
    assert!(!prompt1.contains("At risk only"));

    // Test 2: Debug context
    let prompt2 = expertise.to_prompt_with_context(&RenderContext::new().with_task_type("Debug"));
    assert!(prompt2.contains("Always visible"));
    assert!(prompt2.contains("Debug only"));
    assert!(!prompt2.contains("Beginner only"));
    assert!(!prompt2.contains("Debug + Beginner"));

    // Test 3: Debug + Beginner
    let prompt3 = expertise.to_prompt_with_context(
        &RenderContext::new()
            .with_task_type("Debug")
            .with_user_state("Beginner"),
    );
    assert!(prompt3.contains("Always visible"));
    assert!(prompt3.contains("Debug only"));
    assert!(prompt3.contains("Beginner only"));
    assert!(prompt3.contains("Debug + Beginner"));

    // Test 4: At risk health
    let prompt4 = expertise
        .to_prompt_with_context(&RenderContext::new().with_task_health(TaskHealth::AtRisk));
    assert!(prompt4.contains("Always visible"));
    assert!(prompt4.contains("At risk only"));
    assert!(!prompt4.contains("Debug only"));
}

#[test]
fn test_priority_ordering_in_prompt() {
    let expertise = Expertise::new("priority-test", "1.0.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Fragment 1: Low".to_string()))
                .with_priority(Priority::Low),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Fragment 2: Critical".to_string()))
                .with_priority(Priority::Critical),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Fragment 3: Normal".to_string()))
                .with_priority(Priority::Normal),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Fragment 4: High".to_string()))
                .with_priority(Priority::High),
        );

    let prompt = expertise.to_prompt();

    // Find positions
    let critical_pos = prompt.find("Fragment 2: Critical").unwrap();
    let high_pos = prompt.find("Fragment 4: High").unwrap();
    let normal_pos = prompt.find("Fragment 3: Normal").unwrap();
    let low_pos = prompt.find("Fragment 1: Low").unwrap();

    // Verify ordering
    assert!(critical_pos < high_pos);
    assert!(high_pos < normal_pos);
    assert!(normal_pos < low_pos);
}

#[test]
fn test_visualization_outputs() {
    let expertise = Expertise::new("viz-test", "1.0.0")
        .with_tag("visualization")
        .with_fragment(WeightedFragment::new(KnowledgeFragment::Text(
            "Test content".to_string(),
        )));

    // Test tree output
    let tree = expertise.to_tree();
    assert!(tree.contains("Expertise: viz-test"));
    assert!(tree.contains("visualization"));
    assert!(tree.contains("Test content"));

    // Test Mermaid output
    let mermaid = expertise.to_mermaid();
    assert!(mermaid.contains("graph TD"));
    assert!(mermaid.contains("ROOT"));
    assert!(mermaid.contains("viz-test"));
    assert!(mermaid.contains("Test content"));
    assert!(mermaid.contains("classDef"));
}

#[test]
fn test_json_schema_generation() {
    use schemars::schema_for;

    let schema = schema_for!(Expertise);
    let schema_value = serde_json::to_value(&schema).expect("Failed to serialize schema");
    assert!(schema_value.is_object());

    let schema_obj = schema_value.as_object().unwrap();
    assert!(schema_obj.contains_key("$schema"));
    assert!(schema_obj.contains_key("title"));
    assert!(schema_obj.contains_key("definitions") || schema_obj.contains_key("$defs"));
}

#[test]
fn test_builder_pattern() {
    let expertise = Expertise::new("builder-test", "1.0.0")
        .with_tag("tag1")
        .with_tag("tag2")
        .with_tags(vec!["tag3".to_string(), "tag4".to_string()])
        .with_fragment(WeightedFragment::new(KnowledgeFragment::Text(
            "Fragment 1".to_string(),
        )))
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Fragment 2".to_string()))
                .with_priority(Priority::High)
                .with_context(ContextProfile::Conditional {
                    task_types: vec!["Test".to_string()],
                    user_states: vec![],
                    task_health: None,
                }),
        );

    assert_eq!(expertise.id, "builder-test");
    assert_eq!(expertise.version, "1.0.0");
    assert_eq!(expertise.tags.len(), 4);
    assert_eq!(expertise.content.len(), 2);
    assert_eq!(expertise.content[1].priority, Priority::High);
}

#[test]
fn test_task_health_behaviors() {
    let expertise = Expertise::new("health-test", "1.0.0")
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Go fast".to_string())).with_context(
                ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec![],
                    task_health: Some(TaskHealth::OnTrack),
                },
            ),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Be careful".to_string())).with_context(
                ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec![],
                    task_health: Some(TaskHealth::AtRisk),
                },
            ),
        )
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text("Stop and reassess".to_string()))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec![],
                    task_health: Some(TaskHealth::OffTrack),
                }),
        );

    // On track
    let on_track_prompt = expertise
        .to_prompt_with_context(&RenderContext::new().with_task_health(TaskHealth::OnTrack));
    assert!(on_track_prompt.contains("Go fast"));
    assert!(!on_track_prompt.contains("Be careful"));
    assert!(!on_track_prompt.contains("Stop and reassess"));

    // At risk
    let at_risk_prompt = expertise
        .to_prompt_with_context(&RenderContext::new().with_task_health(TaskHealth::AtRisk));
    assert!(!at_risk_prompt.contains("Go fast"));
    assert!(at_risk_prompt.contains("Be careful"));
    assert!(!at_risk_prompt.contains("Stop and reassess"));

    // Off track
    let off_track_prompt = expertise
        .to_prompt_with_context(&RenderContext::new().with_task_health(TaskHealth::OffTrack));
    assert!(!off_track_prompt.contains("Go fast"));
    assert!(!off_track_prompt.contains("Be careful"));
    assert!(off_track_prompt.contains("Stop and reassess"));
}
