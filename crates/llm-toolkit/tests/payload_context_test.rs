//! Tests for Payload RenderContext integration (Step 1/5 of Context Integration)

#[cfg(feature = "agent")]
mod render_context_tests {
    use llm_toolkit::agent::Payload;
    use llm_toolkit::agent::expertise::RenderContext;
    use llm_toolkit::context::TaskHealth;

    #[test]
    fn test_payload_with_render_context() {
        let context = RenderContext::new()
            .with_task_type("security-review")
            .with_task_health(TaskHealth::AtRisk);

        let payload = Payload::text("Review this code").with_render_context(context.clone());

        assert!(payload.render_context().is_some());
        let extracted = payload.render_context().unwrap();
        assert_eq!(extracted.task_type, Some("security-review".to_string()));
        assert_eq!(extracted.task_health, Some(TaskHealth::AtRisk));
    }

    #[test]
    fn test_payload_with_task_type() {
        let payload = Payload::text("Review this code").with_task_type("security-review");

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.task_type, Some("security-review".to_string()));
        assert_eq!(context.user_states.len(), 0);
        assert_eq!(context.task_health, None);
    }

    #[test]
    fn test_payload_with_user_state() {
        let payload = Payload::text("Explain this concept").with_user_state("beginner");

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.user_states, vec!["beginner"]);
        assert_eq!(context.task_type, None);
        assert_eq!(context.task_health, None);
    }

    #[test]
    fn test_payload_with_task_health() {
        let payload = Payload::text("Debug this issue").with_task_health(TaskHealth::AtRisk);

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.task_health, Some(TaskHealth::AtRisk));
        assert_eq!(context.task_type, None);
        assert_eq!(context.user_states.len(), 0);
    }

    #[test]
    fn test_payload_builder_chain() {
        let payload = Payload::text("Review this code")
            .with_task_type("security-review")
            .with_user_state("beginner")
            .with_task_health(TaskHealth::AtRisk);

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.task_type, Some("security-review".to_string()));
        assert_eq!(context.user_states, vec!["beginner"]);
        assert_eq!(context.task_health, Some(TaskHealth::AtRisk));
    }

    #[test]
    fn test_payload_multiple_user_states() {
        let payload = Payload::text("Question")
            .with_user_state("beginner")
            .with_user_state("confused");

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.user_states, vec!["beginner", "confused"]);
    }

    #[test]
    fn test_payload_render_context_preserved_with_text() {
        let payload = Payload::text("First text")
            .with_task_type("security-review")
            .with_text("Second text");

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.task_type, Some("security-review".to_string()));
    }

    #[test]
    fn test_payload_render_context_preserved_with_prepend() {
        let payload = Payload::text("User question")
            .with_task_health(TaskHealth::AtRisk)
            .prepend_text("System instruction");

        assert!(payload.render_context().is_some());
        let context = payload.render_context().unwrap();
        assert_eq!(context.task_health, Some(TaskHealth::AtRisk));
    }

    #[test]
    fn test_payload_render_context_none_by_default() {
        let payload = Payload::text("Simple text");
        assert!(payload.render_context().is_none());
    }

    #[test]
    fn test_payload_from_messages_has_no_render_context() {
        use llm_toolkit::agent::payload_message::PayloadMessage;

        let payload = Payload::from_messages(vec![PayloadMessage::system("Test message")]);
        assert!(payload.render_context().is_none());
    }

    #[test]
    fn test_payload_merge_preserves_first_render_context() {
        let payload1 = Payload::text("First").with_task_type("review");
        let payload2 = Payload::text("Second").with_task_type("debug");

        let merged = payload1.merge(payload2);

        // Should preserve payload1's context
        assert!(merged.render_context().is_some());
        let context = merged.render_context().unwrap();
        assert_eq!(context.task_type, Some("review".to_string()));
    }

    #[test]
    fn test_payload_render_context_separation_from_context() {
        let payload = Payload::text("Question")
            .with_context("LLM-visible context string")
            .with_task_type("security-review");

        // Both contexts should be present but separate
        assert_eq!(payload.contexts(), vec!["LLM-visible context string"]);
        assert!(payload.render_context().is_some());
        assert_eq!(
            payload.render_context().unwrap().task_type,
            Some("security-review".to_string())
        );
    }

    #[test]
    fn test_payload_clone_preserves_render_context() {
        let payload = Payload::text("Original")
            .with_task_type("review")
            .with_task_health(TaskHealth::OnTrack);

        let cloned = payload.clone();

        // Arc ensures both share same render_context
        assert!(cloned.render_context().is_some());
        let context = cloned.render_context().unwrap();
        assert_eq!(context.task_type, Some("review".to_string()));
        assert_eq!(context.task_health, Some(TaskHealth::OnTrack));
    }

    #[test]
    fn test_payload_render_context_backward_compatibility() {
        // Old code without render_context should still work
        let payload = Payload::text("Simple question");
        assert_eq!(payload.to_text(), "Simple question");
        assert!(payload.render_context().is_none());
    }
}
