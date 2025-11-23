//! E2E Integration Tests for ExpertiseAgent with Context-Aware Rendering
//!
//! This test suite verifies the complete context flow:
//! Payload (with RenderContext) → ExpertiseAgent → Context-aware expertise rendering → Agent execution

#[cfg(feature = "agent")]
mod expertise_agent_integration_tests {
    use llm_toolkit::agent::expertise::{
        Expertise, KnowledgeFragment, RenderContext, WeightedFragment,
    };
    use llm_toolkit::agent::{Agent, ExpertiseAgent, Payload};
    use llm_toolkit::context::{ContextProfile, Priority, TaskHealth};

    // Helper function to extract all content from payload (both text and messages)
    fn get_all_payload_content(payload: &Payload) -> String {
        let mut content = String::new();

        // Get messages
        for msg in payload.to_messages() {
            content.push_str(&msg.content);
            content.push('\n');
        }

        // Get text
        content.push_str(&payload.to_text());

        content
    }

    // Mock agent that captures the payload it receives
    struct PayloadCapturingAgent {
        captured_payload: std::sync::Arc<std::sync::Mutex<Option<Payload>>>,
    }

    impl PayloadCapturingAgent {
        fn new() -> Self {
            Self {
                captured_payload: std::sync::Arc::new(std::sync::Mutex::new(None)),
            }
        }

        fn get_captured_payload(&self) -> Option<Payload> {
            self.captured_payload.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Agent for PayloadCapturingAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            &"Payload Capturing Agent"
        }

        async fn execute(
            &self,
            intent: Payload,
        ) -> Result<Self::Output, llm_toolkit::agent::AgentError> {
            *self.captured_payload.lock().unwrap() = Some(intent.clone());
            Ok("Captured".to_string())
        }

        fn name(&self) -> String {
            "PayloadCapturingAgent".to_string()
        }

        async fn is_available(&self) -> Result<(), llm_toolkit::agent::AgentError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_expertise_agent_e2e_context_aware_rendering() {
        // Step 1: Create Expertise with conditional fragments
        let expertise = Expertise::new("security-reviewer", "1.0")
            .with_description("Security code reviewer with context-aware guidance")
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "Always verify input validation and authentication.".to_string(),
                ))
                .with_priority(Priority::Critical)
                .with_context(ContextProfile::Always),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "SLOW DOWN. Extra vigilance required for at-risk tasks.".to_string(),
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
                    "Focus on SQL injection and XSS vulnerabilities.".to_string(),
                ))
                .with_priority(Priority::High)
                .with_context(ContextProfile::Conditional {
                    task_types: vec!["security-review".to_string()],
                    user_states: vec![],
                    task_health: None,
                }),
            );

        // Step 2: Wrap capturing agent with ExpertiseAgent
        let capturing_agent = PayloadCapturingAgent::new();
        let agent = ExpertiseAgent::new(capturing_agent, expertise.clone());

        // Step 3: Execute with Payload containing RenderContext
        let payload = Payload::text("Review this authentication code")
            .with_task_type("security-review")
            .with_task_health(TaskHealth::AtRisk);

        let result = agent.execute(payload.clone()).await.unwrap();
        assert_eq!(result, "Captured");

        // Step 4: Verify context-aware expertise was applied
        let captured = agent.inner().get_captured_payload().unwrap();
        let captured_text = get_all_payload_content(&captured);

        // Should include the Always fragment
        assert!(
            captured_text.contains("Always verify input validation and authentication"),
            "Expected Always fragment to be included"
        );

        // Should include the AtRisk fragment (task_health matches)
        assert!(
            captured_text.contains("SLOW DOWN. Extra vigilance required"),
            "Expected AtRisk fragment to be included when task_health is AtRisk"
        );

        // Should include the security-review fragment (task_type matches)
        assert!(
            captured_text.contains("SQL injection and XSS vulnerabilities"),
            "Expected security-review fragment to be included when task_type matches"
        );

        // Should also contain original user question
        assert!(
            captured_text.contains("Review this authentication code"),
            "Expected original user question to be preserved"
        );
    }

    #[tokio::test]
    async fn test_expertise_agent_e2e_selective_rendering() {
        // Create Expertise with conditional fragments
        let expertise = Expertise::new("code-reviewer", "1.0")
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "Base guidance for all reviews.".to_string(),
                ))
                .with_context(ContextProfile::Always),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "Beginner-friendly explanations.".to_string(),
                ))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec!["beginner".to_string()],
                    task_health: None,
                }),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "Advanced optimization techniques.".to_string(),
                ))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec!["expert".to_string()],
                    task_health: None,
                }),
            );

        // Test 1: Beginner context
        let capturing_agent = PayloadCapturingAgent::new();
        let agent = ExpertiseAgent::new(capturing_agent, expertise.clone());

        let payload = Payload::text("Review this code").with_user_state("beginner");

        agent.execute(payload).await.unwrap();

        let captured = agent.inner().get_captured_payload().unwrap();
        let captured_text = get_all_payload_content(&captured);

        assert!(
            captured_text.contains("Base guidance for all reviews"),
            "Expected base guidance"
        );
        assert!(
            captured_text.contains("Beginner-friendly explanations"),
            "Expected beginner fragment"
        );
        assert!(
            !captured_text.contains("Advanced optimization techniques"),
            "Should NOT include expert fragment"
        );

        // Test 2: Expert context
        let capturing_agent2 = PayloadCapturingAgent::new();
        let agent2 = ExpertiseAgent::new(capturing_agent2, expertise.clone());

        let payload2 = Payload::text("Review this code").with_user_state("expert");

        agent2.execute(payload2).await.unwrap();

        let captured2 = agent2.inner().get_captured_payload().unwrap();
        let captured_text2 = get_all_payload_content(&captured2);

        assert!(
            captured_text2.contains("Base guidance for all reviews"),
            "Expected base guidance"
        );
        assert!(
            !captured_text2.contains("Beginner-friendly explanations"),
            "Should NOT include beginner fragment"
        );
        assert!(
            captured_text2.contains("Advanced optimization techniques"),
            "Expected expert fragment"
        );
    }

    #[tokio::test]
    async fn test_expertise_agent_e2e_no_context() {
        // Create Expertise with conditional fragments
        let expertise = Expertise::new("reviewer", "1.0").with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                "Special guidance for security reviews.".to_string(),
            ))
            .with_context(ContextProfile::Conditional {
                task_types: vec!["security-review".to_string()],
                user_states: vec![],
                task_health: None,
            }),
        );

        let capturing_agent = PayloadCapturingAgent::new();
        let agent = ExpertiseAgent::new(capturing_agent, expertise);

        // Execute without any render context
        let payload = Payload::text("Review this code");

        agent.execute(payload).await.unwrap();

        let captured = agent.inner().get_captured_payload().unwrap();
        let captured_text = get_all_payload_content(&captured);

        // Should NOT include conditional fragment when context doesn't match
        assert!(
            !captured_text.contains("Special guidance for security reviews"),
            "Should NOT include conditional fragment when context doesn't match"
        );

        // Should still contain original user question
        assert!(
            captured_text.contains("Review this code"),
            "Expected original user question to be preserved"
        );
    }

    #[tokio::test]
    async fn test_expertise_agent_e2e_multiple_user_states() {
        let expertise = Expertise::new("educator", "1.0")
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "Extra patience for confused learners.".to_string(),
                ))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec!["confused".to_string()],
                    task_health: None,
                }),
            )
            .with_fragment(
                WeightedFragment::new(KnowledgeFragment::Text(
                    "Simplified explanations for beginners.".to_string(),
                ))
                .with_context(ContextProfile::Conditional {
                    task_types: vec![],
                    user_states: vec!["beginner".to_string()],
                    task_health: None,
                }),
            );

        let capturing_agent = PayloadCapturingAgent::new();
        let agent = ExpertiseAgent::new(capturing_agent, expertise);

        // Payload with multiple user states
        let payload = Payload::text("Explain this concept")
            .with_user_state("beginner")
            .with_user_state("confused");

        agent.execute(payload).await.unwrap();

        let captured = agent.inner().get_captured_payload().unwrap();
        let captured_text = get_all_payload_content(&captured);

        // Should include both fragments when both user states are present
        assert!(
            captured_text.contains("Extra patience for confused learners"),
            "Expected confused learner fragment"
        );
        assert!(
            captured_text.contains("Simplified explanations for beginners"),
            "Expected beginner fragment"
        );
    }

    #[tokio::test]
    async fn test_expertise_agent_e2e_render_context_preservation() {
        let expertise = Expertise::new("test", "1.0");

        let capturing_agent = PayloadCapturingAgent::new();
        let agent = ExpertiseAgent::new(capturing_agent, expertise);

        // Original payload with render context
        let original_context = RenderContext::new()
            .with_task_type("review")
            .with_task_health(TaskHealth::OnTrack);

        let payload = Payload::text("Test").with_render_context(original_context.clone());

        agent.execute(payload).await.unwrap();

        let captured = agent.inner().get_captured_payload().unwrap();

        // RenderContext should be preserved in the enriched payload
        assert!(captured.render_context().is_some());
        let captured_context = captured.render_context().unwrap();
        assert_eq!(captured_context.task_type, Some("review".to_string()));
        assert_eq!(captured_context.task_health, Some(TaskHealth::OnTrack));
    }
}
