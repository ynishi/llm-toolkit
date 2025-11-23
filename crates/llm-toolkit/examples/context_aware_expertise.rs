//! # Context-Aware Expertise Example
//!
//! This example demonstrates the complete context integration workflow in llm-toolkit v0.57.0:
//!
//! 1. **Payload Enhancement**: Creating payloads with RenderContext metadata
//! 2. **Expertise with Conditional Fragments**: Defining context-aware expertise
//! 3. **ExpertiseAgent**: Automatic context-aware prompt rendering
//! 4. **E2E Flow**: Complete workflow from user input to agent execution
//!
//! ## Key Concepts
//!
//! - **RenderContext**: Runtime metadata (task_type, user_states, task_health) for expertise rendering
//! - **Conditional Fragments**: Expertise knowledge that only applies in specific contexts
//! - **ExpertiseAgent**: Wrapper that automatically extracts context and applies expertise
//! - **Zero-Friction Context Flow**: No manual context management required

#[cfg(feature = "agent")]
fn main() {
    use llm_toolkit::agent::expertise::{Expertise, KnowledgeFragment, WeightedFragment};
    use llm_toolkit::agent::{Agent, ExpertiseAgent, Payload};
    use llm_toolkit::context::{ContextProfile, Priority, TaskHealth};

    println!("=== Context-Aware Expertise Example ===\n");

    // ============================================================================
    // Part 1: Define Expertise with Conditional Fragments
    // ============================================================================

    println!("📚 Part 1: Defining Expertise with Conditional Fragments\n");

    let code_reviewer_expertise = Expertise::new("code-reviewer", "1.0")
        .with_description("AI Code Reviewer with context-aware guidance")
        // Fragment 1: Always visible - base code review principles
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                r#"# Code Review Principles
- Verify code compiles and all tests pass
- Check for proper error handling
- Ensure code follows project conventions
"#
                .to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Always),
        )
        // Fragment 2: Security-specific guidance (conditional on task_type)
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                r#"# Security Review Checklist
- Verify all inputs are validated and sanitized
- Check for SQL injection vulnerabilities
- Ensure proper authentication and authorization
- Look for XSS attack vectors
- Verify secure password storage (bcrypt, Argon2)
"#
                .to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec!["security-review".to_string()],
                user_states: vec![],
                task_health: None,
            }),
        )
        // Fragment 3: Extra vigilance for at-risk tasks (conditional on task_health)
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                r#"⚠️ SLOW DOWN - EXTRA VIGILANCE REQUIRED ⚠️

This task is marked as AT RISK. Apply maximum scrutiny:
- Double-check all logic carefully
- Verify edge cases are handled
- Ensure no regression in existing functionality
- Consider requesting additional review
"#
                .to_string(),
            ))
            .with_priority(Priority::Critical)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec![],
                task_health: Some(TaskHealth::AtRisk),
            }),
        )
        // Fragment 4: Beginner-friendly explanations (conditional on user_state)
        .with_fragment(
            WeightedFragment::new(KnowledgeFragment::Text(
                r#"# Beginner-Friendly Guidance
- Explain technical terms clearly
- Provide step-by-step reasoning
- Include code examples where helpful
- Be patient and encouraging
"#
                .to_string(),
            ))
            .with_priority(Priority::High)
            .with_context(ContextProfile::Conditional {
                task_types: vec![],
                user_states: vec!["beginner".to_string()],
                task_health: None,
            }),
        );

    println!("✅ Defined expertise with 4 conditional fragments:");
    println!("   1. Always visible: Base code review principles");
    println!("   2. task_type='security-review': Security checklist");
    println!("   3. task_health=AtRisk: Extra vigilance warning");
    println!("   4. user_state='beginner': Simplified explanations");
    println!();

    // ============================================================================
    // Part 2: Create Payloads with RenderContext
    // ============================================================================

    println!("📦 Part 2: Creating Payloads with RenderContext\n");

    // Scenario 1: Regular code review
    let _regular_review =
        Payload::text("Review this authentication middleware").with_task_type("code-review");

    println!("Scenario 1: Regular code review");
    println!("  Task Type: code-review");
    println!("  Expected Fragments: Base principles only");
    println!();

    // Scenario 2: Security review for at-risk task
    let _security_review_at_risk = Payload::text("Review this password hashing implementation")
        .with_task_type("security-review")
        .with_task_health(TaskHealth::AtRisk);

    println!("Scenario 2: Security review (at-risk)");
    println!("  Task Type: security-review");
    println!("  Task Health: AtRisk");
    println!("  Expected Fragments: Base + Security + Extra vigilance");
    println!();

    // Scenario 3: Beginner asking for code review
    let _beginner_review = Payload::text("Can you review my first API endpoint?")
        .with_user_state("beginner")
        .with_task_type("code-review");

    println!("Scenario 3: Beginner code review");
    println!("  User State: beginner");
    println!("  Task Type: code-review");
    println!("  Expected Fragments: Base + Beginner guidance");
    println!();

    // ============================================================================
    // Part 3: ExpertiseAgent - Automatic Context Application
    // ============================================================================

    println!("🤖 Part 3: ExpertiseAgent - Automatic Context Application\n");

    // Define a simple mock agent for demonstration
    struct DemoAgent;

    #[async_trait::async_trait]
    impl Agent for DemoAgent {
        type Output = String;
        type Expertise = &'static str;

        fn expertise(&self) -> &&'static str {
            &"Demo Agent"
        }

        async fn execute(
            &self,
            intent: Payload,
        ) -> Result<Self::Output, llm_toolkit::agent::AgentError> {
            // In a real implementation, this would call an LLM
            // Here we just show what the agent received
            let messages = intent.to_messages();
            let text = intent.to_text();

            let mut response = String::from("=== Agent Received ===\n\n");

            if !messages.is_empty() {
                response.push_str("System Messages:\n");
                for msg in messages {
                    response.push_str(&format!("  [{:?}] {}\n", msg.speaker, msg.content));
                }
                response.push('\n');
            }

            response.push_str(&format!("User Question:\n  {}\n", text));

            Ok(response)
        }

        fn name(&self) -> String {
            "DemoAgent".to_string()
        }

        async fn is_available(&self) -> Result<(), llm_toolkit::agent::AgentError> {
            Ok(())
        }
    }

    // Wrap with ExpertiseAgent
    let _expertise_agent = ExpertiseAgent::new(DemoAgent, code_reviewer_expertise);

    println!("✅ Created ExpertiseAgent wrapper");
    println!("   Inner Agent: DemoAgent");
    println!("   Expertise: code-reviewer v1.0");
    println!();

    // ============================================================================
    // Part 4: Execute and See Context-Aware Results
    // ============================================================================

    println!("🚀 Part 4: Execute and See Context-Aware Results\n");
    println!("Note: This is a demonstration. In practice, use tokio::runtime.\n");

    // For demonstration purposes, show what would happen
    println!("Scenario 1 Execution (Regular Review):");
    println!("  Payload: 'Review this authentication middleware'");
    println!("  Context: task_type='code-review'");
    println!("  Expected: Base code review principles only\n");

    println!("Scenario 2 Execution (Security Review - At Risk):");
    println!("  Payload: 'Review this password hashing implementation'");
    println!("  Context: task_type='security-review', task_health=AtRisk");
    println!("  Expected: Base + Security checklist + Extra vigilance warning\n");

    println!("Scenario 3 Execution (Beginner):");
    println!("  Payload: 'Can you review my first API endpoint?'");
    println!("  Context: user_state='beginner', task_type='code-review'");
    println!("  Expected: Base + Beginner-friendly explanations\n");

    // ============================================================================
    // Part 5: Builder Pattern API
    // ============================================================================

    println!("🛠️  Part 5: Builder Pattern API Examples\n");

    println!("Example 1: Chaining context methods");
    println!(
        r#"  let payload = Payload::text("Review this code")
      .with_task_type("security-review")
      .with_user_state("beginner")
      .with_task_health(TaskHealth::AtRisk);"#
    );
    println!();

    println!("Example 2: Multiple user states");
    println!(
        r#"  let payload = Payload::text("Explain this error")
      .with_user_state("beginner")
      .with_user_state("confused");"#
    );
    println!();

    println!("Example 3: Direct RenderContext");
    println!(
        r#"  use llm_toolkit::agent::expertise::RenderContext;

  let context = RenderContext::new()
      .with_task_type("performance-review")
      .with_task_health(TaskHealth::OnTrack);

  let payload = Payload::text("Optimize this query")
      .with_render_context(context);"#
    );
    println!();

    // ============================================================================
    // Part 6: Context Separation
    // ============================================================================

    println!("🔄 Part 6: Context Separation - LLM Context vs RenderContext\n");

    println!("RenderContext (structured metadata for expertise rendering):");
    println!(r#"  payload.with_task_type("security-review")  // Invisible to LLM"#);
    println!();

    println!("LLM Context (natural language visible to LLM):");
    println!(
        r#"  payload.with_context("This code handles payment processing")  // Visible to LLM"#
    );
    println!();

    println!("Both can coexist independently:");
    println!(
        r#"  let payload = Payload::text("Review this code")
      .with_context("This handles user authentication")  // LLM sees this
      .with_task_type("security-review");  // Controls expertise rendering"#
    );
    println!();

    // ============================================================================
    // Summary
    // ============================================================================

    println!("{}", "=".repeat(60));
    println!("\n✨ Summary: Benefits of Context-Aware Expertise\n");
    println!("1. Zero Manual Context Management");
    println!("   - Context flows automatically from Payload → ExpertiseAgent");
    println!("   - No need to manually extract or apply context\n");

    println!("2. Precise Expertise Rendering");
    println!("   - Only relevant knowledge fragments are included");
    println!("   - Reduces prompt bloat and improves focus\n");

    println!("3. Flexible Context Dimensions");
    println!("   - task_type: What kind of task (security-review, etc.)");
    println!("   - user_states: User characteristics (beginner, expert, etc.)");
    println!("   - task_health: Task status (AtRisk, OnTrack, etc.)\n");

    println!("4. Backward Compatible");
    println!("   - Payloads without render_context work normally");
    println!("   - Expertise without conditional fragments still work\n");

    println!("5. Type-Safe and Composable");
    println!("   - Builder pattern with fluent API");
    println!("   - ExpertiseAgent composable with any Agent implementation\n");

    println!("{}", "=".repeat(60));
}

#[cfg(not(feature = "agent"))]
fn main() {
    println!("This example requires the 'agent' feature.");
    println!("Run with: cargo run --example context_aware_expertise --features agent");
}
