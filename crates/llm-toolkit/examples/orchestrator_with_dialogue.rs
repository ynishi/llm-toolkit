//! Example: Using Dialogue as an Agent in Orchestrator
//!
//! This example demonstrates how to use Dialogue (multi-agent conversations)
//! as agents within an Orchestrator workflow, enabling flexible composition
//! patterns where teams of agents work together on complex tasks.
//!
//! Architecture:
//! ```text
//! Orchestrator
//!   ├─ Step 1: Design Team (Dialogue)
//!   │   ├─ Designer
//!   │   └─ UX Researcher
//!   ├─ Step 2: Engineering Team (Dialogue)
//!   │   ├─ Backend Engineer
//!   │   └─ Frontend Engineer
//!   └─ Step 3: Regular Agent
//! ```

use async_trait::async_trait;
use llm_toolkit::agent::dialogue::{Dialogue, SequentialOrder};
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentAdapter, AgentError, Payload};
use llm_toolkit::orchestrator::{
    BlueprintWorkflow, ParallelOrchestrator, StrategyMap, StrategyStep,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Mock agent for demonstration
#[derive(Clone)]
struct MockAgent {
    name: String,
    response: String,
}

impl MockAgent {
    fn new(name: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            response: response.into(),
        }
    }
}

#[async_trait]
impl Agent for MockAgent {
    type Output = String;
    type Expertise = &'static str;

    fn expertise(&self) -> &&'static str {
        const EXPERTISE: &'static str = "Mock agent for demonstration";
        &EXPERTISE
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.response.clone())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Orchestrator with Dialogue Teams Example ===\n");

    // ========================================================================
    // Step 1: Create Design Team (Dialogue)
    // ========================================================================
    println!("Creating Design Team...");

    let designer_persona = Persona {
        name: "Sarah".to_string(),
        role: "UX Designer".to_string(),
        background: "10 years of user experience design".to_string(),
        communication_style: "User-focused and empathetic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let researcher_persona = Persona {
        name: "Mike".to_string(),
        role: "UX Researcher".to_string(),
        background: "Expert in user research and usability testing".to_string(),
        communication_style: "Data-driven and methodical".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut design_team = Dialogue::sequential_with_order(SequentialOrder::Explicit(vec![
        "Sarah".to_string(),
        "Mike".to_string(),
    ]));

    design_team.add_participant(
        designer_persona,
        MockAgent::new(
            "Sarah",
            "Design proposal: Clean, minimal interface with focus on key actions",
        ),
    );
    design_team.add_participant(
        researcher_persona,
        MockAgent::new(
            "Mike",
            "Research validation: User testing shows 85% preference for this approach",
        ),
    );

    println!("  ✓ Design Team: 2 participants (Sequential: Sarah → Mike)");

    // ========================================================================
    // Step 2: Create Engineering Team (Dialogue)
    // ========================================================================
    println!("Creating Engineering Team...");

    let backend_persona = Persona {
        name: "Alex".to_string(),
        role: "Backend Engineer".to_string(),
        background: "Expert in API design and scalable systems".to_string(),
        communication_style: "Technical and systematic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let frontend_persona = Persona {
        name: "Emma".to_string(),
        role: "Frontend Engineer".to_string(),
        background: "React specialist with strong UX implementation skills".to_string(),
        communication_style: "Pragmatic and detail-oriented".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut engineering_team = Dialogue::broadcast();

    engineering_team.add_participant(
        backend_persona,
        MockAgent::new(
            "Alex",
            "Backend assessment: API endpoints ready, 2 days for implementation",
        ),
    );
    engineering_team.add_participant(
        frontend_persona,
        MockAgent::new(
            "Emma",
            "Frontend assessment: Component library supports this, 3 days needed",
        ),
    );

    println!("  ✓ Engineering Team: 2 participants (Broadcast)");

    // ========================================================================
    // Step 3: Create Orchestrator with Dialogue Teams
    // ========================================================================
    println!("\nSetting up Orchestrator...");

    let blueprint = BlueprintWorkflow::new(
        "Product feature development workflow with team dialogues".to_string(),
    );
    let mut orchestrator = ParallelOrchestrator::new(blueprint);

    // Register dialogue teams as agents
    orchestrator.add_agent("design_team", Arc::new(AgentAdapter::new(design_team)));
    orchestrator.add_agent(
        "engineering_team",
        Arc::new(AgentAdapter::new(engineering_team)),
    );

    // Add a regular agent for documentation
    orchestrator.add_agent(
        "documentation",
        Arc::new(AgentAdapter::new(MockAgent::new(
            "DocWriter",
            "Documentation: Feature spec written and reviewed",
        ))),
    );

    println!("  ✓ Registered 2 dialogue teams + 1 regular agent");

    // ========================================================================
    // Step 4: Define Strategy
    // ========================================================================
    println!("\nDefining execution strategy...");

    let mut strategy = StrategyMap::new("Feature development workflow".to_string());

    strategy.add_step(StrategyStep::new(
        "design".to_string(),
        "Design team creates UI/UX proposal".to_string(),
        "design_team".to_string(),
        "Design the user onboarding flow".to_string(),
        "UI/UX proposal with research validation".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "engineering".to_string(),
        "Engineering team assesses technical feasibility".to_string(),
        "engineering_team".to_string(),
        "Assess the feasibility of: {{ step_design_output }}".to_string(),
        "Technical assessment from backend and frontend".to_string(),
    ));

    strategy.add_step(StrategyStep::new(
        "documentation".to_string(),
        "Document the feature specification".to_string(),
        "documentation".to_string(),
        "Document based on: {{ step_design_output }} and {{ step_engineering_output }}".to_string(),
        "Feature documentation".to_string(),
    ));

    orchestrator.set_strategy_map(strategy);

    println!("  ✓ Strategy: Design Team → Engineering Team → Documentation");

    // ========================================================================
    // Step 5: Execute Orchestrator
    // ========================================================================
    println!("\n=== Executing Orchestrator ===\n");

    let result = orchestrator
        .execute(
            "Build a streamlined user onboarding experience",
            CancellationToken::new(),
            None,
            None,
        )
        .await?;

    println!("\n=== Execution Results ===");
    println!(
        "Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("Steps executed: {}", result.steps_executed);
    println!("Steps skipped: {}", result.steps_skipped);

    // ========================================================================
    // Step 6: Display Outputs
    // ========================================================================
    println!("\n=== Step Outputs ===\n");

    if let Some(design_output) = result.context.get("step_design_output") {
        println!("Design Team Output:");
        if let Some(turns) = design_output.as_array() {
            for turn in turns {
                if let Some(speaker) = turn.get("speaker").and_then(|s| s.get("name"))
                    && let Some(content) = turn.get("content").and_then(|c| c.as_str())
                {
                    println!("  [{}]: {}", speaker, content);
                }
            }
        }
        println!();
    }

    if let Some(eng_output) = result.context.get("step_engineering_output") {
        println!("Engineering Team Output:");
        if let Some(turns) = eng_output.as_array() {
            for turn in turns {
                if let Some(speaker) = turn.get("speaker").and_then(|s| s.get("name"))
                    && let Some(content) = turn.get("content").and_then(|c| c.as_str())
                {
                    println!("  [{}]: {}", speaker, content);
                }
            }
        }
        println!();
    }

    if let Some(doc_output) = result.context.get("step_documentation_output") {
        println!("Documentation Output:");
        if let Some(doc_str) = doc_output.as_str() {
            println!("  {}", doc_str);
        }
        println!();
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("=== Key Takeaways ===");
    println!("✓ Dialogue can be used as an Agent in Orchestrator workflows");
    println!("✓ Multiple dialogue teams can collaborate in a single orchestration");
    println!("✓ Each dialogue team can have its own execution model:");
    println!("  - Design Team: Sequential (ordered discussion)");
    println!("  - Engineering Team: Broadcast (parallel assessment)");
    println!("✓ Dialogue outputs are properly passed to subsequent steps");
    println!("\nThis enables complex, hierarchical multi-agent workflows!");

    Ok(())
}
