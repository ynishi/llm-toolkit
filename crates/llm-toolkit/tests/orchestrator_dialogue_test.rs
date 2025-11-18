//! Tests for using Dialogue as an Agent in Orchestrator workflows.
//!
//! This test suite verifies that Dialogue can be registered and executed
//! as a regular Agent in orchestration scenarios, enabling bidirectional
//! composition patterns.

use async_trait::async_trait;
use llm_toolkit::agent::dialogue::Dialogue;
use llm_toolkit::agent::persona::Persona;
use llm_toolkit::agent::{Agent, AgentError, Payload};
use llm_toolkit::orchestrator::{ParallelOrchestrator, StrategyMap, StrategyStep};
use tokio_util::sync::CancellationToken;

/// Mock agent for testing
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

    fn expertise(&self) -> &str {
        "Mock agent for testing"
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&self, _payload: Payload) -> Result<Self::Output, AgentError> {
        Ok(self.response.clone())
    }
}

#[tokio::test]
async fn test_orchestrator_with_dialogue_agent() {
    // Create a dialogue with mock participants
    let persona1 = Persona {
        name: "Alice".to_string(),
        role: "Engineer".to_string(),
        background: "Senior developer".to_string(),
        communication_style: "Technical".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let persona2 = Persona {
        name: "Bob".to_string(),
        role: "Designer".to_string(),
        background: "UX specialist".to_string(),
        communication_style: "Creative".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let agent1 = MockAgent::new("Alice", "Technical analysis: looks good");
    let agent2 = MockAgent::new("Bob", "Design perspective: needs refinement");

    let mut dialogue = Dialogue::broadcast();
    dialogue.add_participant(persona1, agent1);
    dialogue.add_participant(persona2, agent2);

    // Create orchestrator and register dialogue as an agent
    let blueprint = llm_toolkit::orchestrator::BlueprintWorkflow::new(
        "Test workflow with dialogue".to_string(),
    );
    let mut orchestrator = ParallelOrchestrator::new(blueprint);

    // Register dialogue as an agent
    orchestrator.add_agent(
        "design_team",
        std::sync::Arc::new(llm_toolkit::agent::AgentAdapter::new(dialogue)),
    );

    // Create a simple strategy that uses the dialogue
    let mut strategy = StrategyMap::new("Gather team feedback".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Get team perspective on the proposal".to_string(),
        "design_team".to_string(),
        "What do you think about the new feature?".to_string(),
        "Team feedback".to_string(),
    ));

    orchestrator.set_strategy_map(strategy);

    // Execute orchestrator
    let result = orchestrator
        .execute(
            "Evaluate the proposal",
            CancellationToken::new(),
            None,
            None,
        )
        .await;

    // Verify success
    assert!(result.is_ok());
    let orchestration_result = result.unwrap();
    assert_eq!(orchestration_result.steps_executed, 1);

    // Verify we got output from both dialogue participants
    let step_output = orchestration_result.context.get("step_1_output").unwrap();
    let output_array = step_output.as_array().unwrap();
    assert_eq!(output_array.len(), 2); // Broadcast returns all participants

    // Check both participants contributed
    let speakers: Vec<String> = output_array
        .iter()
        .filter_map(|turn| turn.get("speaker").and_then(|s| s.get("name")))
        .filter_map(|name| name.as_str())
        .map(|s| s.to_string())
        .collect();

    assert!(speakers.contains(&"Alice".to_string()));
    assert!(speakers.contains(&"Bob".to_string()));
}

#[tokio::test]
async fn test_orchestrator_with_sequential_dialogue() {
    // Create a sequential dialogue
    let persona1 = Persona {
        name: "Analyzer".to_string(),
        role: "Data Analyst".to_string(),
        background: "Statistics expert".to_string(),
        communication_style: "Analytical".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let persona2 = Persona {
        name: "Writer".to_string(),
        role: "Technical Writer".to_string(),
        background: "Documentation specialist".to_string(),
        communication_style: "Clear".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let agent1 = MockAgent::new("Analyzer", "Data shows positive trend");
    let agent2 = MockAgent::new("Writer", "Documented the analysis");

    let mut dialogue = Dialogue::sequential();
    dialogue.add_participant(persona1, agent1);
    dialogue.add_participant(persona2, agent2);

    // Setup orchestrator
    let blueprint = llm_toolkit::orchestrator::BlueprintWorkflow::new(
        "Analysis and documentation workflow".to_string(),
    );
    let mut orchestrator = ParallelOrchestrator::new(blueprint);

    orchestrator.add_agent(
        "analysis_pipeline",
        std::sync::Arc::new(llm_toolkit::agent::AgentAdapter::new(dialogue)),
    );

    // Create strategy
    let mut strategy = StrategyMap::new("Analyze and document data".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Run analysis pipeline".to_string(),
        "analysis_pipeline".to_string(),
        "Analyze the data".to_string(),
        "Analysis report".to_string(),
    ));

    orchestrator.set_strategy_map(strategy);

    // Execute
    let result = orchestrator
        .execute(
            "Process quarterly data",
            CancellationToken::new(),
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(result.steps_executed, 1);

    // Sequential dialogue returns only the final output
    let step_output = result.context.get("step_1_output").unwrap();
    let output_array = step_output.as_array().unwrap();
    assert_eq!(output_array.len(), 1); // Only Writer's final output

    let final_turn = &output_array[0];
    let speaker_name = final_turn
        .get("speaker")
        .and_then(|s| s.get("name"))
        .and_then(|n| n.as_str())
        .unwrap();
    assert_eq!(speaker_name, "Writer");
}

#[tokio::test]
async fn test_multiple_dialogue_agents_in_orchestrator() {
    // Create two different dialogue teams
    let persona_tech1 = Persona {
        name: "Backend".to_string(),
        role: "Backend Engineer".to_string(),
        background: "API specialist".to_string(),
        communication_style: "Technical".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let persona_tech2 = Persona {
        name: "Frontend".to_string(),
        role: "Frontend Engineer".to_string(),
        background: "UI specialist".to_string(),
        communication_style: "User-focused".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut tech_team = Dialogue::broadcast();
    tech_team.add_participant(
        persona_tech1,
        MockAgent::new("Backend", "API design is solid"),
    );
    tech_team.add_participant(persona_tech2, MockAgent::new("Frontend", "UI needs work"));

    let persona_biz1 = Persona {
        name: "PM".to_string(),
        role: "Product Manager".to_string(),
        background: "Business strategy".to_string(),
        communication_style: "Strategic".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let persona_biz2 = Persona {
        name: "Sales".to_string(),
        role: "Sales Lead".to_string(),
        background: "Customer relations".to_string(),
        communication_style: "Customer-centric".to_string(),
        visual_identity: None,
        capabilities: None,
    };

    let mut business_team = Dialogue::broadcast();
    business_team.add_participant(persona_biz1, MockAgent::new("PM", "Market fit is good"));
    business_team.add_participant(
        persona_biz2,
        MockAgent::new("Sales", "Customers will love it"),
    );

    // Setup orchestrator with both teams
    let blueprint =
        llm_toolkit::orchestrator::BlueprintWorkflow::new("Cross-functional review".to_string());
    let mut orchestrator = ParallelOrchestrator::new(blueprint);

    orchestrator.add_agent(
        "tech_team",
        std::sync::Arc::new(llm_toolkit::agent::AgentAdapter::new(tech_team)),
    );
    orchestrator.add_agent(
        "business_team",
        std::sync::Arc::new(llm_toolkit::agent::AgentAdapter::new(business_team)),
    );

    // Create strategy with both teams
    let mut strategy = StrategyMap::new("Comprehensive review".to_string());
    strategy.add_step(StrategyStep::new(
        "step_1".to_string(),
        "Get technical feedback".to_string(),
        "tech_team".to_string(),
        "Review the technical implementation".to_string(),
        "Technical feedback".to_string(),
    ));
    strategy.add_step(StrategyStep::new(
        "step_2".to_string(),
        "Get business feedback".to_string(),
        "business_team".to_string(),
        "Review from business perspective".to_string(),
        "Business feedback".to_string(),
    ));

    orchestrator.set_strategy_map(strategy);

    // Execute
    let result = orchestrator
        .execute(
            "Review the new feature",
            CancellationToken::new(),
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(result.steps_executed, 2);

    // Both dialogue teams should have provided feedback
    assert!(result.context.get("step_1_output").is_some());
    assert!(result.context.get("step_2_output").is_some());
}
