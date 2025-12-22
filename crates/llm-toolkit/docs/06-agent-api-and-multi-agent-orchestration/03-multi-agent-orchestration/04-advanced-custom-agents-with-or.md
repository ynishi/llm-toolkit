#### Advanced: Custom Agents with Orchestrator


You can combine custom agents (defined with `#[derive(Agent)]`) with the orchestrator:

```rust
#[derive(Serialize, Deserialize)]
struct ResearchData {
    sources: Vec<String>,
    key_points: Vec<String>,
}

#[derive(Agent)]
#[agent(
    expertise = "Deep research on technical topics with source citations",
    output = "ResearchData"
)]
struct ResearchAgent;

#[derive(Agent)]
#[agent(
    expertise = "Writing clear, beginner-friendly technical content",
    output = "ArticleDraft"
)]
struct WriterAgent;

// Add both to orchestrator (InnerValidatorAgent is automatically registered)
let mut orchestrator = Orchestrator::new(blueprint);
orchestrator.add_agent(Box::new(ResearchAgent));
orchestrator.add_agent(Box::new(WriterAgent));

// The orchestrator will automatically select the best agent for each step
```

