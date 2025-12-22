#### Accessing Intermediate Results (v0.13.6+)


You can now access intermediate step results using the context accessor methods:

```rust
let result = orchestrator.execute(task).await;

// Option 1: Get specific step output
if let Some(concept) = orchestrator.get_step_output("step_1") {
    // Deserialize to your type
    let concept: HighConceptResponse = serde_json::from_value(concept.clone())?;
    println!("Concept: {:?}", concept);
}

// Option 2: Get human-readable version (if ToPrompt was used)
if let Some(prompt) = orchestrator.get_step_output_prompt("step_1") {
    println!("Concept (readable):\n{}", prompt);
}

// Option 3: Get all step outputs
for (step_id, output) in orchestrator.get_all_step_outputs() {
    println!("Step {}: {:?}", step_id, output);
}

// Option 4: Access raw context
let context = orchestrator.context();
println!("Full context: {:?}", context);
```

**Available methods:**

- `context()` - Returns full context HashMap
- `get_step_output(step_id)` - Get JSON output of a specific step
- `get_step_output_prompt(step_id)` - Get ToPrompt version (human-readable)
- `get_all_step_outputs()` - Get all step outputs as HashMap

**Note:** These methods are available after `execute()` completes. The context is preserved until the next `execute()` call.

