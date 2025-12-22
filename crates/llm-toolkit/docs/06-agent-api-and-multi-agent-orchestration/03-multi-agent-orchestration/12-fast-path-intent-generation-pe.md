#### Fast Path Intent Generation (Performance Optimization)


By default, the orchestrator uses LLM-based intent generation for each step, which provides high-quality, context-aware prompts but incurs API latency and costs. For workflows with simple template substitution (all placeholders resolved from context), you can enable **fast path optimization** to skip LLM calls.

**When to Enable:**
- ✅ **Thick Agents**: Agents that contain detailed domain logic and don't need LLM-optimized prompts
- ✅ **Simple Templates**: Intent templates with straightforward placeholder substitution
- ✅ **Performance-Critical Workflows**: When latency matters more than prompt quality
- ✅ **High-Volume Operations**: When API costs need to be minimized

**When to Keep Disabled (Default):**
- ❌ **Thin Agents**: Agents that rely on rich, context-aware prompts from the LLM
- ❌ **Complex Reasoning**: Workflows requiring semantic understanding and prompt adaptation
- ❌ **Quality-First Applications**: When prompt quality is more important than speed

**Usage:**

```rust
use std::time::Duration;
use llm_toolkit::orchestrator::{Orchestrator, OrchestratorConfig};

let mut orchestrator = Orchestrator::new(blueprint);

// Enable fast path optimization
let config = OrchestratorConfig {
    enable_fast_path_intent_generation: true,  // Default: false
    ..Default::default()
};
orchestrator.set_config(config);

// Execute - fast path will be used when all placeholders are resolved
let result = orchestrator.execute(task).await;
```

**How It Works:**

For each step, the orchestrator:
1. **Checks prerequisites**: Are all placeholders in the intent template resolved in context?
2. **Fast path (if enabled + all resolved)**: Simple string substitution (milliseconds, no API call)
3. **LLM path (fallback)**: LLM generates high-quality, context-aware intent (seconds, API call)

**Example:**

```rust
// Intent template from strategy
"Transform this data: {{previous_output}}"

// If fast path enabled and previous_output exists in context:
// → Fast path: Direct substitution → "Transform this data: <actual output>"
// → Latency: ~1ms, Cost: $0

// If fast path disabled or placeholder not resolved:
// → LLM path: Generate intent considering agent expertise → High-quality prompt
// → Latency: ~2s, Cost: ~$0.001
```

**Performance Benefits (Example E2E Test Results):**

```
3-step workflow with mock 100ms LLM delay:
- Fast Path ENABLED:  412ms (1.49x faster)
- Fast Path DISABLED: 615ms

Real-world with actual LLM calls:
- Fast Path: ~50ms per step → 150ms for 3 steps
- LLM Path: ~2s per step → 6s for 3 steps
- Speedup: 40x faster!
```

**Trade-offs:**

| Aspect | Fast Path (Enabled) | LLM Path (Disabled, Default) |
|--------|---------------------|------------------------------|
| **Performance** | ⚡ Milliseconds | 🐌 Seconds |
| **API Cost** | 💰 Zero | 💰💰 Per step |
| **Prompt Quality** | Basic (template substitution) | High (context-aware, semantic) |
| **Best For** | Thick agents, simple templates | Thin agents, complex reasoning |

**Best Practices:**

1. **Default to disabled** - Prioritize quality for thin agent architectures
2. **Enable selectively** - Use for specific workflows where you've validated template quality
3. **Test both modes** - Compare results to ensure fast path doesn't sacrifice quality
4. **Monitor logs** - Watch for `"Using fast path"` vs `"Using LLM-based intent generation"` messages

**Complete E2E Example:**

See `examples/orchestrator_fast_path_e2e.rs` for a complete example comparing both modes:

```bash
cargo run --example orchestrator_fast_path_e2e --features agent,derive
```

This example demonstrates:
- Performance comparison between fast path and LLM path
- Validation that both produce equivalent results
- Configuration toggling
- Practical speedup measurements

