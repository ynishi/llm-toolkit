#### Orchestrator Features


- ✅ **Natural Language Blueprints**: Define workflows in plain English
- ✅ **Ad-hoc Strategy Generation**: LLM generates execution plans based on available agents
- ✅ **Two-Layer Error Recovery**: Combine RetryAgent (transient errors) + Orchestrator (structural errors)
- ✅ **3-Stage Error Recovery**:
  - **Retry**: For transient errors
  - **Tactical Redesign**: Modify failed steps and continue
  - **Full Regenerate**: Start over with a new strategy
- ✅ **Built-in Validation**: Automatic registration of `InnerValidatorAgent` as a fallback validator
- ✅ **Smart Context Management**: Automatic passing of outputs between steps with `ToPrompt` support
- ✅ **Configurable Error Recovery Limits**: Control retry behavior to prevent infinite loops
- ✅ **Fast Path Intent Generation**: Optional optimization to skip LLM calls for deterministic template substitution
- ✅ **Logging and Observability**: Stream execution logs in JSON format using `tracing` for real-time monitoring
- ✅ **Loop Control Flow**: Iterative refinement with `LoopBlock` (while/until convergence patterns)
- ✅ **Early Termination**: Conditional workflow exit with `TerminateInstruction`
- ✅ **Control Flow Safety**: Single-level loops only (nested loops rejected), global iteration limits
- ✅ **Execution Journal**: Complete execution history with step-by-step outcomes, timestamps, and error details
- ✅ **Strategy Lifecycle Management**: Unified trait for strategy injection, retrieval, and generation across orchestrator types

