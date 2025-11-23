# Phase 2: Context-Aware Prompt Rendering

## Overview

Phase 2 adds context-aware prompt generation to `llm-toolkit-expertise`. This allows dynamic filtering and ordering of knowledge fragments based on runtime context (task type, user state, task health).

## Design Goals

1. **Runtime Flexibility**: Enable context-aware prompt generation without compile-time decisions
2. **Ecosystem Integration**: Seamlessly integrate with existing `ToPrompt` infrastructure
3. **DTO Pattern Support**: Work naturally with the template + DTO pattern
4. **Backward Compatibility**: Keep existing `to_prompt()` simple and unchanged

## Architecture

### 1. RenderContext

Encapsulates runtime context for prompt rendering:

```rust
#[derive(Debug, Clone, Default)]
pub struct RenderContext {
    /// Current task type (e.g., "security-review", "code-review", "debug")
    pub task_type: Option<String>,

    /// User states (e.g., "beginner", "expert", "confused")
    pub user_states: Vec<String>,

    /// Current task health status
    pub task_health: Option<TaskHealth>,
}

impl RenderContext {
    pub fn new() -> Self;
    pub fn with_task_type(self, task_type: impl Into<String>) -> Self;
    pub fn with_user_state(self, state: impl Into<String>) -> Self;
    pub fn with_task_health(self, health: TaskHealth) -> Self;

    /// Check if this context matches a ContextProfile
    pub fn matches(&self, profile: &ContextProfile) -> bool;
}
```

### 2. Expertise Extension

Add context-aware rendering to `Expertise`:

```rust
impl Expertise {
    /// Existing simple rendering (Phase 1)
    pub fn to_prompt(&self) -> PromptResult {
        // All fragments in order
    }

    /// NEW: Context-aware rendering (Phase 2)
    pub fn to_prompt_with_context(&self, context: &RenderContext) -> PromptResult {
        // 1. Filter fragments by context
        // 2. Sort by priority (Critical → High → Normal → Low)
        // 3. Render
    }
}
```

**Filtering Logic:**
- Fragment with `ContextProfile::Always` → Always included
- Fragment with `ContextProfile::Conditional { ... }` → Included only if `context.matches(profile)`

**Priority Ordering:**
1. Critical (MUST follow)
2. High (Recommended)
3. Normal (Standard guidance)
4. Low (Optional reference)

### 3. ContextualPrompt Wrapper

Wrapper type implementing `ToPrompt` for DTO integration:

```rust
/// Context-aware prompt renderer
pub struct ContextualPrompt<'a> {
    expertise: &'a Expertise,
    context: RenderContext,
}

impl<'a> ContextualPrompt<'a> {
    /// Create from expertise and context
    pub fn from_expertise(expertise: &'a Expertise, context: RenderContext) -> Self;

    /// Builder methods
    pub fn with_task_type(self, task_type: impl Into<String>) -> Self;
    pub fn with_user_state(self, state: impl Into<String>) -> Self;
    pub fn with_task_health(self, health: TaskHealth) -> Self;
}

/// Integrate with ToPrompt ecosystem
impl<'a> ToPrompt for ContextualPrompt<'a> {
    fn to_prompt(&self) -> PromptResult {
        self.expertise.to_prompt_with_context(&self.context)
    }
}
```

## Usage Patterns

### Pattern 1: Direct Call

```rust
let context = RenderContext::new()
    .with_task_type("security-review")
    .with_task_health(TaskHealth::AtRisk);

let prompt = expertise.to_prompt_with_context(&context)?;
```

### Pattern 2: Wrapper with Builder

```rust
let prompt = ContextualPrompt::from_expertise(&expertise, RenderContext::new())
    .with_task_type("security-review")
    .with_task_health(TaskHealth::AtRisk)
    .to_prompt()?;
```

### Pattern 3: DTO Integration

```rust
#[derive(ToPrompt)]
#[prompt(template = "Expert Knowledge:\n{{expertise}}\n\nTask: {{task}}")]
struct AgentRequestDto<'a> {
    expertise: ContextualPrompt<'a>,  // ToPrompt implemented!
    task: String,
}

// Usage
let context = RenderContext::new()
    .with_task_type("security-review")
    .with_task_health(TaskHealth::AtRisk);

let dto = AgentRequestDto {
    expertise: ContextualPrompt::from_expertise(&expertise, context),
    task: "Review this code".to_string(),
};

let prompt = dto.to_prompt()?;
// → expertise.to_prompt_with_context(&context) is automatically expanded
```

## Implementation Plan

### Step 1: Core Types
- [ ] Define `RenderContext` in `crates/llm-toolkit-expertise/src/render.rs`
- [ ] Implement builder methods and `matches()` logic

### Step 2: Context-Aware Rendering
- [ ] Add `to_prompt_with_context()` to `Expertise`
- [ ] Implement filtering logic
- [ ] Implement priority-based ordering

### Step 3: ContextualPrompt Wrapper
- [ ] Define `ContextualPrompt<'a>` type
- [ ] Implement builder pattern
- [ ] Implement `ToPrompt` trait

### Step 4: Testing
- [ ] Unit tests for `RenderContext::matches()`
- [ ] Tests for fragment filtering
- [ ] Tests for priority ordering
- [ ] Integration tests with DTO pattern
- [ ] Example demonstrating all patterns

### Step 5: Documentation
- [ ] Update README with Phase 2 examples
- [ ] Add rustdoc to new types
- [ ] Create example file `examples/contextual_prompt.rs`

## Future Extensions (Phase 3)

- State Analyzer to automatically infer `RenderContext` from conversation history
- Machine learning-based context detection
- Automatic TaskHealth inference

## Backward Compatibility

- ✅ Existing `to_prompt()` unchanged
- ✅ No breaking changes to Phase 1 API
- ✅ Opt-in feature (use `to_prompt_with_context()` when needed)
