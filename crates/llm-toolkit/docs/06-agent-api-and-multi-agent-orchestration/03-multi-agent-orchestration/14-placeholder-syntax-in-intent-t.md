#### Placeholder Syntax in Intent Templates


Intent templates use **Mustache/Jinja2-style double curly braces** `{{ }}` for placeholder substitution. This is **not a typo** - single braces `{ }` are **not recognized**.

**Correct Syntax:**

```rust
"Create an outline for: {{ task }}"           // ✅ Correct
"Based on {{ previous_output }}, continue"    // ✅ Correct
"Transform {{ step_3_output }}"               // ✅ Correct
```

**Incorrect Syntax:**

```rust
"Create an outline for: {task}"               // ❌ Will NOT be recognized
"Based on {previous_output}, continue"        // ❌ Will NOT be recognized
```

**Important Notes:**

- Always use **double curly braces** with spaces: `{{ name }}` (not `{{name}}`)
- This matches the Mustache/Jinja2 templating convention
- The orchestrator's `extract_placeholders` only detects `{{ }}` format
- LLM-generated intent templates follow this convention from prompts.rs

**Common Placeholders:**

- `{{ task }}` - The original user task
- `{{ previous_output }}` - Output from the immediately previous step
- `{{ step_N_output }}` - Output from a specific step (e.g., `{{ step_3_output }}`)
- Custom semantic names (e.g., `{{ concept_content }}`, `{{ emblem_design }}`)

