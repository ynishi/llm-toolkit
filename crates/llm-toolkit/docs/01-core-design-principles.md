## Core Design Principles


1.  **Minimalist & Unopinionated:**
    The toolkit will NOT impose any specific application architecture. Developers are free to design their own `UseCase`s and `Service`s. `llm-toolkit` simply provides a set of sharp, reliable "tools" to be called when needed.

2.  **Focused on the "Last Mile Problem":**
    The toolkit focuses on solving the most common and frustrating problems that occur at the boundary between a strongly-typed Rust application and the unstructured, often unpredictable string-based responses from LLM APIs.

3.  **Minimal Dependencies:**
    The toolkit will have minimal dependencies (primarily `serde` and `minijinja`) to ensure it can be added to any Rust project with negligible overhead and maximum compatibility.

4.  **Prompt as Presentation with Logic:**
    Just as JSX revolutionized web development by acknowledging that "UI contains logic" and providing type-safe ways to handle it, `ToPrompt` acknowledges that **prompts contain logic** and provides type-safe ways to handle them in Rust.

    Think of it as an **ORM for Prompts**:
    - **ORM**: "Database access contains logic" → type-safe Rust ↔ DB mapping
    - **JSX**: "UI contains logic" → type-safe component rendering
    - **ToPrompt**: "Prompts contain logic" → type-safe Rust → Prompt transformation

    Each type controls its own prompt representation through `to_prompt()`, just as React components control their own rendering. This enables:
    - **Encapsulation**: Types own their prompt logic
    - **Composability**: Complex prompts from simple building blocks
    - **Type Safety**: Compile-time guarantees for prompt generation
    - **Testability**: Unit test prompt output like any other code

