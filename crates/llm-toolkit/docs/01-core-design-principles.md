## Core Design Principles


1.  **Minimalist & Unopinionated:**
    The toolkit will NOT impose any specific application architecture. Developers are free to design their own `UseCase`s and `Service`s. `llm-toolkit` simply provides a set of sharp, reliable "tools" to be called when needed.

2.  **Focused on the "Last Mile Problem":**
    The toolkit focuses on solving the most common and frustrating problems that occur at the boundary between a strongly-typed Rust application and the unstructured, often unpredictable string-based responses from LLM APIs.

3.  **Minimal Dependencies:**
    The toolkit will have minimal dependencies (primarily `serde` and `minijinja`) to ensure it can be added to any Rust project with negligible overhead and maximum compatibility.

