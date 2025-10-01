// This file will contain the integration test for the Agent derive macro.
// We will define a simple struct for output, and an agent struct that uses `#[derive(Agent)]`.
// Then we will use trybuild to ensure it compiles successfully.

#[test]
fn agent_derive_compile_pass() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/agent_derive_basic.rs");
}

#[test]
fn agent_derive_with_backend_compile_pass() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/agent_derive_with_backend.rs");
}
