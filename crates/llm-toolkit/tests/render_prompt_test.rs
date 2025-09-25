use llm_toolkit::prompt::render_prompt;
use serde::Serialize;

#[derive(Serialize)]
struct TestData {
    name: String,
    value: i32,
}

#[test]
fn test_render_prompt_basic() {
    let data = TestData {
        name: "test".to_string(),
        value: 42,
    };

    let template = "Name: {{name}}, Value: {{value}}";
    let result = render_prompt(template, &data).unwrap();

    assert_eq!(result, "Name: test, Value: 42");
}

#[test]
fn test_render_prompt_with_nested() {
    #[derive(Serialize)]
    struct Inner {
        field: String,
    }

    #[derive(Serialize)]
    struct Outer {
        inner: Inner,
    }

    let data = Outer {
        inner: Inner {
            field: "nested".to_string(),
        },
    };

    let template = "Inner field: {{inner.field}}";
    let result = render_prompt(template, &data).unwrap();

    assert_eq!(result, "Inner field: nested");
}
