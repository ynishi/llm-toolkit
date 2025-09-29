use llm_toolkit::define_intent;

#[define_intent]
#[intent(mode = "multi_tag", prompt = "Test: {{ actions_doc }}")]
#[derive(Debug, Clone, PartialEq)]
pub enum SimpleAction {
    #[action(tag = "Test")]
    Test,
}

#[test]
fn test_simple_multi_tag() {
    let prompt = build_simple_action_prompt();
    assert!(prompt.contains("Test"));

    let extractor = SimpleActionExtractor;
    let actions = extractor.extract_actions("<Test />").unwrap();
    assert_eq!(actions.len(), 1);
}
