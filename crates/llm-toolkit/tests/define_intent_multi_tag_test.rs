use llm_toolkit::define_intent;

// 1. Define the enum with the new `multi_tag` mode
#[define_intent]
#[intent(
    mode = "multi_tag",
    prompt = r#"Based on the user request, generate a response using the following available actions.

**Available Actions:**
{{ actions_doc }}

**User Request:**
{{ user_request }}"#
)]
#[derive(Debug, Clone, PartialEq)]
pub enum ChatAction {
    /// A simple, parameter-less action to get the weather.
    #[action(tag = "GetWeather")]
    GetWeather,

    /// Shows an image to the user.
    #[action(tag = "ShowImage")]
    ShowImage {
        /// The URL of the image to display.
        #[action(attribute)]
        href: String,
    },

    /// A thought process that is not shown to the user.
    #[action(tag = "Thought")]
    Thought(#[action(inner_text)] String),
    /// Sends a message to a recipient.
    #[action(tag = "SendMessage")]
    SendMessage {
        /// The recipient of the message.
        #[action(attribute)]
        to: String,
        /// The content of the message.
        #[action(inner_text)]
        content: String,
    },
}

// The macro should generate:
// - `build_chat_action_prompt(user_request: &str) -> String`
// - `struct ChatActionExtractor` with `extract_actions(&self, text: &str) -> Result<Vec<ChatAction>, IntentError>`

#[test]
fn test_multi_tag_prompt_generation() {
    let prompt = build_chat_action_prompt("What's the weather and can you show me a cat picture?");

    // Check that the main prompt structure is there
    assert!(prompt.contains("User Request:"));
    assert!(prompt.contains("What's the weather and can you show me a cat picture?"));

    // Check that `actions_doc` was generated and injected correctly
    let expected_docs = [
        r"- `<GetWeather />`: A simple, parameter-less action to get the weather.",
        r#"- `<ShowImage href="..." />`: Shows an image to the user."#,
        r"  - `href` (attribute): The URL of the image to display.",
        r"- `<Thought>...</Thought>`: A thought process that is not shown to the user.",
    ];

    for doc_line in expected_docs {
        assert!(prompt.contains(doc_line), "Missing doc line: {}", doc_line);
    }
}

#[test]
fn test_multi_tag_extraction_success() {
    let extractor = ChatActionExtractor;
    let llm_response = r#"
        <Thought>Okay, I need to get the weather and show a picture.</Thought>
        Here is the weather: <GetWeather />
        And here is a cat picture: <ShowImage href="https://cataas.com/cat" />
        <SendMessage to="user">I've completed both of your requests.</SendMessage>
        <NoIntent />
        <NoIntentContent>no intent</NoIntentContent>
    "#;

    let actions = extractor.extract_actions(llm_response).unwrap();

    assert_eq!(actions.len(), 4);
    assert_eq!(
        actions[0],
        ChatAction::Thought("Okay, I need to get the weather and show a picture.".to_string())
    );
    assert_eq!(actions[1], ChatAction::GetWeather);
    assert_eq!(
        actions[2],
        ChatAction::ShowImage {
            href: "https://cataas.com/cat".to_string()
        }
    );
    assert_eq!(
        actions[3],
        ChatAction::SendMessage {
            to: "user".to_string(),
            content: "I've completed both of your requests.".to_string(),
        }
    );
}

#[test]
fn test_multi_tag_extraction_empty_and_unknown() {
    let extractor = ChatActionExtractor;
    let llm_response = "This response has <UnknownTag /> and no valid actions.";

    let actions = extractor.extract_actions(llm_response).unwrap();
    assert!(actions.is_empty());

    let llm_response_empty = "";
    let actions_empty = extractor.extract_actions(llm_response_empty).unwrap();
    assert!(actions_empty.is_empty());
}

#[test]
fn test_multi_tag_strip_actions() {
    let extractor = ChatActionExtractor;

    // Test stripping single tag
    let text = "Hello <GetWeather /> world";
    let result = extractor.strip_actions(text);
    assert_eq!(result, "Hello  world");

    // Test stripping multiple tags
    let text = "Start <GetWeather /> middle <ShowImage href=\"test.jpg\" /> end";
    let result = extractor.strip_actions(text);
    assert_eq!(result, "Start  middle  end");

    // Test text without tags
    let text = "Just plain text";
    let result = extractor.strip_actions(text);
    assert_eq!(result, "Just plain text");
}

#[test]
fn test_multi_tag_transform_actions() {
    let extractor = ChatActionExtractor;

    // Test transform with placeholder
    let text = "Execute <GetWeather /> and then <ShowImage href=\"cat.jpg\" />";
    let result = extractor.transform_actions(text, |action| match action {
        ChatAction::GetWeather => "[WEATHER]".to_string(),
        ChatAction::ShowImage { href } => format!("[IMAGE: {}]", href),
        ChatAction::Thought(content) => format!("[THOUGHT: {}]", content),
        ChatAction::SendMessage { to, content } => format!("[MSG TO {}: {}]", to, content),
    });
    assert_eq!(result, "Execute [WEATHER] and then [IMAGE: cat.jpg]");

    // Test transform with descriptive text
    let text = "Please <GetWeather /> for me";
    let result = extractor.transform_actions(text, |action| match action {
        ChatAction::GetWeather => "check the weather".to_string(),
        _ => "perform action".to_string(),
    });
    assert_eq!(result, "Please check the weather for me");
}
