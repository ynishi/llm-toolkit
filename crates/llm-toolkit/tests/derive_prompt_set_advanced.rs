use llm_toolkit::{ToPrompt, ToPromptSet};
use serde::Serialize;

// Test: Invalid format_with function
// This intentionally causes a compile error to demonstrate error handling
// Uncomment to test compile-time error:
// #[derive(ToPromptSet, Serialize, Debug)]
// struct InvalidFormatWith {
//     #[prompt_for(name = "Test", format_with = "non::existent::function")]
//     value: String,
// }

// This test should fail to compile with our improved error handling
// We can't actually test this in a normal test, but we can document it
// #[test]
// fn test_invalid_format_with() {
//     // This should produce a compile error: "Invalid function path in format_with"
// }

// Test: Unicode and internationalization
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Japanese",
    template = "タイトル: {{title}}\n説明: {{description}}"
)]
#[prompt_for(name = "Emoji", template = "🎯 {{title}}\n📝 {{description}}")]
struct InternationalContent {
    title: String,
    description: String,
}

#[test]
fn test_unicode_templates() {
    let content = InternationalContent {
        title: "テスト".to_string(),
        description: "これは日本語のテストです".to_string(),
    };

    let japanese = content.to_prompt_for("Japanese").unwrap();
    assert!(japanese.contains("タイトル: テスト"));
    assert!(japanese.contains("説明: これは日本語のテストです"));

    let emoji = content.to_prompt_for("Emoji").unwrap();
    assert!(emoji.contains("🎯 テスト"));
    assert!(emoji.contains("📝 これは日本語のテストです"));
}

// Test: Large number of fields performance
#[derive(ToPromptSet, Serialize, Debug)]
struct LargeStruct {
    field1: String,
    field2: String,
    field3: String,
    field4: String,
    field5: String,
    field6: String,
    field7: String,
    field8: String,
    field9: String,
    field10: String,
    #[prompt_for(name = "Subset")]
    field11: String,
    #[prompt_for(name = "Subset")]
    field12: String,
    #[prompt_for(name = "Subset")]
    field13: String,
    #[prompt_for(name = "Subset")]
    field14: String,
    #[prompt_for(name = "Subset")]
    field15: String,
    #[prompt_for(skip)]
    field16: String,
    #[prompt_for(skip)]
    field17: String,
    #[prompt_for(skip)]
    field18: String,
    #[prompt_for(skip)]
    field19: String,
    #[prompt_for(skip)]
    field20: String,
}

#[test]
fn test_large_struct_performance() {
    let large = LargeStruct {
        field1: "1".to_string(),
        field2: "2".to_string(),
        field3: "3".to_string(),
        field4: "4".to_string(),
        field5: "5".to_string(),
        field6: "6".to_string(),
        field7: "7".to_string(),
        field8: "8".to_string(),
        field9: "9".to_string(),
        field10: "10".to_string(),
        field11: "11".to_string(),
        field12: "12".to_string(),
        field13: "13".to_string(),
        field14: "14".to_string(),
        field15: "15".to_string(),
        field16: "16".to_string(),
        field17: "17".to_string(),
        field18: "18".to_string(),
        field19: "19".to_string(),
        field20: "20".to_string(),
    };

    // Test that all fields are included except skipped ones
    // Since no default target is defined, we need to check if the error is correct
    let default_result = large.to_prompt_for("Default");
    assert!(default_result.is_err());

    // Create another instance without target-specific fields for general test
    let err_msg = large.to_prompt_for("General").unwrap_err().to_string();
    assert!(err_msg.contains("Available targets: [\"Subset\"]"));

    // Subset target should only have specific fields
    let subset = large.to_prompt_for("Subset").unwrap();
    for i in 11..=15 {
        assert!(subset.contains(&format!("field{}: {}", i, i)));
    }
    // Skipped fields should not appear
    for i in 16..=20 {
        assert!(!subset.contains(&format!("field{}", i)));
    }
}

// Test: Thread safety with multiple targets
#[derive(ToPromptSet, Serialize, Debug, Clone)]
#[prompt_for(name = "Thread1", template = "T1: {{data}}")]
#[prompt_for(name = "Thread2", template = "T2: {{data}}")]
struct ThreadSafeStruct {
    data: String,
}

#[test]
fn test_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let shared = Arc::new(ThreadSafeStruct {
        data: "shared_data".to_string(),
    });

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let shared_clone = Arc::clone(&shared);
            thread::spawn(move || {
                let target = if i % 2 == 0 { "Thread1" } else { "Thread2" };
                let result = shared_clone.to_prompt_for(target).unwrap();
                if i % 2 == 0 {
                    assert!(result.contains("T1: shared_data"));
                } else {
                    assert!(result.contains("T2: shared_data"));
                }
                i
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

// Test: Complex template with conditionals and loops (if minijinja supports them)
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Complex",
    template = "Items: {% for item in items %}{{item}}{% if not loop.last %}, {% endif %}{% endfor %}"
)]
struct ComplexTemplate {
    items: Vec<String>,
}

#[test]
fn test_complex_template() {
    let data = ComplexTemplate {
        items: vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ],
    };

    let result = data.to_prompt_for("Complex").unwrap();
    assert_eq!(result, "Items: apple, banana, cherry");
}

// Test: Special characters in field values with multiple targets
#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(
    name = "Raw",
    template = "{{field_with_quotes}}|{{field_with_newlines}}|{{field_with_tabs}}"
)]
struct SpecialChars {
    #[prompt_for(name = "Escaped")]
    field_with_quotes: String,
    #[prompt_for(name = "Escaped")]
    field_with_newlines: String,
    #[prompt_for(name = "Escaped")]
    field_with_tabs: String,
}

#[test]
fn test_special_characters() {
    let data = SpecialChars {
        field_with_quotes: r#"He said "Hello""#.to_string(),
        field_with_newlines: "Line1\nLine2\nLine3".to_string(),
        field_with_tabs: "Tab\tSeparated\tValues".to_string(),
    };

    // Test Raw target with template
    let raw = data.to_prompt_for("Raw").unwrap();
    assert_eq!(
        raw,
        r#"He said "Hello"|Line1
Line2
Line3|Tab	Separated	Values"#
    );

    // Test Escaped target with key-value format
    let escaped = data.to_prompt_for("Escaped").unwrap();
    assert!(escaped.contains(r#"field_with_quotes: He said "Hello""#));
    assert!(escaped.contains("field_with_newlines: Line1\nLine2\nLine3"));
    assert!(escaped.contains("field_with_tabs: Tab\tSeparated\tValues"));
}

// Test: Empty Vec and Option fields with wrapper types
#[derive(Debug, Clone, Serialize)]
struct OptionalString(Option<String>);

impl ToPrompt for OptionalString {
    fn to_prompt_parts(&self) -> Vec<llm_toolkit::prompt::PromptPart> {
        vec![llm_toolkit::prompt::PromptPart::Text(
            self.0.as_deref().unwrap_or("None").to_string(),
        )]
    }

    fn to_prompt(&self) -> String {
        self.0.as_deref().unwrap_or("None").to_string()
    }
}

#[derive(Debug, Clone, Serialize)]
struct StringList(Vec<String>);

impl ToPrompt for StringList {
    fn to_prompt_parts(&self) -> Vec<llm_toolkit::prompt::PromptPart> {
        vec![llm_toolkit::prompt::PromptPart::Text(
            if self.0.is_empty() {
                "[]".to_string()
            } else {
                format!("[{}]", self.0.join(", "))
            },
        )]
    }

    fn to_prompt(&self) -> String {
        if self.0.is_empty() {
            "[]".to_string()
        } else {
            format!("[{}]", self.0.join(", "))
        }
    }
}

#[derive(ToPromptSet, Serialize, Debug)]
#[prompt_for(name = "Compact", template = "{{required}}: {{optional_some}}")]
struct OptionalFields {
    #[prompt_for(name = "Full")]
    required: String,
    #[prompt_for(name = "Full")]
    optional_some: OptionalString,
    #[prompt_for(name = "Full")]
    optional_none: OptionalString,
    #[prompt_for(name = "Full")]
    vec_empty: StringList,
    #[prompt_for(name = "Full")]
    vec_filled: StringList,
}

#[test]
fn test_optional_and_vec_fields() {
    let data = OptionalFields {
        required: "present".to_string(),
        optional_some: OptionalString(Some("value".to_string())),
        optional_none: OptionalString(None),
        vec_empty: StringList(vec![]),
        vec_filled: StringList(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
    };

    // Test Full target with all fields
    let full = data.to_prompt_for("Full").unwrap();
    assert!(full.contains("required: present"));
    assert!(full.contains("optional_some: value"));
    assert!(full.contains("optional_none: None"));
    assert!(full.contains("vec_empty: []"));
    assert!(full.contains("vec_filled: [a, b, c]"));

    // Test Compact target with template
    let compact = data.to_prompt_for("Compact").unwrap();
    assert_eq!(compact, "present: value");
}
