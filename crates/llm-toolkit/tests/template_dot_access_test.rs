//! Test for dot access in templates with #[prompt(as_serialize)] attribute
//!
//! ToPrompt philosophy:
//! - Default: {{ profile }} calls profile.to_prompt() - type controls its representation
//! - #[prompt(as_serialize)]: enables dot access like {{ profile.name }}

use llm_toolkit::ToPrompt;
use serde::Serialize;

/// Nested struct with Serialize for dot access
#[derive(Debug, Clone, Serialize)]
struct Profile {
    name: String,
    role: String,
}

/// ToPrompt implementation for Profile that returns a formatted string
impl ToPrompt for Profile {
    fn to_prompt(&self) -> String {
        format!("Profile: {} ({})", self.name, self.role)
    }
}

/// Test: Default behavior uses to_prompt() - type controls its representation
#[derive(ToPrompt, Serialize)]
#[prompt(template = "{{ profile }}")]
struct DefaultExample {
    profile: Profile,
}

#[test]
fn test_default_uses_to_prompt() {
    let example = DefaultExample {
        profile: Profile {
            name: "Alice".to_string(),
            role: "Admin".to_string(),
        },
    };

    let output = example.to_prompt();

    // Default: should use to_prompt() which returns "Profile: Alice (Admin)"
    assert_eq!(output, "Profile: Alice (Admin)");
}

/// Test: #[prompt(as_serialize)] enables dot access
#[derive(ToPrompt, Serialize)]
#[prompt(template = "User: {{ profile.name }}, Role: {{ profile.role }}")]
struct DotAccessExample {
    #[prompt(as_serialize)]
    profile: Profile,
}

#[test]
fn test_as_serialize_enables_dot_access() {
    let example = DotAccessExample {
        profile: Profile {
            name: "Bob".to_string(),
            role: "User".to_string(),
        },
    };

    let output = example.to_prompt();

    // as_serialize: enables dot access to individual fields
    assert_eq!(output, "User: Bob, Role: User");
}

/// Test: Mixed - some fields with dot access, some with default to_prompt
#[derive(ToPrompt, Serialize)]
#[prompt(template = "Name: {{ data.name }}, Description: {{ description }}")]
struct MixedExample {
    /// Data with dot access (as_serialize)
    #[prompt(as_serialize)]
    data: Profile,
    /// Description using default to_prompt()
    description: Profile,
}

#[test]
fn test_mixed_as_serialize_and_default() {
    let example = MixedExample {
        data: Profile {
            name: "Charlie".to_string(),
            role: "Developer".to_string(),
        },
        description: Profile {
            name: "Charlie".to_string(),
            role: "Developer".to_string(),
        },
    };

    let output = example.to_prompt();

    // data.name uses dot access, description uses to_prompt()
    assert_eq!(
        output,
        "Name: Charlie, Description: Profile: Charlie (Developer)"
    );
}

/// Deeply nested struct for multi-level dot access
#[derive(Debug, Clone, Serialize, ToPrompt)]
struct Company {
    name: String,
    ceo: Profile,
}

#[derive(ToPrompt, Serialize)]
#[prompt(template = "Company: {{ company.name }}, CEO: {{ company.ceo.name }}")]
struct DeepDotAccessExample {
    #[prompt(as_serialize)]
    company: Company,
}

#[test]
fn test_deep_dot_access() {
    let example = DeepDotAccessExample {
        company: Company {
            name: "Acme Corp".to_string(),
            ceo: Profile {
                name: "Dana".to_string(),
                role: "CEO".to_string(),
            },
        },
    };

    let output = example.to_prompt();

    // Should access deeply nested fields
    assert_eq!(output, "Company: Acme Corp, CEO: Dana");
}

/// Test with Option<T> - dot access should work with as_serialize
#[derive(ToPrompt, Serialize)]
#[prompt(template = "{% if profile %}Name: {{ profile.name }}{% else %}No profile{% endif %}")]
struct OptionDotAccessExample {
    #[prompt(as_serialize)]
    profile: Option<Profile>,
}

#[test]
fn test_option_dot_access_some() {
    let example = OptionDotAccessExample {
        profile: Some(Profile {
            name: "Eve".to_string(),
            role: "Manager".to_string(),
        }),
    };

    let output = example.to_prompt();
    assert_eq!(output, "Name: Eve");
}

#[test]
fn test_option_dot_access_none() {
    let example = OptionDotAccessExample { profile: None };

    let output = example.to_prompt();
    assert_eq!(output, "No profile");
}

/// Test with Vec<T> - iteration and dot access with as_serialize
#[derive(ToPrompt, Serialize)]
#[prompt(template = "{% for p in profiles %}{{ p.name }}, {% endfor %}")]
struct VecDotAccessExample {
    #[prompt(as_serialize)]
    profiles: Vec<Profile>,
}

#[test]
fn test_vec_dot_access() {
    let example = VecDotAccessExample {
        profiles: vec![
            Profile {
                name: "Frank".to_string(),
                role: "Dev".to_string(),
            },
            Profile {
                name: "Grace".to_string(),
                role: "QA".to_string(),
            },
        ],
    };

    let output = example.to_prompt();
    assert_eq!(output, "Frank, Grace, ");
}
