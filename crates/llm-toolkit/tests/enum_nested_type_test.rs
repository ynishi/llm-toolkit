//! Test for nested type expansion in enum variants

#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToPrompt)]
    struct UserProfile {
        /// User's name
        name: String,
        /// User's age
        age: u32,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToPrompt)]
    enum Action {
        /// Create a new user
        CreateUser { user: UserProfile },
        /// Update user profile
        UpdateProfile(UserProfile),
        /// Simple text message
        Message(String),
    }

    #[test]
    fn test_nested_struct_in_enum_variant() {
        // First, check what UserProfile schema looks like
        let user_profile_schema = UserProfile::prompt_schema();
        println!(
            "\n=== UserProfile schema ===\n{}\n==========================",
            user_profile_schema
        );

        let schema = Action::prompt_schema();
        println!("\n=== Action schema ===\n{}\n=====================", schema);

        // Should reference UserProfile type with proper casing
        assert!(
            schema.contains("UserProfile"),
            "Schema should reference UserProfile type (not lowercase)"
        );

        // Should include nested type definition
        assert!(
            schema.contains("name:") && schema.contains("age:"),
            "Schema should include nested UserProfile definition with name and age fields"
        );
    }
}
