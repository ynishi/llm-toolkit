//! Test for HashMap/BTreeMap/HashSet/BTreeSet with nested types expansion

#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, ToPrompt)]
    pub enum Priority {
        Low,
        Medium,
        High,
    }

    #[derive(
        Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, ToPrompt,
    )]
    pub enum Status {
        Pending,
        InProgress,
        Done,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, ToPrompt)]
    #[prompt(mode = "full")]
    pub struct TaskCollection {
        /// Map of task IDs to their priorities
        pub priorities: HashMap<String, Priority>,
        /// Map of user IDs to their statuses
        pub statuses: BTreeMap<String, Status>,
        /// Set of allowed priorities
        pub allowed_priorities: HashSet<Priority>,
        /// Ordered set of valid statuses
        pub valid_statuses: BTreeSet<Status>,
    }

    #[test]
    fn test_hashmap_nested_type_expansion() {
        let schema = TaskCollection::prompt_schema();
        println!(
            "\n=== TaskCollection schema ===\n{}\n=============================",
            schema
        );

        // Should reference nested enum types even when wrapped in HashMap
        assert!(
            schema.contains("Priority"),
            "Schema should reference Priority type"
        );
        assert!(
            schema.contains("Status"),
            "Schema should reference Status type"
        );

        // Should include nested enum definitions
        assert!(
            schema.contains("Low") || schema.contains("\"Low\""),
            "Schema should include Priority::Low"
        );
        assert!(
            schema.contains("Medium") || schema.contains("\"Medium\""),
            "Schema should include Priority::Medium"
        );
        assert!(
            schema.contains("High") || schema.contains("\"High\""),
            "Schema should include Priority::High"
        );

        assert!(
            schema.contains("Pending") || schema.contains("\"Pending\""),
            "Schema should include Status::Pending"
        );
        assert!(
            schema.contains("InProgress") || schema.contains("\"InProgress\""),
            "Schema should include Status::InProgress"
        );
        assert!(
            schema.contains("Done") || schema.contains("\"Done\""),
            "Schema should include Status::Done"
        );
    }

    #[test]
    fn test_collection_with_primitives() {
        #[derive(Debug, Clone, Serialize, Deserialize, ToPrompt)]
        #[prompt(mode = "full")]
        pub struct Config {
            /// Map of string keys to string values
            pub settings: HashMap<String, String>,
            /// Set of numbers
            pub ports: HashSet<u16>,
        }

        let schema = Config::prompt_schema();
        println!("\n=== Config schema ===\n{}\n=====================", schema);

        // Primitive collections should work correctly
        assert!(
            schema.contains("settings"),
            "Schema should contain settings field"
        );
        assert!(
            schema.contains("ports"),
            "Schema should contain ports field"
        );
    }

    #[test]
    fn test_optional_collection_nested_type() {
        #[derive(Debug, Clone, Serialize, Deserialize, ToPrompt)]
        #[prompt(mode = "full")]
        pub struct OptionalCollections {
            /// Optional map of priorities
            pub priorities: Option<HashMap<String, Priority>>,
            /// Optional set of statuses
            pub statuses: Option<HashSet<Status>>,
        }

        let schema = OptionalCollections::prompt_schema();
        println!(
            "\n=== OptionalCollections schema ===\n{}\n==================================",
            schema
        );

        // Should reference nested types even when wrapped in Option<HashMap>
        assert!(
            schema.contains("Priority"),
            "Schema should reference Priority type in Option<HashMap>"
        );
        assert!(
            schema.contains("Status"),
            "Schema should reference Status type in Option<HashSet>"
        );

        // Should include nested enum definitions
        assert!(
            schema.contains("Low") || schema.contains("\"Low\""),
            "Schema should include Priority::Low"
        );
        assert!(
            schema.contains("Pending") || schema.contains("\"Pending\""),
            "Schema should include Status::Pending"
        );
    }
}
