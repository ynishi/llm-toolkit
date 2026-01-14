//! Content extraction and JSON repair utilities for LLM responses.
//!
//! This module provides tools for extracting structured data from unstructured
//! LLM outputs and repairing common JSON syntax errors.
//!
//! # Features
//!
//! - **Content Extraction**: Extract JSON objects, tagged content, and code blocks
//! - **JSON Sanitization**: Auto-fix trailing commas, unclosed brackets/strings
//! - **Fuzzy Repair**: Schema-based typo correction for tagged enums
//!
//! # Examples
//!
//! ## Extract JSON from LLM response
//!
//! ```rust
//! use llm_toolkit::extract::FlexibleExtractor;
//!
//! let extractor = FlexibleExtractor::new();
//! let response = r#"Here's the data: {"status": "ok", "count": 42}"#;
//! let json = extractor.extract(response).unwrap();
//! assert_eq!(json, r#"{"status": "ok", "count": 42}"#);
//! ```
//!
//! ## Sanitize malformed JSON
//!
//! ```rust
//! use llm_toolkit::extract::sanitize_json;
//!
//! // Fix trailing commas
//! let fixed = sanitize_json(r#"{"name": "Alice", "age": 30,}"#);
//! assert_eq!(fixed, r#"{"name": "Alice", "age": 30}"#);
//! ```
//!
//! ## Repair typos in tagged enums
//!
//! ```rust
//! use llm_toolkit::extract::{repair_tagged_enum_json, TaggedEnumSchema, FuzzyOptions};
//!
//! let schema = TaggedEnumSchema::new(
//!     "type",
//!     &["AddDerive", "RemoveDerive"],
//!     |_| None,
//! );
//!
//! // LLM output has typo: "AddDeriv" instead of "AddDerive"
//! let result = repair_tagged_enum_json(
//!     r#"{"type": "AddDeriv", "target": "MyStruct"}"#,
//!     &schema,
//!     &FuzzyOptions::default(),
//! ).unwrap();
//!
//! assert!(result.repaired.to_string().contains("AddDerive"));
//! ```

pub mod core;
pub mod error;
pub mod extractors;

pub use self::core::{ContentExtractor, ExtractionStrategy, ParsingConfig};
pub use self::error::ParseError;
pub use self::extractors::{FlexibleExtractor, MarkdownCodeBlockExtractor};

// Re-export fuzzy-parser for LLM JSON repair capabilities
pub use fuzzy_parser::{
    Algorithm, Correction, FuzzyError, FuzzyOptions, RepairResult, TaggedEnumSchema,
    repair_tagged_enum_json, sanitize_json,
};
