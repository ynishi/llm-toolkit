use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::quote;
use regex::Regex;
use syn::{
    Data, DeriveInput, Meta, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// Parse template placeholders using regex to find :mode patterns
/// Returns a list of (field_name, optional_mode)
fn parse_template_placeholders_with_mode(template: &str) -> Vec<(String, Option<String>)> {
    let mut placeholders = Vec::new();
    let mut seen_fields = std::collections::HashSet::new();

    // First, find all {{ field:mode }} patterns
    let mode_pattern = Regex::new(r"\{\{\s*(\w+)\s*:\s*(\w+)\s*\}\}").unwrap();
    for cap in mode_pattern.captures_iter(template) {
        let field_name = cap[1].to_string();
        let mode = cap[2].to_string();
        placeholders.push((field_name.clone(), Some(mode)));
        seen_fields.insert(field_name);
    }

    // Then, find all standard {{ field }} patterns (without mode)
    let standard_pattern = Regex::new(r"\{\{\s*(\w+)\s*\}\}").unwrap();
    for cap in standard_pattern.captures_iter(template) {
        let field_name = cap[1].to_string();
        // Check if this field was already captured with a mode
        if !seen_fields.contains(&field_name) {
            placeholders.push((field_name, None));
        }
    }

    placeholders
}

/// Extract doc comments from attributes
fn extract_doc_comments(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc")
                && let syn::Meta::NameValue(meta_name_value) = &attr.meta
                && let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(lit_str),
                    ..
                }) = &meta_name_value.value
            {
                return Some(lit_str.value());
            }
            None
        })
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generate example JSON representation for a struct
fn generate_example_only_parts(
    fields: &syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
    has_default: bool,
    crate_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let mut field_values = Vec::new();

    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let attrs = parse_field_prompt_attrs(&field.attrs);

        // Skip __type field - it's metadata that shouldn't be in examples
        // It's typically marked with #[serde(skip_serializing)] or #[serde(default)]
        // and won't appear in actual JSON output
        if field_name_str == "__type" {
            continue;
        }

        // Skip if marked to skip
        if attrs.skip {
            continue;
        }

        // Check if field has example attribute
        if let Some(example) = attrs.example {
            // Use the provided example value
            field_values.push(quote! {
                json_obj.insert(#field_name_str.to_string(), serde_json::Value::String(#example.to_string()));
            });
        } else if has_default {
            // Use Default value if available
            field_values.push(quote! {
                let default_value = serde_json::to_value(&default_instance.#field_name)
                    .unwrap_or(serde_json::Value::Null);
                json_obj.insert(#field_name_str.to_string(), default_value);
            });
        } else {
            // Use self's actual value
            field_values.push(quote! {
                let value = serde_json::to_value(&self.#field_name)
                    .unwrap_or(serde_json::Value::Null);
                json_obj.insert(#field_name_str.to_string(), value);
            });
        }
    }

    if has_default {
        quote! {
            {
                let default_instance = Self::default();
                let mut json_obj = serde_json::Map::new();
                #(#field_values)*
                let json_value = serde_json::Value::Object(json_obj);
                let json_str = serde_json::to_string_pretty(&json_value)
                    .unwrap_or_else(|_| "{}".to_string());
                vec![#crate_path::prompt::PromptPart::Text(json_str)]
            }
        }
    } else {
        quote! {
            {
                let mut json_obj = serde_json::Map::new();
                #(#field_values)*
                let json_value = serde_json::Value::Object(json_obj);
                let json_str = serde_json::to_string_pretty(&json_value)
                    .unwrap_or_else(|_| "{}".to_string());
                vec![#crate_path::prompt::PromptPart::Text(json_str)]
            }
        }
    }
}

/// Generate schema-only representation for a struct
fn generate_schema_only_parts(
    struct_name: &str,
    struct_docs: &str,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
    crate_path: &proc_macro2::TokenStream,
    _has_type_marker: bool,
) -> proc_macro2::TokenStream {
    let mut field_schema_parts = vec![];
    let mut nested_type_collectors = vec![];

    // Process fields to build runtime schema generation
    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let attrs = parse_field_prompt_attrs(&field.attrs);

        // Skip __type field - it's metadata that shouldn't be in the schema
        // LLMs misinterpret "__type": "string" as "output the literal string 'string'"
        // The __type field will be automatically added during deserialization via #[serde(default)]
        if field_name_str == "__type" {
            continue;
        }

        // Skip if marked to skip
        if attrs.skip {
            continue;
        }

        // Get field documentation
        let field_docs = extract_doc_comments(&field.attrs);

        // Check if this is a Vec<T> where T might implement ToPrompt
        let (is_vec, inner_type) = extract_vec_inner_type(&field.ty);

        if is_vec {
            // For Vec<T>, use TypeScript array syntax: T[]
            // Format: field_name: TypeName[];  // comment
            let comment = if !field_docs.is_empty() {
                format!("  // {}", field_docs)
            } else {
                String::new()
            };

            field_schema_parts.push(quote! {
                {
                    let type_name = stringify!(#inner_type);
                    format!("  {}: {}[];{}", #field_name_str, type_name, #comment)
                }
            });

            // Collect nested type schema if not primitive
            if let Some(inner) = inner_type
                && !is_primitive_type(inner)
            {
                nested_type_collectors.push(quote! {
                    <#inner as #crate_path::prompt::ToPrompt>::prompt_schema()
                });
            }
        } else {
            // Check if this is a custom type that implements ToPrompt (nested object)
            let field_type = &field.ty;
            let is_primitive = is_primitive_type(field_type);

            if !is_primitive {
                // For nested objects, use TypeScript type reference AND collect nested schema
                // Format: field_name: TypeName;  // comment
                let comment = if !field_docs.is_empty() {
                    format!("  // {}", field_docs)
                } else {
                    String::new()
                };

                field_schema_parts.push(quote! {
                    {
                        let type_name = stringify!(#field_type);
                        format!("  {}: {};{}", #field_name_str, type_name, #comment)
                    }
                });

                // Collect nested type schema for type definitions section
                nested_type_collectors.push(quote! {
                    <#field_type as #crate_path::prompt::ToPrompt>::prompt_schema()
                });
            } else {
                // Primitive type - use TypeScript formatting
                // Format: field_name: type;  // comment
                let type_str = format_type_for_schema(&field.ty);
                let comment = if !field_docs.is_empty() {
                    format!("  // {}", field_docs)
                } else {
                    String::new()
                };

                field_schema_parts.push(quote! {
                    format!("  {}: {};{}", #field_name_str, #type_str, #comment)
                });
            }
        }
    }

    // Build TypeScript-style type definitions with nested types first
    // Format:
    // type NestedType1 = { ... }
    //
    // type NestedType2 = { ... }
    //
    // /**
    //  * Struct description
    //  */
    // type StructName = {
    //   field1: NestedType1;  // comment1
    //   field2: NestedType2;  // comment2
    // }

    let mut header_lines = Vec::new();

    // Add JSDoc comment if struct has description
    if !struct_docs.is_empty() {
        header_lines.push("/**".to_string());
        header_lines.push(format!(" * {}", struct_docs));
        header_lines.push(" */".to_string());
    }

    // Add type definition line
    header_lines.push(format!("type {} = {{", struct_name));

    quote! {
        {
            let mut all_lines: Vec<String> = Vec::new();

            // Collect nested type definitions
            let nested_schemas: Vec<String> = vec![#(#nested_type_collectors),*];
            let mut seen_types = std::collections::HashSet::<String>::new();

            for schema in nested_schemas {
                if !schema.is_empty() {
                    // Avoid duplicates by checking if we've seen this schema
                    if seen_types.insert(schema.clone()) {
                        all_lines.push(schema);
                        all_lines.push(String::new());  // Empty line separator
                    }
                }
            }

            // Add main type definition
            let mut lines: Vec<String> = Vec::new();
            #(lines.push(#header_lines.to_string());)*
            #(lines.push(#field_schema_parts);)*
            lines.push("}".to_string());
            all_lines.push(lines.join("\n"));

            vec![#crate_path::prompt::PromptPart::Text(all_lines.join("\n"))]
        }
    }
}

/// Extract inner type from Vec<T>, returns (is_vec, inner_type)
fn extract_vec_inner_type(ty: &syn::Type) -> (bool, Option<&syn::Type>) {
    if let syn::Type::Path(type_path) = ty
        && let Some(last_segment) = type_path.path.segments.last()
        && last_segment.ident == "Vec"
        && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
        && let Some(syn::GenericArgument::Type(inner_type)) = args.args.first()
    {
        return (true, Some(inner_type));
    }
    (false, None)
}

/// Check if a type is a primitive type (should not be expanded as nested object)
fn is_primitive_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty
        && let Some(last_segment) = type_path.path.segments.last()
    {
        let type_name = last_segment.ident.to_string();
        matches!(
            type_name.as_str(),
            "String"
                | "str"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f32"
                | "f64"
                | "bool"
                | "Vec"
                | "Option"
                | "HashMap"
                | "BTreeMap"
                | "HashSet"
                | "BTreeSet"
        )
    } else {
        // References, arrays, etc. are considered primitive for now
        true
    }
}

/// Format a type for schema representation
fn format_type_for_schema(ty: &syn::Type) -> String {
    // Simple type formatting - can be enhanced
    match ty {
        syn::Type::Path(type_path) => {
            let path = &type_path.path;
            if let Some(last_segment) = path.segments.last() {
                let type_name = last_segment.ident.to_string();

                // Handle Option<T>
                if type_name == "Option"
                    && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                    && let Some(syn::GenericArgument::Type(inner_type)) = args.args.first()
                {
                    return format!("{} | null", format_type_for_schema(inner_type));
                }

                // Map common types
                match type_name.as_str() {
                    "String" | "str" => "string".to_string(),
                    "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32"
                    | "u64" | "u128" | "usize" => "number".to_string(),
                    "f32" | "f64" => "number".to_string(),
                    "bool" => "boolean".to_string(),
                    "Vec" => {
                        if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                            && let Some(syn::GenericArgument::Type(inner_type)) = args.args.first()
                        {
                            return format!("{}[]", format_type_for_schema(inner_type));
                        }
                        "array".to_string()
                    }
                    _ => type_name.to_lowercase(),
                }
            } else {
                "unknown".to_string()
            }
        }
        _ => "unknown".to_string(),
    }
}

/// Result of parsing prompt attributes on a variant
#[derive(Default)]
struct PromptAttributes {
    skip: bool,
    rename: Option<String>,
    description: Option<String>,
}

/// Parse #[prompt(...)] attributes on enum variant
/// Collects all prompt attributes (rename, description, skip) from multiple attributes
fn parse_prompt_attributes(attrs: &[syn::Attribute]) -> PromptAttributes {
    let mut result = PromptAttributes::default();

    for attr in attrs {
        if attr.path().is_ident("prompt") {
            // Check for #[prompt(rename = "...")], #[prompt(description = "...")], etc.
            if let Ok(meta_list) = attr.meta.require_list() {
                // Try parsing as key-value pairs
                if let Ok(metas) =
                    meta_list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                {
                    for meta in metas {
                        if let Meta::NameValue(nv) = meta {
                            if nv.path.is_ident("rename") {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value
                                {
                                    result.rename = Some(lit_str.value());
                                }
                            } else if nv.path.is_ident("description")
                                && let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value
                            {
                                result.description = Some(lit_str.value());
                            }
                        } else if let Meta::Path(path) = meta
                            && path.is_ident("skip")
                        {
                            result.skip = true;
                        }
                    }
                }

                // Fallback: check for simple #[prompt(skip)]
                let tokens_str = meta_list.tokens.to_string();
                if tokens_str == "skip" {
                    result.skip = true;
                }
            }

            // Check for #[prompt("description")] (shorthand)
            if let Ok(lit_str) = attr.parse_args::<syn::LitStr>() {
                result.description = Some(lit_str.value());
            }
        }
    }
    result
}

/// Generate example value for a type in JSON format
fn generate_example_value_for_type(type_str: &str) -> String {
    match type_str {
        "string" => "\"example\"".to_string(),
        "number" => "0".to_string(),
        "boolean" => "false".to_string(),
        s if s.ends_with("[]") => "[]".to_string(),
        s if s.contains("|") => {
            // For union types like "string | null", use the first type
            let first_type = s.split('|').next().unwrap().trim();
            generate_example_value_for_type(first_type)
        }
        _ => "null".to_string(),
    }
}

/// Parse #[serde(rename = "...")] attribute on enum variant
fn parse_serde_variant_rename(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("serde")
            && let Ok(meta_list) = attr.meta.require_list()
            && let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        {
            for meta in metas {
                if let Meta::NameValue(nv) = meta
                    && nv.path.is_ident("rename")
                    && let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = nv.value
                {
                    return Some(lit_str.value());
                }
            }
        }
    }
    None
}

/// Serde rename rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenameRule {
    #[allow(dead_code)]
    None,
    LowerCase,
    UpperCase,
    PascalCase,
    CamelCase,
    SnakeCase,
    ScreamingSnakeCase,
    KebabCase,
    ScreamingKebabCase,
}

impl RenameRule {
    /// Parse from serde rename_all string
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "lowercase" => Some(Self::LowerCase),
            "UPPERCASE" => Some(Self::UpperCase),
            "PascalCase" => Some(Self::PascalCase),
            "camelCase" => Some(Self::CamelCase),
            "snake_case" => Some(Self::SnakeCase),
            "SCREAMING_SNAKE_CASE" => Some(Self::ScreamingSnakeCase),
            "kebab-case" => Some(Self::KebabCase),
            "SCREAMING-KEBAB-CASE" => Some(Self::ScreamingKebabCase),
            _ => None,
        }
    }

    /// Apply rename rule to a variant name
    fn apply(&self, name: &str) -> String {
        match self {
            Self::None => name.to_string(),
            Self::LowerCase => name.to_lowercase(),
            Self::UpperCase => name.to_uppercase(),
            Self::PascalCase => name.to_string(), // PascalCase is the Rust default
            Self::CamelCase => {
                // Convert PascalCase to camelCase
                let mut chars = name.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_lowercase().chain(chars).collect(),
                }
            }
            Self::SnakeCase => {
                // Convert PascalCase to snake_case
                let mut result = String::new();
                for (i, ch) in name.chars().enumerate() {
                    if ch.is_uppercase() && i > 0 {
                        result.push('_');
                    }
                    result.push(ch.to_lowercase().next().unwrap());
                }
                result
            }
            Self::ScreamingSnakeCase => {
                // Convert PascalCase to SCREAMING_SNAKE_CASE
                let mut result = String::new();
                for (i, ch) in name.chars().enumerate() {
                    if ch.is_uppercase() && i > 0 {
                        result.push('_');
                    }
                    result.push(ch.to_uppercase().next().unwrap());
                }
                result
            }
            Self::KebabCase => {
                // Convert PascalCase to kebab-case
                let mut result = String::new();
                for (i, ch) in name.chars().enumerate() {
                    if ch.is_uppercase() && i > 0 {
                        result.push('-');
                    }
                    result.push(ch.to_lowercase().next().unwrap());
                }
                result
            }
            Self::ScreamingKebabCase => {
                // Convert PascalCase to SCREAMING-KEBAB-CASE
                let mut result = String::new();
                for (i, ch) in name.chars().enumerate() {
                    if ch.is_uppercase() && i > 0 {
                        result.push('-');
                    }
                    result.push(ch.to_uppercase().next().unwrap());
                }
                result
            }
        }
    }
}

/// Parse #[serde(rename_all = "...")] attribute on enum/struct
fn parse_serde_rename_all(attrs: &[syn::Attribute]) -> Option<RenameRule> {
    for attr in attrs {
        if attr.path().is_ident("serde")
            && let Ok(meta_list) = attr.meta.require_list()
        {
            // Parse the tokens inside the parentheses
            if let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            {
                for meta in metas {
                    if let Meta::NameValue(nv) = meta
                        && nv.path.is_ident("rename_all")
                        && let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                    {
                        return RenameRule::from_str(&lit_str.value());
                    }
                }
            }
        }
    }
    None
}

/// Parse #[serde(tag = "...")] attribute on enum
/// Returns Some(tag_name) if present, None otherwise
fn parse_serde_tag(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("serde")
            && let Ok(meta_list) = attr.meta.require_list()
        {
            // Parse the tokens inside the parentheses
            if let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            {
                for meta in metas {
                    if let Meta::NameValue(nv) = meta
                        && nv.path.is_ident("tag")
                        && let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                    {
                        return Some(lit_str.value());
                    }
                }
            }
        }
    }
    None
}

/// Parse #[serde(untagged)] attribute on enum
/// Returns true if the enum is untagged
fn parse_serde_untagged(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("serde")
            && let Ok(meta_list) = attr.meta.require_list()
        {
            // Parse the tokens inside the parentheses
            if let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            {
                for meta in metas {
                    if let Meta::Path(path) = meta
                        && path.is_ident("untagged")
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Parsed field-level prompt attributes
#[derive(Debug, Default)]
struct FieldPromptAttrs {
    skip: bool,
    rename: Option<String>,
    format_with: Option<String>,
    image: bool,
    example: Option<String>,
}

/// Parse #[prompt(...)] attributes for struct fields
fn parse_field_prompt_attrs(attrs: &[syn::Attribute]) -> FieldPromptAttrs {
    let mut result = FieldPromptAttrs::default();

    for attr in attrs {
        if attr.path().is_ident("prompt") {
            // Try to parse as meta list #[prompt(key = value, ...)]
            if let Ok(meta_list) = attr.meta.require_list() {
                // Parse the tokens inside the parentheses
                if let Ok(metas) =
                    meta_list.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
                {
                    for meta in metas {
                        match meta {
                            Meta::Path(path) if path.is_ident("skip") => {
                                result.skip = true;
                            }
                            Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value
                                {
                                    result.rename = Some(lit_str.value());
                                }
                            }
                            Meta::NameValue(nv) if nv.path.is_ident("format_with") => {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value
                                {
                                    result.format_with = Some(lit_str.value());
                                }
                            }
                            Meta::Path(path) if path.is_ident("image") => {
                                result.image = true;
                            }
                            Meta::NameValue(nv) if nv.path.is_ident("example") => {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value
                                {
                                    result.example = Some(lit_str.value());
                                }
                            }
                            _ => {}
                        }
                    }
                } else if meta_list.tokens.to_string() == "skip" {
                    // Handle simple #[prompt(skip)] case
                    result.skip = true;
                } else if meta_list.tokens.to_string() == "image" {
                    // Handle simple #[prompt(image)] case
                    result.image = true;
                }
            }
        }
    }

    result
}

/// Derives the `ToPrompt` trait for a struct or enum.
///
/// This macro provides two main functionalities depending on the type.
///
/// ## For Structs
///
/// It can generate a prompt based on a template string or by creating a key-value representation of the struct's fields.
///
/// ### Template-based Prompt
///
/// Use the `#[prompt(template = "...")]` attribute to provide a `minijinja` template. The struct fields will be available as variables in the template. The struct must also derive `serde::Serialize`.
///
/// ```rust,ignore
/// #[derive(ToPrompt, Serialize)]
/// #[prompt(template = "User {{ name }} is a {{ role }}.")]
/// struct UserProfile {
///     name: &'static str,
///     role: &'static str,
/// }
/// ```
///
/// ### Tip: Handling Special Characters in Templates
///
/// When using raw string literals (e.g., `r#"..."#`) for your templates, be aware of a potential parsing issue if your template content includes the `#` character. To avoid this, use a different number of `#` symbols for the raw string delimiter.
///
/// **Problematic Example:**
/// ```rust,ignore
/// // This might fail to parse correctly
/// #[prompt(template = r#"{"color": "#FFFFFF"}"#)]
/// struct Color { /* ... */ }
/// ```
///
/// **Solution:**
/// ```rust,ignore
/// // Use r##"..."## to avoid ambiguity with the inner '#'
/// #[prompt(template = r##"{"color": "#FFFFFF"}"##)]
/// struct Color { /* ... */ }
/// ```
///
/// ## For Enums
///
/// For enums, the macro generates a descriptive prompt based on doc comments and attributes, outlining the available variants. See the documentation on the `ToPrompt` trait for more details.
#[proc_macro_derive(ToPrompt, attributes(prompt))]
pub fn to_prompt_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            // Even when it's the same crate, use absolute path to support examples/tests/bins
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Check if this is a struct or enum
    match &input.data {
        Data::Enum(data_enum) => {
            // For enums, generate prompt from doc comments
            let enum_name = &input.ident;
            let enum_docs = extract_doc_comments(&input.attrs);

            // Check for serde tagging strategy attributes
            let serde_tag = parse_serde_tag(&input.attrs);
            let is_internally_tagged = serde_tag.is_some();
            let is_untagged = parse_serde_untagged(&input.attrs);

            // Check for #[serde(rename_all = "...")] attribute
            let rename_rule = parse_serde_rename_all(&input.attrs);

            // Generate TypeScript-style union type with descriptions
            // Format:
            // /**
            //  * Enum description
            //  */
            // type EnumName =
            //   | "Variant1"  // Description1
            //   | "Variant2"  // Description2
            //   | "Variant3"; // Description3
            //
            // Example value: "Variant1"

            let mut variant_lines = Vec::new();
            let mut first_variant_name = None;

            // Collect examples for each variant type
            let mut example_unit: Option<String> = None;
            let mut example_struct: Option<String> = None;
            let mut example_tuple: Option<String> = None;

            for variant in &data_enum.variants {
                let variant_name = &variant.ident;
                let variant_name_str = variant_name.to_string();

                // Parse prompt attributes
                let prompt_attrs = parse_prompt_attributes(&variant.attrs);

                // Skip if marked with #[prompt(skip)]
                if prompt_attrs.skip {
                    continue;
                }

                // Determine variant value with priority:
                // 1. #[prompt(rename = "...")]
                // 2. #[serde(rename = "...")]
                // 3. #[serde(rename_all = "...")] rule
                // 4. Default (variant name as-is)
                let variant_value = if let Some(prompt_rename) = &prompt_attrs.rename {
                    prompt_rename.clone()
                } else if let Some(serde_rename) = parse_serde_variant_rename(&variant.attrs) {
                    serde_rename
                } else if let Some(rule) = rename_rule {
                    rule.apply(&variant_name_str)
                } else {
                    variant_name_str.clone()
                };

                // Check variant type: Unit, Struct, or Tuple
                let variant_line = match &variant.fields {
                    syn::Fields::Unit => {
                        // Collect example for Unit variant (if first one)
                        if example_unit.is_none() {
                            example_unit = Some(format!("\"{}\"", variant_value));
                        }

                        // Unit variant: "VariantName"
                        if let Some(desc) = &prompt_attrs.description {
                            format!("  | \"{}\"  // {}", variant_value, desc)
                        } else {
                            let docs = extract_doc_comments(&variant.attrs);
                            if !docs.is_empty() {
                                format!("  | \"{}\"  // {}", variant_value, docs)
                            } else {
                                format!("  | \"{}\"", variant_value)
                            }
                        }
                    }
                    syn::Fields::Named(fields) => {
                        let mut field_parts = Vec::new();
                        let mut example_field_parts = Vec::new();

                        // For Internally Tagged, include the tag field first
                        if is_internally_tagged && let Some(tag_name) = &serde_tag {
                            field_parts.push(format!("{}: \"{}\"", tag_name, variant_value));
                            example_field_parts
                                .push(format!("{}: \"{}\"", tag_name, variant_value));
                        }

                        for field in &fields.named {
                            let field_name = field.ident.as_ref().unwrap().to_string();
                            let field_type = format_type_for_schema(&field.ty);
                            field_parts.push(format!("{}: {}", field_name, field_type.clone()));

                            // Generate example value for this field
                            let example_value = generate_example_value_for_type(&field_type);
                            example_field_parts.push(format!("{}: {}", field_name, example_value));
                        }

                        let field_str = field_parts.join(", ");
                        let example_field_str = example_field_parts.join(", ");

                        // Collect example for Struct variant (if first one)
                        if example_struct.is_none() {
                            if is_untagged || is_internally_tagged {
                                example_struct = Some(format!("{{ {} }}", example_field_str));
                            } else {
                                example_struct = Some(format!(
                                    "{{ \"{}\": {{ {} }} }}",
                                    variant_value, example_field_str
                                ));
                            }
                        }

                        let comment = if let Some(desc) = &prompt_attrs.description {
                            format!("  // {}", desc)
                        } else {
                            let docs = extract_doc_comments(&variant.attrs);
                            if !docs.is_empty() {
                                format!("  // {}", docs)
                            } else if is_untagged {
                                // For untagged enums, add variant name as comment since it's not in the type
                                format!("  // {}", variant_value)
                            } else {
                                String::new()
                            }
                        };

                        if is_untagged {
                            // Untagged format: bare object { field1: Type1, ... }
                            format!("  | {{ {} }}{}", field_str, comment)
                        } else if is_internally_tagged {
                            // Internally Tagged format: { type: "VariantName", field1: Type1, ... }
                            format!("  | {{ {} }}{}", field_str, comment)
                        } else {
                            // Externally Tagged format (default): { "VariantName": { field1: Type1, ... } }
                            format!(
                                "  | {{ \"{}\": {{ {} }} }}{}",
                                variant_value, field_str, comment
                            )
                        }
                    }
                    syn::Fields::Unnamed(fields) => {
                        let field_types: Vec<String> = fields
                            .unnamed
                            .iter()
                            .map(|f| format_type_for_schema(&f.ty))
                            .collect();

                        let tuple_str = field_types.join(", ");

                        // Generate example values for tuple elements
                        let example_values: Vec<String> = field_types
                            .iter()
                            .map(|type_str| generate_example_value_for_type(type_str))
                            .collect();
                        let example_tuple_str = example_values.join(", ");

                        // Collect example for Tuple variant (if first one)
                        if example_tuple.is_none() {
                            if is_untagged || is_internally_tagged {
                                example_tuple = Some(format!("[{}]", example_tuple_str));
                            } else {
                                example_tuple = Some(format!(
                                    "{{ \"{}\": [{}] }}",
                                    variant_value, example_tuple_str
                                ));
                            }
                        }

                        let comment = if let Some(desc) = &prompt_attrs.description {
                            format!("  // {}", desc)
                        } else {
                            let docs = extract_doc_comments(&variant.attrs);
                            if !docs.is_empty() {
                                format!("  // {}", docs)
                            } else if is_untagged {
                                // For untagged enums, add variant name as comment since it's not in the type
                                format!("  // {}", variant_value)
                            } else {
                                String::new()
                            }
                        };

                        if is_untagged || is_internally_tagged {
                            // Untagged or Internally Tagged: bare array [Type1, Type2, ...]
                            // (Internally Tagged enums don't support tuple variants well)
                            format!("  | [{}]{}", tuple_str, comment)
                        } else {
                            // Externally Tagged format (default): { "VariantName": [tuple elements] }
                            format!(
                                "  | {{ \"{}\": [{}] }}{}",
                                variant_value, tuple_str, comment
                            )
                        }
                    }
                };

                variant_lines.push(variant_line);

                if first_variant_name.is_none() {
                    first_variant_name = Some(variant_value);
                }
            }

            // Build complete TypeScript-style schema
            let mut lines = Vec::new();

            // Add JSDoc comment if enum has description
            if !enum_docs.is_empty() {
                lines.push("/**".to_string());
                lines.push(format!(" * {}", enum_docs));
                lines.push(" */".to_string());
            }

            // Add type definition header
            lines.push(format!("type {} =", enum_name));

            // Add all variant lines
            for line in &variant_lines {
                lines.push(line.clone());
            }

            // Add semicolon to last variant
            if let Some(last) = lines.last_mut()
                && !last.ends_with(';')
            {
                last.push(';');
            }

            // Add example values for different variant types
            let mut examples = Vec::new();
            if let Some(ex) = example_unit {
                examples.push(ex);
            }
            if let Some(ex) = example_struct {
                examples.push(ex);
            }
            if let Some(ex) = example_tuple {
                examples.push(ex);
            }

            if !examples.is_empty() {
                lines.push("".to_string()); // Empty line
                if examples.len() == 1 {
                    lines.push(format!("Example value: {}", examples[0]));
                } else {
                    lines.push("Example values:".to_string());
                    for ex in examples {
                        lines.push(format!("  {}", ex));
                    }
                }
            }

            let prompt_string = lines.join("\n");
            let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

            // Generate match arms for instance-level to_prompt()
            let mut match_arms = Vec::new();
            for variant in &data_enum.variants {
                let variant_name = &variant.ident;
                let variant_name_str = variant_name.to_string();

                // Parse prompt attributes
                let prompt_attrs = parse_prompt_attributes(&variant.attrs);

                // Determine variant value with same priority as schema generation:
                // 1. #[prompt(rename = "...")]
                // 2. #[serde(rename = "...")]
                // 3. #[serde(rename_all = "...")] rule
                // 4. Default (variant name as-is)
                let variant_value = if let Some(prompt_rename) = &prompt_attrs.rename {
                    prompt_rename.clone()
                } else if let Some(serde_rename) = parse_serde_variant_rename(&variant.attrs) {
                    serde_rename
                } else if let Some(rule) = rename_rule {
                    rule.apply(&variant_name_str)
                } else {
                    variant_name_str.clone()
                };

                // Generate match arm based on variant type
                match &variant.fields {
                    syn::Fields::Unit => {
                        // Unit variant - existing behavior
                        if prompt_attrs.skip {
                            match_arms.push(quote! {
                                Self::#variant_name => stringify!(#variant_name).to_string()
                            });
                        } else if let Some(desc) = &prompt_attrs.description {
                            match_arms.push(quote! {
                                Self::#variant_name => format!("{}: {}", #variant_value, #desc)
                            });
                        } else {
                            let variant_docs = extract_doc_comments(&variant.attrs);
                            if !variant_docs.is_empty() {
                                match_arms.push(quote! {
                                    Self::#variant_name => format!("{}: {}", #variant_value, #variant_docs)
                                });
                            } else {
                                match_arms.push(quote! {
                                    Self::#variant_name => #variant_value.to_string()
                                });
                            }
                        }
                    }
                    syn::Fields::Named(fields) => {
                        // Struct variant - serialize fields to JSON-like string
                        let field_bindings: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect();

                        let field_displays: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| {
                                let field_name = f.ident.as_ref().unwrap();
                                let field_name_str = field_name.to_string();
                                quote! {
                                    format!("{}: {:?}", #field_name_str, #field_name)
                                }
                            })
                            .collect();

                        let doc_or_desc = if let Some(desc) = &prompt_attrs.description {
                            desc.clone()
                        } else {
                            let docs = extract_doc_comments(&variant.attrs);
                            if !docs.is_empty() {
                                docs
                            } else {
                                String::new()
                            }
                        };

                        if doc_or_desc.is_empty() {
                            match_arms.push(quote! {
                                Self::#variant_name { #(#field_bindings),* } => {
                                    let fields = vec![#(#field_displays),*];
                                    format!("{} {{ {} }}", #variant_value, fields.join(", "))
                                }
                            });
                        } else {
                            match_arms.push(quote! {
                                Self::#variant_name { #(#field_bindings),* } => {
                                    let fields = vec![#(#field_displays),*];
                                    format!("{}: {} {{ {} }}", #variant_value, #doc_or_desc, fields.join(", "))
                                }
                            });
                        }
                    }
                    syn::Fields::Unnamed(fields) => {
                        // Tuple variant - bind fields and display them
                        let field_count = fields.unnamed.len();
                        let field_bindings: Vec<_> = (0..field_count)
                            .map(|i| {
                                syn::Ident::new(
                                    &format!("field{}", i),
                                    proc_macro2::Span::call_site(),
                                )
                            })
                            .collect();

                        let field_displays: Vec<_> = field_bindings
                            .iter()
                            .map(|field_name| {
                                quote! {
                                    format!("{:?}", #field_name)
                                }
                            })
                            .collect();

                        let doc_or_desc = if let Some(desc) = &prompt_attrs.description {
                            desc.clone()
                        } else {
                            let docs = extract_doc_comments(&variant.attrs);
                            if !docs.is_empty() {
                                docs
                            } else {
                                String::new()
                            }
                        };

                        if doc_or_desc.is_empty() {
                            match_arms.push(quote! {
                                Self::#variant_name(#(#field_bindings),*) => {
                                    let fields = vec![#(#field_displays),*];
                                    format!("{}({})", #variant_value, fields.join(", "))
                                }
                            });
                        } else {
                            match_arms.push(quote! {
                                Self::#variant_name(#(#field_bindings),*) => {
                                    let fields = vec![#(#field_displays),*];
                                    format!("{}: {}({})", #variant_value, #doc_or_desc, fields.join(", "))
                                }
                            });
                        }
                    }
                }
            }

            let to_prompt_impl = if match_arms.is_empty() {
                // Empty enum: no variants to match
                quote! {
                    fn to_prompt(&self) -> String {
                        match *self {}
                    }
                }
            } else {
                quote! {
                    fn to_prompt(&self) -> String {
                        match self {
                            #(#match_arms),*
                        }
                    }
                }
            };

            let expanded = quote! {
                impl #impl_generics #crate_path::prompt::ToPrompt for #enum_name #ty_generics #where_clause {
                    fn to_prompt_parts(&self) -> Vec<#crate_path::prompt::PromptPart> {
                        vec![#crate_path::prompt::PromptPart::Text(self.to_prompt())]
                    }

                    #to_prompt_impl

                    fn prompt_schema() -> String {
                        #prompt_string.to_string()
                    }
                }
            };

            TokenStream::from(expanded)
        }
        Data::Struct(data_struct) => {
            // Parse struct-level prompt attributes for template, template_file, mode, and validate
            let mut template_attr = None;
            let mut template_file_attr = None;
            let mut mode_attr = None;
            let mut validate_attr = false;
            let mut type_marker_attr = false;

            for attr in &input.attrs {
                if attr.path().is_ident("prompt") {
                    // Try to parse the attribute arguments
                    if let Ok(metas) =
                        attr.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
                    {
                        for meta in metas {
                            match meta {
                                Meta::NameValue(nv) if nv.path.is_ident("template") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value
                                        && let syn::Lit::Str(lit_str) = expr_lit.lit
                                    {
                                        template_attr = Some(lit_str.value());
                                    }
                                }
                                Meta::NameValue(nv) if nv.path.is_ident("template_file") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value
                                        && let syn::Lit::Str(lit_str) = expr_lit.lit
                                    {
                                        template_file_attr = Some(lit_str.value());
                                    }
                                }
                                Meta::NameValue(nv) if nv.path.is_ident("mode") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value
                                        && let syn::Lit::Str(lit_str) = expr_lit.lit
                                    {
                                        mode_attr = Some(lit_str.value());
                                    }
                                }
                                Meta::NameValue(nv) if nv.path.is_ident("validate") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value
                                        && let syn::Lit::Bool(lit_bool) = expr_lit.lit
                                    {
                                        validate_attr = lit_bool.value();
                                    }
                                }
                                Meta::NameValue(nv) if nv.path.is_ident("type_marker") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value
                                        && let syn::Lit::Bool(lit_bool) = expr_lit.lit
                                    {
                                        type_marker_attr = lit_bool.value();
                                    }
                                }
                                Meta::Path(path) if path.is_ident("type_marker") => {
                                    // Support both #[prompt(type_marker)] and #[prompt(type_marker = true)]
                                    type_marker_attr = true;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Check for mutual exclusivity between template and template_file
            if template_attr.is_some() && template_file_attr.is_some() {
                return syn::Error::new(
                    input.ident.span(),
                    "The `template` and `template_file` attributes are mutually exclusive. Please use only one.",
                ).to_compile_error().into();
            }

            // Load template from file if template_file is specified
            let template_str = if let Some(file_path) = template_file_attr {
                // Try multiple strategies to find the template file
                // This is necessary to support both normal compilation and trybuild tests

                let mut full_path = None;

                // Strategy 1: Try relative to CARGO_MANIFEST_DIR (normal compilation)
                if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                    // Check if this is a trybuild temporary directory
                    let is_trybuild = manifest_dir.contains("target/tests/trybuild");

                    if !is_trybuild {
                        // Normal compilation - use CARGO_MANIFEST_DIR directly
                        let candidate = std::path::Path::new(&manifest_dir).join(&file_path);
                        if candidate.exists() {
                            full_path = Some(candidate);
                        }
                    } else {
                        // For trybuild, we need to find the original source directory
                        // The manifest_dir looks like: .../target/tests/trybuild/llm-toolkit-macros
                        // We need to get back to the original llm-toolkit-macros source directory

                        // Extract the workspace root from the path
                        if let Some(target_pos) = manifest_dir.find("/target/tests/trybuild") {
                            let workspace_root = &manifest_dir[..target_pos];
                            // Now construct the path to the original llm-toolkit-macros source
                            let original_macros_dir = std::path::Path::new(workspace_root)
                                .join("crates")
                                .join("llm-toolkit-macros");

                            let candidate = original_macros_dir.join(&file_path);
                            if candidate.exists() {
                                full_path = Some(candidate);
                            }
                        }
                    }
                }

                // Strategy 2: Try as an absolute path or relative to current directory
                if full_path.is_none() {
                    let candidate = std::path::Path::new(&file_path).to_path_buf();
                    if candidate.exists() {
                        full_path = Some(candidate);
                    }
                }

                // Strategy 3: For trybuild tests - try to find the file by looking in parent directories
                // This handles the case where trybuild creates a temporary project
                if full_path.is_none()
                    && let Ok(current_dir) = std::env::current_dir()
                {
                    let mut search_dir = current_dir.as_path();
                    // Search up to 10 levels up
                    for _ in 0..10 {
                        // Try from the llm-toolkit-macros directory
                        let macros_dir = search_dir.join("crates/llm-toolkit-macros");
                        if macros_dir.exists() {
                            let candidate = macros_dir.join(&file_path);
                            if candidate.exists() {
                                full_path = Some(candidate);
                                break;
                            }
                        }
                        // Try directly
                        let candidate = search_dir.join(&file_path);
                        if candidate.exists() {
                            full_path = Some(candidate);
                            break;
                        }
                        if let Some(parent) = search_dir.parent() {
                            search_dir = parent;
                        } else {
                            break;
                        }
                    }
                }

                // Validate file existence at compile time
                if full_path.is_none() {
                    // Build helpful error message with search locations
                    let mut error_msg = format!(
                        "Template file '{}' not found at compile time.\n\nSearched in:",
                        file_path
                    );

                    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                        let candidate = std::path::Path::new(&manifest_dir).join(&file_path);
                        error_msg.push_str(&format!("\n  - {}", candidate.display()));
                    }

                    if let Ok(current_dir) = std::env::current_dir() {
                        let candidate = current_dir.join(&file_path);
                        error_msg.push_str(&format!("\n  - {}", candidate.display()));
                    }

                    error_msg.push_str("\n\nPlease ensure:");
                    error_msg.push_str("\n  1. The template file exists");
                    error_msg.push_str("\n  2. The path is relative to CARGO_MANIFEST_DIR");
                    error_msg.push_str("\n  3. There are no typos in the path");

                    return syn::Error::new(input.ident.span(), error_msg)
                        .to_compile_error()
                        .into();
                }

                let final_path = full_path.unwrap();

                // Read the file at compile time
                match std::fs::read_to_string(&final_path) {
                    Ok(content) => Some(content),
                    Err(e) => {
                        return syn::Error::new(
                            input.ident.span(),
                            format!(
                                "Failed to read template file '{}': {}\n\nPath resolved to: {}",
                                file_path,
                                e,
                                final_path.display()
                            ),
                        )
                        .to_compile_error()
                        .into();
                    }
                }
            } else {
                template_attr
            };

            // Perform validation if requested
            if validate_attr && let Some(template) = &template_str {
                // Validate Jinja syntax
                let mut env = minijinja::Environment::new();
                if let Err(e) = env.add_template("validation", template) {
                    // Generate a compile warning using deprecated const hack
                    let warning_msg =
                        format!("Template validation warning: Invalid Jinja syntax - {}", e);
                    let warning_ident = syn::Ident::new(
                        "TEMPLATE_VALIDATION_WARNING",
                        proc_macro2::Span::call_site(),
                    );
                    let _warning_tokens = quote! {
                        #[deprecated(note = #warning_msg)]
                        const #warning_ident: () = ();
                        let _ = #warning_ident;
                    };
                    // We'll inject this warning into the generated code
                    eprintln!("cargo:warning={}", warning_msg);
                }

                // Extract variables from template and check against struct fields
                let fields = if let syn::Fields::Named(fields) = &data_struct.fields {
                    &fields.named
                } else {
                    panic!("Template validation is only supported for structs with named fields.");
                };

                let field_names: std::collections::HashSet<String> = fields
                    .iter()
                    .filter_map(|f| f.ident.as_ref().map(|i| i.to_string()))
                    .collect();

                // Parse template placeholders
                let placeholders = parse_template_placeholders_with_mode(template);

                for (placeholder_name, _mode) in &placeholders {
                    if placeholder_name != "self" && !field_names.contains(placeholder_name) {
                        let warning_msg = format!(
                            "Template validation warning: Variable '{}' used in template but not found in struct fields",
                            placeholder_name
                        );
                        eprintln!("cargo:warning={}", warning_msg);
                    }
                }
            }

            let name = input.ident;
            let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

            // Extract struct name and doc comment for use in schema generation
            let struct_docs = extract_doc_comments(&input.attrs);

            // Check if this is a mode-based struct (mode attribute present)
            let is_mode_based =
                mode_attr.is_some() || (template_str.is_none() && struct_docs.contains("mode"));

            let expanded = if is_mode_based || mode_attr.is_some() {
                // Mode-based generation: support schema_only, example_only, full
                let fields = if let syn::Fields::Named(fields) = &data_struct.fields {
                    &fields.named
                } else {
                    panic!(
                        "Mode-based prompt generation is only supported for structs with named fields."
                    );
                };

                let struct_name_str = name.to_string();

                // Check if struct derives Default
                let has_default = input.attrs.iter().any(|attr| {
                    if attr.path().is_ident("derive")
                        && let Ok(meta_list) = attr.meta.require_list()
                    {
                        let tokens_str = meta_list.tokens.to_string();
                        tokens_str.contains("Default")
                    } else {
                        false
                    }
                });

                // Note: type_marker_attr is used as a marker/flag indicating this struct uses the TypeMarker pattern
                // When type_marker is set (via #[prompt(type_marker)]), it indicates:
                // - This struct is used for type-based retrieval in Orchestrator
                // - The __type field must be manually defined by the user (for custom configurations)
                // - The __type field will be automatically excluded from LLM schema (see Line 154)
                //
                // For standard cases, users should use #[type_marker] attribute macro instead,
                // which automatically adds the __type field.

                // Generate schema-only parts (type_marker_attr comes from prompt attribute parsing above)
                let schema_parts = generate_schema_only_parts(
                    &struct_name_str,
                    &struct_docs,
                    fields,
                    &crate_path,
                    type_marker_attr,
                );

                // Generate example parts
                let example_parts = generate_example_only_parts(fields, has_default, &crate_path);

                quote! {
                    impl #impl_generics #crate_path::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt_parts_with_mode(&self, mode: &str) -> Vec<#crate_path::prompt::PromptPart> {
                            match mode {
                                "schema_only" => #schema_parts,
                                "example_only" => #example_parts,
                                "full" | _ => {
                                    // Combine schema and example
                                    let mut parts = Vec::new();

                                    // Add schema
                                    let schema_parts = #schema_parts;
                                    parts.extend(schema_parts);

                                    // Add separator and example header
                                    parts.push(#crate_path::prompt::PromptPart::Text("\n### Example".to_string()));
                                    parts.push(#crate_path::prompt::PromptPart::Text(
                                        format!("Here is an example of a valid `{}` object:", #struct_name_str)
                                    ));

                                    // Add example
                                    let example_parts = #example_parts;
                                    parts.extend(example_parts);

                                    parts
                                }
                            }
                        }

                        fn to_prompt_parts(&self) -> Vec<#crate_path::prompt::PromptPart> {
                            self.to_prompt_parts_with_mode("full")
                        }

                        fn to_prompt(&self) -> String {
                            self.to_prompt_parts()
                                .into_iter()
                                .filter_map(|part| match part {
                                    #crate_path::prompt::PromptPart::Text(text) => Some(text),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }

                        fn prompt_schema() -> String {
                            use std::sync::OnceLock;
                            static SCHEMA_CACHE: OnceLock<String> = OnceLock::new();

                            SCHEMA_CACHE.get_or_init(|| {
                                let schema_parts = #schema_parts;
                                schema_parts
                                    .into_iter()
                                    .filter_map(|part| match part {
                                        #crate_path::prompt::PromptPart::Text(text) => Some(text),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }).clone()
                        }
                    }
                }
            } else if let Some(template) = template_str {
                // Use template-based approach if template is provided
                // Collect image fields separately for to_prompt_parts()
                let fields = if let syn::Fields::Named(fields) = &data_struct.fields {
                    &fields.named
                } else {
                    panic!(
                        "Template prompt generation is only supported for structs with named fields."
                    );
                };

                // Parse template to detect mode syntax
                let placeholders = parse_template_placeholders_with_mode(&template);
                // Only use custom mode processing if template actually contains :mode syntax
                let has_mode_syntax = placeholders.iter().any(|(field_name, mode)| {
                    mode.is_some()
                        && fields
                            .iter()
                            .any(|f| f.ident.as_ref().unwrap() == field_name)
                });

                let mut image_field_parts = Vec::new();
                for f in fields.iter() {
                    let field_name = f.ident.as_ref().unwrap();
                    let attrs = parse_field_prompt_attrs(&f.attrs);

                    if attrs.image {
                        // This field is marked as an image
                        image_field_parts.push(quote! {
                            parts.extend(self.#field_name.to_prompt_parts());
                        });
                    }
                }

                // Generate appropriate code based on whether mode syntax is used
                if has_mode_syntax {
                    // Build custom context for fields with mode specifications
                    let mut context_fields = Vec::new();
                    let mut modified_template = template.clone();

                    // Process each placeholder with mode
                    for (field_name, mode_opt) in &placeholders {
                        if let Some(mode) = mode_opt {
                            // Create a unique key for this field:mode combination
                            let unique_key = format!("{}__{}", field_name, mode);

                            // Replace {{ field:mode }} with {{ field__mode }} in template
                            let pattern = format!("{{{{ {}:{} }}}}", field_name, mode);
                            let replacement = format!("{{{{ {} }}}}", unique_key);
                            modified_template = modified_template.replace(&pattern, &replacement);

                            // Find the corresponding field
                            let field_ident =
                                syn::Ident::new(field_name, proc_macro2::Span::call_site());

                            // Add to context with mode specification
                            context_fields.push(quote! {
                                context.insert(
                                    #unique_key.to_string(),
                                    minijinja::Value::from(self.#field_ident.to_prompt_with_mode(#mode))
                                );
                            });
                        }
                    }

                    // Add individual fields via direct access (for non-mode fields)
                    for field in fields.iter() {
                        let field_name = field.ident.as_ref().unwrap();
                        let field_name_str = field_name.to_string();

                        // Skip if this field already has a mode-specific entry
                        let has_mode_entry = placeholders
                            .iter()
                            .any(|(name, mode)| name == &field_name_str && mode.is_some());

                        if !has_mode_entry {
                            // Check if field type is likely a struct that implements ToPrompt
                            // (not a primitive type)
                            let is_primitive = match &field.ty {
                                syn::Type::Path(type_path) => {
                                    if let Some(segment) = type_path.path.segments.last() {
                                        let type_name = segment.ident.to_string();
                                        matches!(
                                            type_name.as_str(),
                                            "String"
                                                | "str"
                                                | "i8"
                                                | "i16"
                                                | "i32"
                                                | "i64"
                                                | "i128"
                                                | "isize"
                                                | "u8"
                                                | "u16"
                                                | "u32"
                                                | "u64"
                                                | "u128"
                                                | "usize"
                                                | "f32"
                                                | "f64"
                                                | "bool"
                                                | "char"
                                        )
                                    } else {
                                        false
                                    }
                                }
                                _ => false,
                            };

                            if is_primitive {
                                context_fields.push(quote! {
                                    context.insert(
                                        #field_name_str.to_string(),
                                        minijinja::Value::from_serialize(&self.#field_name)
                                    );
                                });
                            } else {
                                // For non-primitive types, use to_prompt()
                                context_fields.push(quote! {
                                    context.insert(
                                        #field_name_str.to_string(),
                                        minijinja::Value::from(self.#field_name.to_prompt())
                                    );
                                });
                            }
                        }
                    }

                    quote! {
                        impl #impl_generics #crate_path::prompt::ToPrompt for #name #ty_generics #where_clause {
                            fn to_prompt_parts(&self) -> Vec<#crate_path::prompt::PromptPart> {
                                let mut parts = Vec::new();

                                // Add image parts first
                                #(#image_field_parts)*

                                // Build custom context and render template
                                let text = {
                                    let mut env = minijinja::Environment::new();
                                    env.add_template("prompt", #modified_template).unwrap_or_else(|e| {
                                        panic!("Failed to parse template: {}", e)
                                    });

                                    let tmpl = env.get_template("prompt").unwrap();

                                    let mut context = std::collections::HashMap::new();
                                    #(#context_fields)*

                                    tmpl.render(context).unwrap_or_else(|e| {
                                        format!("Failed to render prompt: {}", e)
                                    })
                                };

                                if !text.is_empty() {
                                    parts.push(#crate_path::prompt::PromptPart::Text(text));
                                }

                                parts
                            }

                            fn to_prompt(&self) -> String {
                                // Same logic for to_prompt
                                let mut env = minijinja::Environment::new();
                                env.add_template("prompt", #modified_template).unwrap_or_else(|e| {
                                    panic!("Failed to parse template: {}", e)
                                });

                                let tmpl = env.get_template("prompt").unwrap();

                                let mut context = std::collections::HashMap::new();
                                #(#context_fields)*

                                tmpl.render(context).unwrap_or_else(|e| {
                                    format!("Failed to render prompt: {}", e)
                                })
                            }

                            fn prompt_schema() -> String {
                                String::new() // Template-based structs don't have auto-generated schema
                            }
                        }
                    }
                } else {
                    // No mode syntax, use direct template rendering with render_prompt
                    quote! {
                        impl #impl_generics #crate_path::prompt::ToPrompt for #name #ty_generics #where_clause {
                            fn to_prompt_parts(&self) -> Vec<#crate_path::prompt::PromptPart> {
                                let mut parts = Vec::new();

                                // Add image parts first
                                #(#image_field_parts)*

                                // Add the rendered template as text
                                let text = #crate_path::prompt::render_prompt(#template, self).unwrap_or_else(|e| {
                                    format!("Failed to render prompt: {}", e)
                                });
                                if !text.is_empty() {
                                    parts.push(#crate_path::prompt::PromptPart::Text(text));
                                }

                                parts
                            }

                            fn to_prompt(&self) -> String {
                                #crate_path::prompt::render_prompt(#template, self).unwrap_or_else(|e| {
                                    format!("Failed to render prompt: {}", e)
                                })
                            }

                            fn prompt_schema() -> String {
                                String::new() // Template-based structs don't have auto-generated schema
                            }
                        }
                    }
                }
            } else {
                // Use default key-value format if no template is provided
                // Now also generate to_prompt_parts() for multimodal support
                let fields = if let syn::Fields::Named(fields) = &data_struct.fields {
                    &fields.named
                } else {
                    panic!(
                        "Default prompt generation is only supported for structs with named fields."
                    );
                };

                // Separate image fields from text fields
                let mut text_field_parts = Vec::new();
                let mut image_field_parts = Vec::new();

                for f in fields.iter() {
                    let field_name = f.ident.as_ref().unwrap();
                    let attrs = parse_field_prompt_attrs(&f.attrs);

                    // Skip if #[prompt(skip)] is present
                    if attrs.skip {
                        continue;
                    }

                    if attrs.image {
                        // This field is marked as an image
                        image_field_parts.push(quote! {
                            parts.extend(self.#field_name.to_prompt_parts());
                        });
                    } else {
                        // This is a regular text field
                        // Determine the key based on priority:
                        // 1. #[prompt(rename = "new_name")]
                        // 2. Doc comment
                        // 3. Field name (fallback)
                        let key = if let Some(rename) = attrs.rename {
                            rename
                        } else {
                            let doc_comment = extract_doc_comments(&f.attrs);
                            if !doc_comment.is_empty() {
                                doc_comment
                            } else {
                                field_name.to_string()
                            }
                        };

                        // Determine the value based on format_with attribute
                        let value_expr = if let Some(format_with) = attrs.format_with {
                            // Parse the function path string into a syn::Path
                            let func_path: syn::Path =
                                syn::parse_str(&format_with).unwrap_or_else(|_| {
                                    panic!("Invalid function path: {}", format_with)
                                });
                            quote! { #func_path(&self.#field_name) }
                        } else {
                            quote! { self.#field_name.to_prompt() }
                        };

                        text_field_parts.push(quote! {
                            text_parts.push(format!("{}: {}", #key, #value_expr));
                        });
                    }
                }

                // Generate the implementation with to_prompt_parts()
                quote! {
                    impl #impl_generics #crate_path::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt_parts(&self) -> Vec<#crate_path::prompt::PromptPart> {
                            let mut parts = Vec::new();

                            // Add image parts first
                            #(#image_field_parts)*

                            // Collect text parts and add as a single text prompt part
                            let mut text_parts = Vec::new();
                            #(#text_field_parts)*

                            if !text_parts.is_empty() {
                                parts.push(#crate_path::prompt::PromptPart::Text(text_parts.join("\n")));
                            }

                            parts
                        }

                        fn to_prompt(&self) -> String {
                            let mut text_parts = Vec::new();
                            #(#text_field_parts)*
                            text_parts.join("\n")
                        }

                        fn prompt_schema() -> String {
                            String::new() // Default key-value format doesn't have auto-generated schema
                        }
                    }
                }
            };

            TokenStream::from(expanded)
        }
        Data::Union(_) => {
            panic!("`#[derive(ToPrompt)]` is not supported for unions");
        }
    }
}

/// Information about a prompt target
#[derive(Debug, Clone)]
struct TargetInfo {
    name: String,
    template: Option<String>,
    field_configs: std::collections::HashMap<String, FieldTargetConfig>,
}

/// Configuration for how a field should be handled for a specific target
#[derive(Debug, Clone, Default)]
struct FieldTargetConfig {
    skip: bool,
    rename: Option<String>,
    format_with: Option<String>,
    image: bool,
    include_only: bool, // true if this field is specifically included for this target
}

/// Parse #[prompt_for(...)] attributes for ToPromptSet
fn parse_prompt_for_attrs(attrs: &[syn::Attribute]) -> Vec<(String, FieldTargetConfig)> {
    let mut configs = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("prompt_for")
            && let Ok(meta_list) = attr.meta.require_list()
        {
            // Try to parse as meta list
            if meta_list.tokens.to_string() == "skip" {
                // Simple #[prompt_for(skip)] applies to all targets
                let config = FieldTargetConfig {
                    skip: true,
                    ..Default::default()
                };
                configs.push(("*".to_string(), config));
            } else if let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
            {
                let mut target_name = None;
                let mut config = FieldTargetConfig::default();

                for meta in metas {
                    match meta {
                        Meta::NameValue(nv) if nv.path.is_ident("name") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit_str),
                                ..
                            }) = nv.value
                            {
                                target_name = Some(lit_str.value());
                            }
                        }
                        Meta::Path(path) if path.is_ident("skip") => {
                            config.skip = true;
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit_str),
                                ..
                            }) = nv.value
                            {
                                config.rename = Some(lit_str.value());
                            }
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("format_with") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit_str),
                                ..
                            }) = nv.value
                            {
                                config.format_with = Some(lit_str.value());
                            }
                        }
                        Meta::Path(path) if path.is_ident("image") => {
                            config.image = true;
                        }
                        _ => {}
                    }
                }

                if let Some(name) = target_name {
                    config.include_only = true;
                    configs.push((name, config));
                }
            }
        }
    }

    configs
}

/// Parse struct-level #[prompt_for(...)] attributes to find target templates
fn parse_struct_prompt_for_attrs(attrs: &[syn::Attribute]) -> Vec<TargetInfo> {
    let mut targets = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("prompt_for")
            && let Ok(meta_list) = attr.meta.require_list()
            && let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            let mut target_name = None;
            let mut template = None;

            for meta in metas {
                match meta {
                    Meta::NameValue(nv) if nv.path.is_ident("name") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            target_name = Some(lit_str.value());
                        }
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("template") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            template = Some(lit_str.value());
                        }
                    }
                    _ => {}
                }
            }

            if let Some(name) = target_name {
                targets.push(TargetInfo {
                    name,
                    template,
                    field_configs: std::collections::HashMap::new(),
                });
            }
        }
    }

    targets
}

#[proc_macro_derive(ToPromptSet, attributes(prompt_for))]
pub fn to_prompt_set_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            // Even when it's the same crate, use absolute path to support examples/tests/bins
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Only support structs with named fields
    let data_struct = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return syn::Error::new(
                input.ident.span(),
                "`#[derive(ToPromptSet)]` is only supported for structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let fields = match &data_struct.fields {
        syn::Fields::Named(fields) => &fields.named,
        _ => {
            return syn::Error::new(
                input.ident.span(),
                "`#[derive(ToPromptSet)]` is only supported for structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    // Parse struct-level attributes to find targets
    let mut targets = parse_struct_prompt_for_attrs(&input.attrs);

    // Parse field-level attributes
    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap().to_string();
        let field_configs = parse_prompt_for_attrs(&field.attrs);

        for (target_name, config) in field_configs {
            if target_name == "*" {
                // Apply to all targets
                for target in &mut targets {
                    target
                        .field_configs
                        .entry(field_name.clone())
                        .or_insert_with(FieldTargetConfig::default)
                        .skip = config.skip;
                }
            } else {
                // Find or create the target
                let target_exists = targets.iter().any(|t| t.name == target_name);
                if !target_exists {
                    // Add implicit target if not defined at struct level
                    targets.push(TargetInfo {
                        name: target_name.clone(),
                        template: None,
                        field_configs: std::collections::HashMap::new(),
                    });
                }

                let target = targets.iter_mut().find(|t| t.name == target_name).unwrap();

                target.field_configs.insert(field_name.clone(), config);
            }
        }
    }

    // Generate match arms for each target
    let mut match_arms = Vec::new();

    for target in &targets {
        let target_name = &target.name;

        if let Some(template_str) = &target.template {
            // Template-based generation
            let mut image_parts = Vec::new();

            for field in fields.iter() {
                let field_name = field.ident.as_ref().unwrap();
                let field_name_str = field_name.to_string();

                if let Some(config) = target.field_configs.get(&field_name_str)
                    && config.image
                {
                    image_parts.push(quote! {
                        parts.extend(self.#field_name.to_prompt_parts());
                    });
                }
            }

            match_arms.push(quote! {
                #target_name => {
                    let mut parts = Vec::new();

                    #(#image_parts)*

                    let text = #crate_path::prompt::render_prompt(#template_str, self)
                        .map_err(|e| #crate_path::prompt::PromptSetError::RenderFailed {
                            target: #target_name.to_string(),
                            source: e,
                        })?;

                    if !text.is_empty() {
                        parts.push(#crate_path::prompt::PromptPart::Text(text));
                    }

                    Ok(parts)
                }
            });
        } else {
            // Key-value based generation
            let mut text_field_parts = Vec::new();
            let mut image_field_parts = Vec::new();

            for field in fields.iter() {
                let field_name = field.ident.as_ref().unwrap();
                let field_name_str = field_name.to_string();

                // Check if field should be included for this target
                let config = target.field_configs.get(&field_name_str);

                // Skip if explicitly marked to skip
                if let Some(cfg) = config
                    && cfg.skip
                {
                    continue;
                }

                // For non-template targets, only include fields that are:
                // 1. Explicitly marked for this target with #[prompt_for(name = "Target")]
                // 2. Not marked for any specific target (default fields)
                let is_explicitly_for_this_target = config.is_some_and(|c| c.include_only);
                let has_any_target_specific_config = parse_prompt_for_attrs(&field.attrs)
                    .iter()
                    .any(|(name, _)| name != "*");

                if has_any_target_specific_config && !is_explicitly_for_this_target {
                    continue;
                }

                if let Some(cfg) = config {
                    if cfg.image {
                        image_field_parts.push(quote! {
                            parts.extend(self.#field_name.to_prompt_parts());
                        });
                    } else {
                        let key = cfg.rename.clone().unwrap_or_else(|| field_name_str.clone());

                        let value_expr = if let Some(format_with) = &cfg.format_with {
                            // Parse the function path - if it fails, generate code that will produce a compile error
                            match syn::parse_str::<syn::Path>(format_with) {
                                Ok(func_path) => quote! { #func_path(&self.#field_name) },
                                Err(_) => {
                                    // Generate a compile error by using an invalid identifier
                                    let error_msg = format!(
                                        "Invalid function path in format_with: '{}'",
                                        format_with
                                    );
                                    quote! {
                                        compile_error!(#error_msg);
                                        String::new()
                                    }
                                }
                            }
                        } else {
                            quote! { self.#field_name.to_prompt() }
                        };

                        text_field_parts.push(quote! {
                            text_parts.push(format!("{}: {}", #key, #value_expr));
                        });
                    }
                } else {
                    // Default handling for fields without specific config
                    text_field_parts.push(quote! {
                        text_parts.push(format!("{}: {}", #field_name_str, self.#field_name.to_prompt()));
                    });
                }
            }

            match_arms.push(quote! {
                #target_name => {
                    let mut parts = Vec::new();

                    #(#image_field_parts)*

                    let mut text_parts = Vec::new();
                    #(#text_field_parts)*

                    if !text_parts.is_empty() {
                        parts.push(#crate_path::prompt::PromptPart::Text(text_parts.join("\n")));
                    }

                    Ok(parts)
                }
            });
        }
    }

    // Collect all target names for error reporting
    let target_names: Vec<String> = targets.iter().map(|t| t.name.clone()).collect();

    // Add default case for unknown targets
    match_arms.push(quote! {
        _ => {
            let available = vec![#(#target_names.to_string()),*];
            Err(#crate_path::prompt::PromptSetError::TargetNotFound {
                target: target.to_string(),
                available,
            })
        }
    });

    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics #crate_path::prompt::ToPromptSet for #struct_name #ty_generics #where_clause {
            fn to_prompt_parts_for(&self, target: &str) -> Result<Vec<#crate_path::prompt::PromptPart>, #crate_path::prompt::PromptSetError> {
                match target {
                    #(#match_arms)*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Wrapper struct for parsing a comma-separated list of types
struct TypeList {
    types: Punctuated<syn::Type, Token![,]>,
}

impl Parse for TypeList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(TypeList {
            types: Punctuated::parse_terminated(input)?,
        })
    }
}

/// Generates a formatted Markdown examples section for the provided types.
///
/// This macro accepts a comma-separated list of types and generates a single
/// formatted Markdown string containing examples of each type.
///
/// # Example
///
/// ```rust,ignore
/// let examples = examples_section!(User, Concept);
/// // Produces a string like:
/// // ---
/// // ### Examples
/// //
/// // Here are examples of the data structures you should use.
/// //
/// // ---
/// // #### `User`
/// // {...json...}
/// // ---
/// // #### `Concept`
/// // {...json...}
/// // ---
/// ```
#[proc_macro]
pub fn examples_section(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TypeList);

    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let _crate_path = match found_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Generate code for each type
    let mut type_sections = Vec::new();

    for ty in input.types.iter() {
        // Extract the type name as a string
        let type_name_str = quote!(#ty).to_string();

        // Generate the section for this type
        type_sections.push(quote! {
            {
                let type_name = #type_name_str;
                let json_example = <#ty as Default>::default().to_prompt_with_mode("example_only");
                format!("---\n#### `{}`\n{}", type_name, json_example)
            }
        });
    }

    // Build the complete examples string
    let expanded = quote! {
        {
            let mut sections = Vec::new();
            sections.push("---".to_string());
            sections.push("### Examples".to_string());
            sections.push("".to_string());
            sections.push("Here are examples of the data structures you should use.".to_string());
            sections.push("".to_string());

            #(sections.push(#type_sections);)*

            sections.push("---".to_string());

            sections.join("\n")
        }
    };

    TokenStream::from(expanded)
}

/// Helper function to parse struct-level #[prompt_for(target = "...", template = "...")] attribute
fn parse_to_prompt_for_attribute(attrs: &[syn::Attribute]) -> (syn::Type, String) {
    for attr in attrs {
        if attr.path().is_ident("prompt_for")
            && let Ok(meta_list) = attr.meta.require_list()
            && let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            let mut target_type = None;
            let mut template = None;

            for meta in metas {
                match meta {
                    Meta::NameValue(nv) if nv.path.is_ident("target") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            // Parse the type string into a syn::Type
                            target_type = syn::parse_str::<syn::Type>(&lit_str.value()).ok();
                        }
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("template") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            template = Some(lit_str.value());
                        }
                    }
                    _ => {}
                }
            }

            if let (Some(target), Some(tmpl)) = (target_type, template) {
                return (target, tmpl);
            }
        }
    }

    panic!("ToPromptFor requires #[prompt_for(target = \"TargetType\", template = \"...\")]");
}

/// A procedural attribute macro that generates prompt-building functions and extractor structs for intent enums.
///
/// This macro should be applied to an enum to generate:
/// 1. A prompt-building function that incorporates enum documentation
/// 2. An extractor struct that implements `IntentExtractor`
///
/// # Requirements
///
/// The enum must have an `#[intent(...)]` attribute with:
/// - `prompt`: The prompt template (supports Jinja-style variables)
/// - `extractor_tag`: The tag to use for extraction
///
/// # Example
///
/// ```rust,ignore
/// #[define_intent]
/// #[intent(
///     prompt = "Analyze the intent: {{ user_input }}",
///     extractor_tag = "intent"
/// )]
/// enum MyIntent {
///     /// Create a new item
///     Create,
///     /// Update an existing item
///     Update,
///     /// Delete an item
///     Delete,
/// }
/// ```
///
/// This will generate:
/// - `pub fn build_my_intent_prompt(user_input: &str) -> String`
/// - `pub struct MyIntentExtractor;` with `IntentExtractor<MyIntent>` implementation
#[proc_macro_attribute]
pub fn define_intent(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            // Even when it's the same crate, use absolute path to support examples/tests/bins
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Verify this is an enum
    let enum_data = match &input.data {
        Data::Enum(data) => data,
        _ => {
            return syn::Error::new(
                input.ident.span(),
                "`#[define_intent]` can only be applied to enums",
            )
            .to_compile_error()
            .into();
        }
    };

    // Parse the #[intent(...)] attribute
    let mut prompt_template = None;
    let mut extractor_tag = None;
    let mut mode = None;

    for attr in &input.attrs {
        if attr.path().is_ident("intent")
            && let Ok(metas) =
                attr.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            for meta in metas {
                match meta {
                    Meta::NameValue(nv) if nv.path.is_ident("prompt") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            prompt_template = Some(lit_str.value());
                        }
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("extractor_tag") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            extractor_tag = Some(lit_str.value());
                        }
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("mode") => {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = nv.value
                        {
                            mode = Some(lit_str.value());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Parse the mode parameter (default to "single")
    let mode = mode.unwrap_or_else(|| "single".to_string());

    // Validate mode
    if mode != "single" && mode != "multi_tag" {
        return syn::Error::new(
            input.ident.span(),
            "`mode` must be either \"single\" or \"multi_tag\"",
        )
        .to_compile_error()
        .into();
    }

    // Validate required attributes
    let prompt_template = match prompt_template {
        Some(p) => p,
        None => {
            return syn::Error::new(
                input.ident.span(),
                "`#[intent(...)]` attribute must include `prompt = \"...\"`",
            )
            .to_compile_error()
            .into();
        }
    };

    // Handle multi_tag mode
    if mode == "multi_tag" {
        let enum_name = &input.ident;
        let actions_doc = generate_multi_tag_actions_doc(&enum_data.variants);
        return generate_multi_tag_output(
            &input,
            enum_name,
            enum_data,
            prompt_template,
            actions_doc,
        );
    }

    // Continue with single mode logic
    let extractor_tag = match extractor_tag {
        Some(t) => t,
        None => {
            return syn::Error::new(
                input.ident.span(),
                "`#[intent(...)]` attribute must include `extractor_tag = \"...\"`",
            )
            .to_compile_error()
            .into();
        }
    };

    // Generate the intents documentation
    let enum_name = &input.ident;
    let enum_docs = extract_doc_comments(&input.attrs);

    let mut intents_doc_lines = Vec::new();

    // Add enum description if present
    if !enum_docs.is_empty() {
        intents_doc_lines.push(format!("{}: {}", enum_name, enum_docs));
    } else {
        intents_doc_lines.push(format!("{}:", enum_name));
    }
    intents_doc_lines.push(String::new()); // Empty line
    intents_doc_lines.push("Possible values:".to_string());

    // Add each variant with its documentation
    for variant in &enum_data.variants {
        let variant_name = &variant.ident;
        let variant_docs = extract_doc_comments(&variant.attrs);

        if !variant_docs.is_empty() {
            intents_doc_lines.push(format!("- {}: {}", variant_name, variant_docs));
        } else {
            intents_doc_lines.push(format!("- {}", variant_name));
        }
    }

    let intents_doc_str = intents_doc_lines.join("\n");

    // Parse template variables (excluding intents_doc which we'll inject)
    let placeholders = parse_template_placeholders_with_mode(&prompt_template);
    let user_variables: Vec<String> = placeholders
        .iter()
        .filter_map(|(name, _)| {
            if name != "intents_doc" {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect();

    // Generate function name (snake_case)
    let enum_name_str = enum_name.to_string();
    let snake_case_name = to_snake_case(&enum_name_str);
    let function_name = syn::Ident::new(
        &format!("build_{}_prompt", snake_case_name),
        proc_macro2::Span::call_site(),
    );

    // Generate function parameters (all &str for simplicity)
    let function_params: Vec<proc_macro2::TokenStream> = user_variables
        .iter()
        .map(|var| {
            let ident = syn::Ident::new(var, proc_macro2::Span::call_site());
            quote! { #ident: &str }
        })
        .collect();

    // Generate context insertions
    let context_insertions: Vec<proc_macro2::TokenStream> = user_variables
        .iter()
        .map(|var| {
            let var_str = var.clone();
            let ident = syn::Ident::new(var, proc_macro2::Span::call_site());
            quote! {
                __template_context.insert(#var_str.to_string(), minijinja::Value::from(#ident));
            }
        })
        .collect();

    // Template is already in Jinja syntax, no conversion needed
    let converted_template = prompt_template.clone();

    // Generate extractor struct name
    let extractor_name = syn::Ident::new(
        &format!("{}Extractor", enum_name),
        proc_macro2::Span::call_site(),
    );

    // Filter out the #[intent(...)] attribute from the enum attributes
    let filtered_attrs: Vec<_> = input
        .attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("intent"))
        .collect();

    // Rebuild the enum with filtered attributes
    let vis = &input.vis;
    let generics = &input.generics;
    let variants = &enum_data.variants;
    let enum_output = quote! {
        #(#filtered_attrs)*
        #vis enum #enum_name #generics {
            #variants
        }
    };

    // Generate the complete output
    let expanded = quote! {
        // Output the enum without the #[intent(...)] attribute
        #enum_output

        // Generate the prompt-building function
        pub fn #function_name(#(#function_params),*) -> String {
            let mut env = minijinja::Environment::new();
            env.add_template("prompt", #converted_template)
                .expect("Failed to parse intent prompt template");

            let tmpl = env.get_template("prompt").unwrap();

            let mut __template_context = std::collections::HashMap::new();

            // Add intents_doc
            __template_context.insert("intents_doc".to_string(), minijinja::Value::from(#intents_doc_str));

            // Add user-provided variables
            #(#context_insertions)*

            tmpl.render(&__template_context)
                .unwrap_or_else(|e| format!("Failed to render intent prompt: {}", e))
        }

        // Generate the extractor struct
        pub struct #extractor_name;

        impl #extractor_name {
            pub const EXTRACTOR_TAG: &'static str = #extractor_tag;
        }

        impl #crate_path::intent::IntentExtractor<#enum_name> for #extractor_name {
            fn extract_intent(&self, response: &str) -> Result<#enum_name, #crate_path::intent::IntentExtractionError> {
                // Use the common extraction function with our tag
                #crate_path::intent::extract_intent_from_response(response, Self::EXTRACTOR_TAG)
            }
        }
    };

    TokenStream::from(expanded)
}

/// Convert PascalCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let mut prev_upper = false;

    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
            prev_upper = true;
        } else {
            result.push(ch);
            prev_upper = false;
        }
    }

    result
}

/// Parse #[action(...)] attributes for enum variants
#[derive(Debug, Default)]
struct ActionAttrs {
    tag: Option<String>,
}

fn parse_action_attrs(attrs: &[syn::Attribute]) -> ActionAttrs {
    let mut result = ActionAttrs::default();

    for attr in attrs {
        if attr.path().is_ident("action")
            && let Ok(meta_list) = attr.meta.require_list()
            && let Ok(metas) =
                meta_list.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            for meta in metas {
                if let Meta::NameValue(nv) = meta
                    && nv.path.is_ident("tag")
                    && let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = nv.value
                {
                    result.tag = Some(lit_str.value());
                }
            }
        }
    }

    result
}

/// Parse #[action(...)] attributes for struct fields in variants
#[derive(Debug, Default)]
struct FieldActionAttrs {
    is_attribute: bool,
    is_inner_text: bool,
}

fn parse_field_action_attrs(attrs: &[syn::Attribute]) -> FieldActionAttrs {
    let mut result = FieldActionAttrs::default();

    for attr in attrs {
        if attr.path().is_ident("action")
            && let Ok(meta_list) = attr.meta.require_list()
        {
            let tokens_str = meta_list.tokens.to_string();
            if tokens_str == "attribute" {
                result.is_attribute = true;
            } else if tokens_str == "inner_text" {
                result.is_inner_text = true;
            }
        }
    }

    result
}

/// Generate actions_doc for multi_tag mode
fn generate_multi_tag_actions_doc(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::Token![,]>,
) -> String {
    let mut doc_lines = Vec::new();

    for variant in variants {
        let action_attrs = parse_action_attrs(&variant.attrs);

        if let Some(tag) = action_attrs.tag {
            let variant_docs = extract_doc_comments(&variant.attrs);

            match &variant.fields {
                syn::Fields::Unit => {
                    // Simple tag without parameters
                    doc_lines.push(format!("- `<{} />`: {}", tag, variant_docs));
                }
                syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                    // Tuple variant with inner text
                    doc_lines.push(format!("- `<{}>...</{}>`: {}", tag, tag, variant_docs));
                }
                syn::Fields::Named(fields) => {
                    // Struct variant with attributes and/or inner text
                    let mut attrs_str = Vec::new();
                    let mut has_inner_text = false;

                    for field in &fields.named {
                        let field_name = field.ident.as_ref().unwrap();
                        let field_attrs = parse_field_action_attrs(&field.attrs);

                        if field_attrs.is_attribute {
                            attrs_str.push(format!("{}=\"...\"", field_name));
                        } else if field_attrs.is_inner_text {
                            has_inner_text = true;
                        }
                    }

                    let attrs_part = if !attrs_str.is_empty() {
                        format!(" {}", attrs_str.join(" "))
                    } else {
                        String::new()
                    };

                    if has_inner_text {
                        doc_lines.push(format!(
                            "- `<{}{}>...</{}>`: {}",
                            tag, attrs_part, tag, variant_docs
                        ));
                    } else if !attrs_str.is_empty() {
                        doc_lines.push(format!("- `<{}{} />`: {}", tag, attrs_part, variant_docs));
                    } else {
                        doc_lines.push(format!("- `<{} />`: {}", tag, variant_docs));
                    }

                    // Add field documentation
                    for field in &fields.named {
                        let field_name = field.ident.as_ref().unwrap();
                        let field_attrs = parse_field_action_attrs(&field.attrs);
                        let field_docs = extract_doc_comments(&field.attrs);

                        if field_attrs.is_attribute {
                            doc_lines
                                .push(format!("  - `{}` (attribute): {}", field_name, field_docs));
                        } else if field_attrs.is_inner_text {
                            doc_lines
                                .push(format!("  - `{}` (inner_text): {}", field_name, field_docs));
                        }
                    }
                }
                _ => {
                    // Other field types not supported
                }
            }
        }
    }

    doc_lines.join("\n")
}

/// Generate regex for matching any of the defined action tags
fn generate_tags_regex(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::Token![,]>,
) -> String {
    let mut tag_names = Vec::new();

    for variant in variants {
        let action_attrs = parse_action_attrs(&variant.attrs);
        if let Some(tag) = action_attrs.tag {
            tag_names.push(tag);
        }
    }

    if tag_names.is_empty() {
        return String::new();
    }

    let tags_pattern = tag_names.join("|");
    // Match both self-closing tags like <Tag /> and content-based tags like <Tag>...</Tag>
    // (?is) enables case-insensitive and single-line mode where . matches newlines
    format!(
        r"(?is)<(?:{})\b[^>]*/>|<(?:{})\b[^>]*>.*?</(?:{})>",
        tags_pattern, tags_pattern, tags_pattern
    )
}

/// Generate output for multi_tag mode
fn generate_multi_tag_output(
    input: &DeriveInput,
    enum_name: &syn::Ident,
    enum_data: &syn::DataEnum,
    prompt_template: String,
    actions_doc: String,
) -> TokenStream {
    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            // Even when it's the same crate, use absolute path to support examples/tests/bins
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Parse template placeholders
    let placeholders = parse_template_placeholders_with_mode(&prompt_template);
    let user_variables: Vec<String> = placeholders
        .iter()
        .filter_map(|(name, _)| {
            if name != "actions_doc" {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect();

    // Generate function name (snake_case)
    let enum_name_str = enum_name.to_string();
    let snake_case_name = to_snake_case(&enum_name_str);
    let function_name = syn::Ident::new(
        &format!("build_{}_prompt", snake_case_name),
        proc_macro2::Span::call_site(),
    );

    // Generate function parameters (all &str for simplicity)
    let function_params: Vec<proc_macro2::TokenStream> = user_variables
        .iter()
        .map(|var| {
            let ident = syn::Ident::new(var, proc_macro2::Span::call_site());
            quote! { #ident: &str }
        })
        .collect();

    // Generate context insertions
    let context_insertions: Vec<proc_macro2::TokenStream> = user_variables
        .iter()
        .map(|var| {
            let var_str = var.clone();
            let ident = syn::Ident::new(var, proc_macro2::Span::call_site());
            quote! {
                __template_context.insert(#var_str.to_string(), minijinja::Value::from(#ident));
            }
        })
        .collect();

    // Generate extractor struct name
    let extractor_name = syn::Ident::new(
        &format!("{}Extractor", enum_name),
        proc_macro2::Span::call_site(),
    );

    // Filter out the #[intent(...)] and #[action(...)] attributes
    let filtered_attrs: Vec<_> = input
        .attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("intent"))
        .collect();

    // Filter action attributes from variants
    let filtered_variants: Vec<proc_macro2::TokenStream> = enum_data
        .variants
        .iter()
        .map(|variant| {
            let variant_name = &variant.ident;
            let variant_attrs: Vec<_> = variant
                .attrs
                .iter()
                .filter(|attr| !attr.path().is_ident("action"))
                .collect();
            let fields = &variant.fields;

            // Filter field attributes
            let filtered_fields = match fields {
                syn::Fields::Named(named_fields) => {
                    let filtered: Vec<_> = named_fields
                        .named
                        .iter()
                        .map(|field| {
                            let field_name = &field.ident;
                            let field_type = &field.ty;
                            let field_vis = &field.vis;
                            let filtered_attrs: Vec<_> = field
                                .attrs
                                .iter()
                                .filter(|attr| !attr.path().is_ident("action"))
                                .collect();
                            quote! {
                                #(#filtered_attrs)*
                                #field_vis #field_name: #field_type
                            }
                        })
                        .collect();
                    quote! { { #(#filtered,)* } }
                }
                syn::Fields::Unnamed(unnamed_fields) => {
                    let types: Vec<_> = unnamed_fields
                        .unnamed
                        .iter()
                        .map(|field| {
                            let field_type = &field.ty;
                            quote! { #field_type }
                        })
                        .collect();
                    quote! { (#(#types),*) }
                }
                syn::Fields::Unit => quote! {},
            };

            quote! {
                #(#variant_attrs)*
                #variant_name #filtered_fields
            }
        })
        .collect();

    let vis = &input.vis;
    let generics = &input.generics;

    // Generate XML parsing logic for extract_actions
    let parsing_arms = generate_parsing_arms(&enum_data.variants, enum_name);

    // Generate the regex pattern for matching tags
    let tags_regex = generate_tags_regex(&enum_data.variants);

    let expanded = quote! {
        // Output the enum without the #[intent(...)] and #[action(...)] attributes
        #(#filtered_attrs)*
        #vis enum #enum_name #generics {
            #(#filtered_variants),*
        }

        // Generate the prompt-building function
        pub fn #function_name(#(#function_params),*) -> String {
            let mut env = minijinja::Environment::new();
            env.add_template("prompt", #prompt_template)
                .expect("Failed to parse intent prompt template");

            let tmpl = env.get_template("prompt").unwrap();

            let mut __template_context = std::collections::HashMap::new();

            // Add actions_doc
            __template_context.insert("actions_doc".to_string(), minijinja::Value::from(#actions_doc));

            // Add user-provided variables
            #(#context_insertions)*

            tmpl.render(&__template_context)
                .unwrap_or_else(|e| format!("Failed to render intent prompt: {}", e))
        }

        // Generate the extractor struct
        pub struct #extractor_name;

        impl #extractor_name {
            fn parse_single_action(&self, text: &str) -> Option<#enum_name> {
                use ::quick_xml::events::Event;
                use ::quick_xml::Reader;

                let mut actions = Vec::new();
                let mut reader = Reader::from_str(text);
                reader.config_mut().trim_text(true);

                let mut buf = Vec::new();

                loop {
                    match reader.read_event_into(&mut buf) {
                        Ok(Event::Start(e)) => {
                            let owned_e = e.into_owned();
                            let tag_name = String::from_utf8_lossy(owned_e.name().as_ref()).to_string();
                            let is_empty = false;

                            #parsing_arms
                        }
                        Ok(Event::Empty(e)) => {
                            let owned_e = e.into_owned();
                            let tag_name = String::from_utf8_lossy(owned_e.name().as_ref()).to_string();
                            let is_empty = true;

                            #parsing_arms
                        }
                        Ok(Event::Eof) => break,
                        Err(_) => {
                            // Silently ignore XML parsing errors
                            break;
                        }
                        _ => {}
                    }
                    buf.clear();
                }

                actions.into_iter().next()
            }

            pub fn extract_actions(&self, text: &str) -> Result<Vec<#enum_name>, #crate_path::intent::IntentError> {
                use ::quick_xml::events::Event;
                use ::quick_xml::Reader;

                let mut actions = Vec::new();
                let mut reader = Reader::from_str(text);
                reader.config_mut().trim_text(true);

                let mut buf = Vec::new();

                loop {
                    match reader.read_event_into(&mut buf) {
                        Ok(Event::Start(e)) => {
                            let owned_e = e.into_owned();
                            let tag_name = String::from_utf8_lossy(owned_e.name().as_ref()).to_string();
                            let is_empty = false;

                            #parsing_arms
                        }
                        Ok(Event::Empty(e)) => {
                            let owned_e = e.into_owned();
                            let tag_name = String::from_utf8_lossy(owned_e.name().as_ref()).to_string();
                            let is_empty = true;

                            #parsing_arms
                        }
                        Ok(Event::Eof) => break,
                        Err(_) => {
                            // Silently ignore XML parsing errors
                            break;
                        }
                        _ => {}
                    }
                    buf.clear();
                }

                Ok(actions)
            }

            pub fn transform_actions<F>(&self, text: &str, mut transformer: F) -> String
            where
                F: FnMut(#enum_name) -> String,
            {
                use ::regex::Regex;

                let regex_pattern = #tags_regex;
                if regex_pattern.is_empty() {
                    return text.to_string();
                }

                let re = Regex::new(&regex_pattern).unwrap_or_else(|e| {
                    panic!("Failed to compile regex for action tags: {}", e);
                });

                re.replace_all(text, |caps: &::regex::Captures| {
                    let matched = caps.get(0).map(|m| m.as_str()).unwrap_or("");

                    // Try to parse the matched tag as an action
                    if let Some(action) = self.parse_single_action(matched) {
                        transformer(action)
                    } else {
                        // If parsing fails, return the original text
                        matched.to_string()
                    }
                }).to_string()
            }

            pub fn strip_actions(&self, text: &str) -> String {
                self.transform_actions(text, |_| String::new())
            }
        }
    };

    TokenStream::from(expanded)
}

/// Generate parsing arms for XML extraction
fn generate_parsing_arms(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::Token![,]>,
    enum_name: &syn::Ident,
) -> proc_macro2::TokenStream {
    let mut arms = Vec::new();

    for variant in variants {
        let variant_name = &variant.ident;
        let action_attrs = parse_action_attrs(&variant.attrs);

        if let Some(tag) = action_attrs.tag {
            match &variant.fields {
                syn::Fields::Unit => {
                    // Simple tag without parameters
                    arms.push(quote! {
                        if &tag_name == #tag {
                            actions.push(#enum_name::#variant_name);
                        }
                    });
                }
                syn::Fields::Unnamed(_fields) => {
                    // Tuple variant with inner text - use reader.read_text()
                    arms.push(quote! {
                        if &tag_name == #tag && !is_empty {
                            // Use read_text to get inner text as owned String
                            match reader.read_text(owned_e.name()) {
                                Ok(text) => {
                                    actions.push(#enum_name::#variant_name(text.to_string()));
                                }
                                Err(_) => {
                                    // If reading text fails, push empty string
                                    actions.push(#enum_name::#variant_name(String::new()));
                                }
                            }
                        }
                    });
                }
                syn::Fields::Named(fields) => {
                    // Struct variant with attributes and/or inner text
                    let mut field_names = Vec::new();
                    let mut has_inner_text_field = None;

                    for field in &fields.named {
                        let field_name = field.ident.as_ref().unwrap();
                        let field_attrs = parse_field_action_attrs(&field.attrs);

                        if field_attrs.is_attribute {
                            field_names.push(field_name.clone());
                        } else if field_attrs.is_inner_text {
                            has_inner_text_field = Some(field_name.clone());
                        }
                    }

                    if let Some(inner_text_field) = has_inner_text_field {
                        // Handle inner text
                        // Build attribute extraction code
                        let attr_extractions: Vec<_> = field_names.iter().map(|field_name| {
                            quote! {
                                let mut #field_name = String::new();
                                for attr in owned_e.attributes() {
                                    if let Ok(attr) = attr {
                                        if attr.key.as_ref() == stringify!(#field_name).as_bytes() {
                                            #field_name = String::from_utf8_lossy(&attr.value).to_string();
                                            break;
                                        }
                                    }
                                }
                            }
                        }).collect();

                        arms.push(quote! {
                            if &tag_name == #tag {
                                #(#attr_extractions)*

                                // Check if it's a self-closing tag
                                if is_empty {
                                    let #inner_text_field = String::new();
                                    actions.push(#enum_name::#variant_name {
                                        #(#field_names,)*
                                        #inner_text_field,
                                    });
                                } else {
                                    // Use read_text to get inner text as owned String
                                    match reader.read_text(owned_e.name()) {
                                        Ok(text) => {
                                            let #inner_text_field = text.to_string();
                                            actions.push(#enum_name::#variant_name {
                                                #(#field_names,)*
                                                #inner_text_field,
                                            });
                                        }
                                        Err(_) => {
                                            // If reading text fails, push with empty string
                                            let #inner_text_field = String::new();
                                            actions.push(#enum_name::#variant_name {
                                                #(#field_names,)*
                                                #inner_text_field,
                                            });
                                        }
                                    }
                                }
                            }
                        });
                    } else {
                        // Only attributes
                        let attr_extractions: Vec<_> = field_names.iter().map(|field_name| {
                            quote! {
                                let mut #field_name = String::new();
                                for attr in owned_e.attributes() {
                                    if let Ok(attr) = attr {
                                        if attr.key.as_ref() == stringify!(#field_name).as_bytes() {
                                            #field_name = String::from_utf8_lossy(&attr.value).to_string();
                                            break;
                                        }
                                    }
                                }
                            }
                        }).collect();

                        arms.push(quote! {
                            if &tag_name == #tag {
                                #(#attr_extractions)*
                                actions.push(#enum_name::#variant_name {
                                    #(#field_names),*
                                });
                            }
                        });
                    }
                }
            }
        }
    }

    quote! {
        #(#arms)*
    }
}

/// Derives the `ToPromptFor` trait for a struct
#[proc_macro_derive(ToPromptFor, attributes(prompt_for))]
pub fn to_prompt_for_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            // Even when it's the same crate, use absolute path to support examples/tests/bins
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Parse the struct-level prompt_for attribute
    let (target_type, template) = parse_to_prompt_for_attribute(&input.attrs);

    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Parse the template to find placeholders
    let placeholders = parse_template_placeholders_with_mode(&template);

    // Convert template to minijinja syntax and build context generation code
    let mut converted_template = template.clone();
    let mut context_fields = Vec::new();

    // Get struct fields for validation
    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            syn::Fields::Named(fields) => &fields.named,
            _ => panic!("ToPromptFor is only supported for structs with named fields"),
        },
        _ => panic!("ToPromptFor is only supported for structs"),
    };

    // Check if the struct has mode support (has #[prompt(mode = ...)] attribute)
    let has_mode_support = input.attrs.iter().any(|attr| {
        if attr.path().is_ident("prompt")
            && let Ok(metas) =
                attr.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            for meta in metas {
                if let Meta::NameValue(nv) = meta
                    && nv.path.is_ident("mode")
                {
                    return true;
                }
            }
        }
        false
    });

    // Process each placeholder
    for (placeholder_name, mode_opt) in &placeholders {
        if placeholder_name == "self" {
            if let Some(specific_mode) = mode_opt {
                // {self:some_mode} - use a unique key
                let unique_key = format!("self__{}", specific_mode);

                // Replace {{ self:mode }} with {{ self__mode }} in template
                let pattern = format!("{{{{ self:{} }}}}", specific_mode);
                let replacement = format!("{{{{ {} }}}}", unique_key);
                converted_template = converted_template.replace(&pattern, &replacement);

                // Add to context with the specific mode
                context_fields.push(quote! {
                    context.insert(
                        #unique_key.to_string(),
                        minijinja::Value::from(self.to_prompt_with_mode(#specific_mode))
                    );
                });
            } else {
                // {{self}} - already in correct format, no replacement needed

                if has_mode_support {
                    // If the struct has mode support, use to_prompt_with_mode with the mode parameter
                    context_fields.push(quote! {
                        context.insert(
                            "self".to_string(),
                            minijinja::Value::from(self.to_prompt_with_mode(mode))
                        );
                    });
                } else {
                    // If the struct doesn't have mode support, use to_prompt() which gives key-value format
                    context_fields.push(quote! {
                        context.insert(
                            "self".to_string(),
                            minijinja::Value::from(self.to_prompt())
                        );
                    });
                }
            }
        } else {
            // It's a field placeholder
            // Check if the field exists
            let field_exists = fields.iter().any(|f| {
                f.ident
                    .as_ref()
                    .is_some_and(|ident| ident == placeholder_name)
            });

            if field_exists {
                let field_ident = syn::Ident::new(placeholder_name, proc_macro2::Span::call_site());

                // {{field}} - already in correct format, no replacement needed

                // Add field to context - serialize the field value
                context_fields.push(quote! {
                    context.insert(
                        #placeholder_name.to_string(),
                        minijinja::Value::from_serialize(&self.#field_ident)
                    );
                });
            }
            // If field doesn't exist, we'll let minijinja handle the error at runtime
        }
    }

    let expanded = quote! {
        impl #impl_generics #crate_path::prompt::ToPromptFor<#target_type> for #struct_name #ty_generics #where_clause
        where
            #target_type: serde::Serialize,
        {
            fn to_prompt_for_with_mode(&self, target: &#target_type, mode: &str) -> String {
                // Create minijinja environment and add template
                let mut env = minijinja::Environment::new();
                env.add_template("prompt", #converted_template).unwrap_or_else(|e| {
                    panic!("Failed to parse template: {}", e)
                });

                let tmpl = env.get_template("prompt").unwrap();

                // Build context
                let mut context = std::collections::HashMap::new();
                // Add self to the context for field access in templates
                context.insert(
                    "self".to_string(),
                    minijinja::Value::from_serialize(self)
                );
                // Add target to the context
                context.insert(
                    "target".to_string(),
                    minijinja::Value::from_serialize(target)
                );
                #(#context_fields)*

                // Render template
                tmpl.render(context).unwrap_or_else(|e| {
                    format!("Failed to render prompt: {}", e)
                })
            }
        }
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Agent Derive Macro
// ============================================================================

/// Attribute parameters for #[agent(...)]
struct AgentAttrs {
    expertise: Option<String>,
    output: Option<syn::Type>,
    backend: Option<String>,
    model: Option<String>,
    inner: Option<String>,
    default_inner: Option<String>,
    max_retries: Option<u32>,
    profile: Option<String>,
}

impl Parse for AgentAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut expertise = None;
        let mut output = None;
        let mut backend = None;
        let mut model = None;
        let mut inner = None;
        let mut default_inner = None;
        let mut max_retries = None;
        let mut profile = None;

        let pairs = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;

        for meta in pairs {
            match meta {
                Meta::NameValue(nv) if nv.path.is_ident("expertise") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        expertise = Some(lit_str.value());
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("output") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        let ty: syn::Type = syn::parse_str(&lit_str.value())?;
                        output = Some(ty);
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("backend") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        backend = Some(lit_str.value());
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("model") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        model = Some(lit_str.value());
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("inner") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        inner = Some(lit_str.value());
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("default_inner") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        default_inner = Some(lit_str.value());
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("max_retries") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(lit_int),
                        ..
                    }) = &nv.value
                    {
                        max_retries = Some(lit_int.base10_parse()?);
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("profile") => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &nv.value
                    {
                        profile = Some(lit_str.value());
                    }
                }
                _ => {}
            }
        }

        Ok(AgentAttrs {
            expertise,
            output,
            backend,
            model,
            inner,
            default_inner,
            max_retries,
            profile,
        })
    }
}

/// Parse #[agent(...)] attributes from a struct
fn parse_agent_attrs(attrs: &[syn::Attribute]) -> syn::Result<AgentAttrs> {
    for attr in attrs {
        if attr.path().is_ident("agent") {
            return attr.parse_args::<AgentAttrs>();
        }
    }

    Ok(AgentAttrs {
        expertise: None,
        output: None,
        backend: None,
        model: None,
        inner: None,
        default_inner: None,
        max_retries: None,
        profile: None,
    })
}

/// Generate backend-specific convenience constructors
fn generate_backend_constructors(
    struct_name: &syn::Ident,
    backend: &str,
    _model: Option<&str>,
    _profile: Option<&str>,
    crate_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match backend {
        "claude" => {
            quote! {
                impl #struct_name {
                    /// Create a new agent with ClaudeCodeAgent backend
                    pub fn with_claude() -> Self {
                        Self::new(#crate_path::agent::impls::ClaudeCodeAgent::new())
                    }

                    /// Create a new agent with ClaudeCodeAgent backend and specific model
                    pub fn with_claude_model(model: &str) -> Self {
                        Self::new(
                            #crate_path::agent::impls::ClaudeCodeAgent::new()
                                .with_model_str(model)
                        )
                    }
                }
            }
        }
        "gemini" => {
            quote! {
                impl #struct_name {
                    /// Create a new agent with GeminiAgent backend
                    pub fn with_gemini() -> Self {
                        Self::new(#crate_path::agent::impls::GeminiAgent::new())
                    }

                    /// Create a new agent with GeminiAgent backend and specific model
                    pub fn with_gemini_model(model: &str) -> Self {
                        Self::new(
                            #crate_path::agent::impls::GeminiAgent::new()
                                .with_model_str(model)
                        )
                    }
                }
            }
        }
        _ => quote! {},
    }
}

/// Generate Default implementation for the agent
fn generate_default_impl(
    struct_name: &syn::Ident,
    backend: &str,
    model: Option<&str>,
    profile: Option<&str>,
    crate_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    // Parse profile string to ExecutionProfile
    let profile_expr = if let Some(profile_str) = profile {
        match profile_str.to_lowercase().as_str() {
            "creative" => quote! { #crate_path::agent::ExecutionProfile::Creative },
            "balanced" => quote! { #crate_path::agent::ExecutionProfile::Balanced },
            "deterministic" => quote! { #crate_path::agent::ExecutionProfile::Deterministic },
            _ => quote! { #crate_path::agent::ExecutionProfile::Balanced }, // Default fallback
        }
    } else {
        quote! { #crate_path::agent::ExecutionProfile::default() }
    };

    let agent_init = match backend {
        "gemini" => {
            let mut builder = quote! { #crate_path::agent::impls::GeminiAgent::new() };

            if let Some(model_str) = model {
                builder = quote! { #builder.with_model_str(#model_str) };
            }

            builder = quote! { #builder.with_execution_profile(#profile_expr) };
            builder
        }
        _ => {
            // Default to Claude
            let mut builder = quote! { #crate_path::agent::impls::ClaudeCodeAgent::new() };

            if let Some(model_str) = model {
                builder = quote! { #builder.with_model_str(#model_str) };
            }

            builder = quote! { #builder.with_execution_profile(#profile_expr) };
            builder
        }
    };

    quote! {
        impl Default for #struct_name {
            fn default() -> Self {
                Self::new(#agent_init)
            }
        }
    }
}

/// Derive macro for implementing the Agent trait
///
/// # Usage
/// ```ignore
/// #[derive(Agent)]
/// #[agent(expertise = "Rust expert", output = "MyOutputType")]
/// struct MyAgent;
/// ```
#[proc_macro_derive(Agent, attributes(agent))]
pub fn derive_agent(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;

    // Parse #[agent(...)] attributes
    let agent_attrs = match parse_agent_attrs(&input.attrs) {
        Ok(attrs) => attrs,
        Err(e) => return e.to_compile_error().into(),
    };

    let expertise = agent_attrs
        .expertise
        .unwrap_or_else(|| String::from("general AI assistant"));
    let output_type = agent_attrs
        .output
        .unwrap_or_else(|| syn::parse_str::<syn::Type>("String").unwrap());
    let backend = agent_attrs
        .backend
        .unwrap_or_else(|| String::from("claude"));
    let model = agent_attrs.model;
    let _profile = agent_attrs.profile; // Not used in simple derive macro
    let max_retries = agent_attrs.max_retries.unwrap_or(3); // Default: 3 retries

    // Determine crate path
    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            // Even when it's the same crate, use absolute path to support examples/tests/bins
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Check if output type is String (no JSON enforcement needed)
    let output_type_str = quote!(#output_type).to_string().replace(" ", "");
    let is_string_output = output_type_str == "String" || output_type_str == "&str";

    // Generate enhanced expertise with JSON schema instruction
    let enhanced_expertise = if is_string_output {
        // Plain text output - no JSON enforcement
        quote! { #expertise }
    } else {
        // Structured output - try to use ToPrompt::prompt_schema(), fallback to type name
        let type_name = quote!(#output_type).to_string();
        quote! {
            {
                use std::sync::OnceLock;
                static EXPERTISE_CACHE: OnceLock<String> = OnceLock::new();

                EXPERTISE_CACHE.get_or_init(|| {
                    // Try to get detailed schema from ToPrompt
                    let schema = <#output_type as #crate_path::prompt::ToPrompt>::prompt_schema();

                    if schema.is_empty() {
                        // Fallback: type name only
                        format!(
                            concat!(
                                #expertise,
                                "\n\nIMPORTANT: You must respond with valid JSON matching the {} type structure. ",
                                "Do not include any text outside the JSON object."
                            ),
                            #type_name
                        )
                    } else {
                        // Use detailed schema from ToPrompt
                        format!(
                            concat!(
                                #expertise,
                                "\n\nIMPORTANT: Respond with valid JSON matching this schema:\n\n{}"
                            ),
                            schema
                        )
                    }
                }).as_str()
            }
        }
    };

    // Generate agent initialization code based on backend
    let agent_init = match backend.as_str() {
        "gemini" => {
            if let Some(model_str) = model {
                quote! {
                    use #crate_path::agent::impls::GeminiAgent;
                    let agent = GeminiAgent::new().with_model_str(#model_str);
                }
            } else {
                quote! {
                    use #crate_path::agent::impls::GeminiAgent;
                    let agent = GeminiAgent::new();
                }
            }
        }
        "claude" => {
            if let Some(model_str) = model {
                quote! {
                    use #crate_path::agent::impls::ClaudeCodeAgent;
                    let agent = ClaudeCodeAgent::new().with_model_str(#model_str);
                }
            } else {
                quote! {
                    use #crate_path::agent::impls::ClaudeCodeAgent;
                    let agent = ClaudeCodeAgent::new();
                }
            }
        }
        _ => {
            // Default to Claude
            if let Some(model_str) = model {
                quote! {
                    use #crate_path::agent::impls::ClaudeCodeAgent;
                    let agent = ClaudeCodeAgent::new().with_model_str(#model_str);
                }
            } else {
                quote! {
                    use #crate_path::agent::impls::ClaudeCodeAgent;
                    let agent = ClaudeCodeAgent::new();
                }
            }
        }
    };

    let expanded = quote! {
        #[async_trait::async_trait]
        impl #impl_generics #crate_path::agent::Agent for #struct_name #ty_generics #where_clause {
            type Output = #output_type;

            fn expertise(&self) -> &str {
                #enhanced_expertise
            }

            async fn execute(&self, intent: #crate_path::agent::Payload) -> Result<Self::Output, #crate_path::agent::AgentError> {
                // Create internal agent based on backend configuration
                #agent_init

                // Use the unified retry_execution function (DRY principle)
                let agent_ref = &agent;
                #crate_path::agent::retry::retry_execution(
                    #max_retries,
                    &intent,
                    move |payload| {
                        let payload = payload.clone();
                        async move {
                            // Execute and get response
                            let response = agent_ref.execute(payload).await?;

                            // Extract JSON from the response
                            let json_str = #crate_path::extract_json(&response)
                                .map_err(|e| #crate_path::agent::AgentError::ParseError {
                                    message: format!("Failed to extract JSON: {}", e),
                                    reason: #crate_path::agent::error::ParseErrorReason::MarkdownExtractionFailed,
                                })?;

                            // Deserialize into output type
                            serde_json::from_str::<Self::Output>(&json_str)
                                .map_err(|e| {
                                    // Determine the parse error reason based on serde_json error type
                                    let reason = if e.is_eof() {
                                        #crate_path::agent::error::ParseErrorReason::UnexpectedEof
                                    } else if e.is_syntax() {
                                        #crate_path::agent::error::ParseErrorReason::InvalidJson
                                    } else {
                                        #crate_path::agent::error::ParseErrorReason::SchemaMismatch
                                    };

                                    #crate_path::agent::AgentError::ParseError {
                                        message: format!("Failed to parse JSON: {}", e),
                                        reason,
                                    }
                                })
                        }
                    }
                ).await
            }

            async fn is_available(&self) -> Result<(), #crate_path::agent::AgentError> {
                // Create internal agent and check availability
                #agent_init
                agent.is_available().await
            }
        }
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Agent Attribute Macro (Generic version with injection support)
// ============================================================================

/// Attribute macro for implementing the Agent trait with Generic support
///
/// This version generates a struct definition with Generic inner agent,
/// allowing for agent injection and testing with mock agents.
///
/// # Usage
/// ```ignore
/// #[agent(expertise = "Rust expert", output = "MyOutputType")]
/// struct MyAgent;
/// ```
#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse attributes
    let agent_attrs = match syn::parse::<AgentAttrs>(attr) {
        Ok(attrs) => attrs,
        Err(e) => return e.to_compile_error().into(),
    };

    // Parse the input struct
    let input = parse_macro_input!(item as DeriveInput);
    let struct_name = &input.ident;
    let vis = &input.vis;

    let expertise = agent_attrs
        .expertise
        .unwrap_or_else(|| String::from("general AI assistant"));
    let output_type = agent_attrs
        .output
        .unwrap_or_else(|| syn::parse_str::<syn::Type>("String").unwrap());
    let backend = agent_attrs
        .backend
        .unwrap_or_else(|| String::from("claude"));
    let model = agent_attrs.model;
    let profile = agent_attrs.profile;

    // Check if output type is String (no JSON enforcement needed)
    let output_type_str = quote!(#output_type).to_string().replace(" ", "");
    let is_string_output = output_type_str == "String" || output_type_str == "&str";

    // Determine crate path
    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Determine generic parameter name for inner agent (default: "A")
    let inner_generic_name = agent_attrs.inner.unwrap_or_else(|| String::from("A"));
    let inner_generic_ident = syn::Ident::new(&inner_generic_name, proc_macro2::Span::call_site());

    // Determine default agent type - prioritize default_inner, fallback to backend
    let default_agent_type = if let Some(ref custom_type) = agent_attrs.default_inner {
        // Custom type specified via default_inner attribute
        let type_path: syn::Type =
            syn::parse_str(custom_type).expect("default_inner must be a valid type path");
        quote! { #type_path }
    } else {
        // Use backend to determine default type
        match backend.as_str() {
            "gemini" => quote! { #crate_path::agent::impls::GeminiAgent },
            _ => quote! { #crate_path::agent::impls::ClaudeCodeAgent },
        }
    };

    // Generate struct definition
    let struct_def = quote! {
        #vis struct #struct_name<#inner_generic_ident = #default_agent_type> {
            inner: #inner_generic_ident,
        }
    };

    // Generate basic constructor
    let constructors = quote! {
        impl<#inner_generic_ident> #struct_name<#inner_generic_ident> {
            /// Create a new agent with a custom inner agent implementation
            pub fn new(inner: #inner_generic_ident) -> Self {
                Self { inner }
            }
        }
    };

    // Generate backend-specific constructors and Default implementation
    let (backend_constructors, default_impl) = if agent_attrs.default_inner.is_some() {
        // Custom type - generate Default impl for the default type
        let default_impl = quote! {
            impl Default for #struct_name {
                fn default() -> Self {
                    Self {
                        inner: <#default_agent_type as Default>::default(),
                    }
                }
            }
        };
        (quote! {}, default_impl)
    } else {
        // Built-in backend - generate backend-specific constructors
        let backend_constructors = generate_backend_constructors(
            struct_name,
            &backend,
            model.as_deref(),
            profile.as_deref(),
            &crate_path,
        );
        let default_impl = generate_default_impl(
            struct_name,
            &backend,
            model.as_deref(),
            profile.as_deref(),
            &crate_path,
        );
        (backend_constructors, default_impl)
    };

    // Generate enhanced expertise with JSON schema instruction (same as derive macro)
    let enhanced_expertise = if is_string_output {
        // Plain text output - no JSON enforcement
        quote! { #expertise }
    } else {
        // Structured output - try to use ToPrompt::prompt_schema(), fallback to type name
        let type_name = quote!(#output_type).to_string();
        quote! {
            {
                use std::sync::OnceLock;
                static EXPERTISE_CACHE: OnceLock<String> = OnceLock::new();

                EXPERTISE_CACHE.get_or_init(|| {
                    // Try to get detailed schema from ToPrompt
                    let schema = <#output_type as #crate_path::prompt::ToPrompt>::prompt_schema();

                    if schema.is_empty() {
                        // Fallback: type name only
                        format!(
                            concat!(
                                #expertise,
                                "\n\nIMPORTANT: You must respond with valid JSON matching the {} type structure. ",
                                "Do not include any text outside the JSON object."
                            ),
                            #type_name
                        )
                    } else {
                        // Use detailed schema from ToPrompt
                        format!(
                            concat!(
                                #expertise,
                                "\n\nIMPORTANT: Respond with valid JSON matching this schema:\n\n{}"
                            ),
                            schema
                        )
                    }
                }).as_str()
            }
        }
    };

    // Generate Agent trait implementation
    let agent_impl = quote! {
        #[async_trait::async_trait]
        impl<#inner_generic_ident> #crate_path::agent::Agent for #struct_name<#inner_generic_ident>
        where
            #inner_generic_ident: #crate_path::agent::Agent<Output = String>,
        {
            type Output = #output_type;

            fn expertise(&self) -> &str {
                #enhanced_expertise
            }

            async fn execute(&self, intent: #crate_path::agent::Payload) -> Result<Self::Output, #crate_path::agent::AgentError> {
                // Prepend expertise to the payload
                let enhanced_payload = intent.prepend_text(self.expertise());

                // Use the inner agent with the enhanced payload
                let response = self.inner.execute(enhanced_payload).await?;

                // Extract JSON from the response
                let json_str = #crate_path::extract_json(&response)
                    .map_err(|e| #crate_path::agent::AgentError::ParseError {
                        message: e.to_string(),
                        reason: #crate_path::agent::error::ParseErrorReason::MarkdownExtractionFailed,
                    })?;

                // Deserialize into output type
                serde_json::from_str(&json_str).map_err(|e| {
                    let reason = if e.is_eof() {
                        #crate_path::agent::error::ParseErrorReason::UnexpectedEof
                    } else if e.is_syntax() {
                        #crate_path::agent::error::ParseErrorReason::InvalidJson
                    } else {
                        #crate_path::agent::error::ParseErrorReason::SchemaMismatch
                    };
                    #crate_path::agent::AgentError::ParseError {
                        message: e.to_string(),
                        reason,
                    }
                })
            }

            async fn is_available(&self) -> Result<(), #crate_path::agent::AgentError> {
                self.inner.is_available().await
            }
        }
    };

    let expanded = quote! {
        #struct_def
        #constructors
        #backend_constructors
        #default_impl
        #agent_impl
    };

    TokenStream::from(expanded)
}

/// Derive macro for TypeMarker trait.
///
/// Automatically implements the TypeMarker trait and adds a `__type` field
/// with a default value based on the struct name.
///
/// # Example
///
/// ```ignore
/// use llm_toolkit::orchestrator::TypeMarker;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize, TypeMarker)]
/// pub struct HighConceptResponse {
///     pub reasoning: String,
///     pub high_concept: String,
/// }
///
/// // Expands to:
/// // - Adds __type: String field with #[serde(default = "...")]
/// // - Implements TypeMarker with TYPE_NAME = "HighConceptResponse"
/// ```
#[proc_macro_derive(TypeMarker)]
pub fn derive_type_marker(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let type_name_str = struct_name.to_string();

    // Get the crate path for llm_toolkit
    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    let expanded = quote! {
        impl #crate_path::orchestrator::TypeMarker for #struct_name {
            const TYPE_NAME: &'static str = #type_name_str;
        }
    };

    TokenStream::from(expanded)
}

/// Attribute macro that adds a `__type` field to a struct and implements TypeMarker.
///
/// This macro transforms a struct by:
/// 1. Adding a `__type: String` field with `#[serde(default = "...", skip_serializing)]`
/// 2. Generating a default function that returns the struct's type name
/// 3. Implementing the `TypeMarker` trait
///
/// # Example
///
/// ```ignore
/// use llm_toolkit_macros::type_marker;
/// use serde::{Serialize, Deserialize};
///
/// #[type_marker]
/// #[derive(Serialize, Deserialize, Debug)]
/// pub struct WorldConceptResponse {
///     pub concept: String,
/// }
///
/// // Expands to:
/// #[derive(Serialize, Deserialize, Debug)]
/// pub struct WorldConceptResponse {
///     #[serde(default = "default_world_concept_response_type", skip_serializing)]
///     __type: String,
///     pub concept: String,
/// }
///
/// fn default_world_concept_response_type() -> String {
///     "WorldConceptResponse".to_string()
/// }
///
/// impl TypeMarker for WorldConceptResponse {
///     const TYPE_NAME: &'static str = "WorldConceptResponse";
/// }
/// ```
#[proc_macro_attribute]
pub fn type_marker(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);
    let struct_name = &input.ident;
    let vis = &input.vis;
    let type_name_str = struct_name.to_string();

    // Generate default function name (snake_case)
    let default_fn_name = syn::Ident::new(
        &format!("default_{}_type", to_snake_case(&type_name_str)),
        struct_name.span(),
    );

    // Get the crate path for llm_toolkit
    let found_crate =
        crate_name("llm-toolkit").expect("llm-toolkit should be present in `Cargo.toml`");
    let crate_path = match found_crate {
        FoundCrate::Itself => {
            let ident = syn::Ident::new("llm_toolkit", proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    // Extract struct fields
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => match &data_struct.fields {
            syn::Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    struct_name,
                    "type_marker only works with structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(struct_name, "type_marker only works with structs")
                .to_compile_error()
                .into();
        }
    };

    // Create new fields with __type prepended
    let mut new_fields = vec![];

    // Convert function name to string literal for serde attribute
    let default_fn_name_str = default_fn_name.to_string();
    let default_fn_name_lit = syn::LitStr::new(&default_fn_name_str, default_fn_name.span());

    // Add __type field first
    // Note: We don't use skip_serializing here because:
    // 1. ToPrompt already excludes __type from LLM prompts at macro generation time
    // 2. Orchestrator needs __type in serialized JSON for type-based retrieval (get_typed_output)
    new_fields.push(quote! {
        #[serde(default = #default_fn_name_lit)]
        __type: String
    });

    // Add original fields
    for field in fields {
        new_fields.push(quote! { #field });
    }

    // Get original attributes (like #[derive(...)])
    let attrs = &input.attrs;
    let generics = &input.generics;

    let expanded = quote! {
        // Generate the default function
        fn #default_fn_name() -> String {
            #type_name_str.to_string()
        }

        // Generate the struct with __type field
        #(#attrs)*
        #vis struct #struct_name #generics {
            #(#new_fields),*
        }

        // Implement TypeMarker trait
        impl #crate_path::orchestrator::TypeMarker for #struct_name {
            const TYPE_NAME: &'static str = #type_name_str;
        }
    };

    TokenStream::from(expanded)
}
