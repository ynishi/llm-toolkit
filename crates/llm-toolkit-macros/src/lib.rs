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
) -> proc_macro2::TokenStream {
    let mut schema_lines = vec![];

    // Add header
    if !struct_docs.is_empty() {
        schema_lines.push(format!("### Schema for `{}`\n{}", struct_name, struct_docs));
    } else {
        schema_lines.push(format!("### Schema for `{}`", struct_name));
    }

    schema_lines.push("{".to_string());

    // Process fields
    for (i, field) in fields.iter().enumerate() {
        let field_name = field.ident.as_ref().unwrap();
        let attrs = parse_field_prompt_attrs(&field.attrs);

        // Skip if marked to skip
        if attrs.skip {
            continue;
        }

        // Get field documentation
        let field_docs = extract_doc_comments(&field.attrs);

        // Determine the type representation
        let type_str = format_type_for_schema(&field.ty);

        // Build field line
        let mut field_line = format!("  \"{}\": \"{}\"", field_name, type_str);

        // Add comment if there's documentation
        if !field_docs.is_empty() {
            field_line.push_str(&format!(", // {}", field_docs));
        }

        // Add comma if not last field (accounting for skipped fields)
        let remaining_fields = fields
            .iter()
            .skip(i + 1)
            .filter(|f| {
                let attrs = parse_field_prompt_attrs(&f.attrs);
                !attrs.skip
            })
            .count();

        if remaining_fields > 0 {
            field_line.push(',');
        }

        schema_lines.push(field_line);
    }

    schema_lines.push("}".to_string());

    let schema_str = schema_lines.join("\n");

    quote! {
        vec![#crate_path::prompt::PromptPart::Text(#schema_str.to_string())]
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

/// Result of parsing prompt attribute
enum PromptAttribute {
    Skip,
    Description(String),
    None,
}

/// Parse #[prompt(...)] attribute on enum variant
fn parse_prompt_attribute(attrs: &[syn::Attribute]) -> PromptAttribute {
    for attr in attrs {
        if attr.path().is_ident("prompt") {
            // Check for #[prompt(skip)]
            if let Ok(meta_list) = attr.meta.require_list() {
                let tokens = &meta_list.tokens;
                let tokens_str = tokens.to_string();
                if tokens_str == "skip" {
                    return PromptAttribute::Skip;
                }
            }

            // Check for #[prompt("description")]
            if let Ok(lit_str) = attr.parse_args::<syn::LitStr>() {
                return PromptAttribute::Description(lit_str.value());
            }
        }
    }
    PromptAttribute::None
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

            let mut prompt_lines = Vec::new();

            // Add enum description
            if !enum_docs.is_empty() {
                prompt_lines.push(format!("{}: {}", enum_name, enum_docs));
            } else {
                prompt_lines.push(format!("{}:", enum_name));
            }
            prompt_lines.push(String::new()); // Empty line
            prompt_lines.push("Possible values:".to_string());

            // Add each variant with its documentation based on priority
            for variant in &data_enum.variants {
                let variant_name = &variant.ident;

                // Apply fallback logic with priority
                match parse_prompt_attribute(&variant.attrs) {
                    PromptAttribute::Skip => {
                        // Skip this variant completely
                        continue;
                    }
                    PromptAttribute::Description(desc) => {
                        // Use custom description from #[prompt("...")]
                        prompt_lines.push(format!("- {}: {}", variant_name, desc));
                    }
                    PromptAttribute::None => {
                        // Fall back to doc comment or just variant name
                        let variant_docs = extract_doc_comments(&variant.attrs);
                        if !variant_docs.is_empty() {
                            prompt_lines.push(format!("- {}: {}", variant_name, variant_docs));
                        } else {
                            prompt_lines.push(format!("- {}", variant_name));
                        }
                    }
                }
            }

            let prompt_string = prompt_lines.join("\n");
            let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

            let expanded = quote! {
                impl #impl_generics #crate_path::prompt::ToPrompt for #enum_name #ty_generics #where_clause {
                    fn to_prompt_parts(&self) -> Vec<#crate_path::prompt::PromptPart> {
                        vec![#crate_path::prompt::PromptPart::Text(#prompt_string.to_string())]
                    }

                    fn to_prompt(&self) -> String {
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

                // If we still haven't found the file, use the original path for a better error message
                let final_path =
                    full_path.unwrap_or_else(|| std::path::Path::new(&file_path).to_path_buf());

                // Read the file at compile time
                match std::fs::read_to_string(&final_path) {
                    Ok(content) => Some(content),
                    Err(e) => {
                        return syn::Error::new(
                            input.ident.span(),
                            format!(
                                "Failed to read template file '{}': {}",
                                final_path.display(),
                                e
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

                // Generate schema-only parts
                let schema_parts =
                    generate_schema_only_parts(&struct_name_str, &struct_docs, fields, &crate_path);

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
}

impl Parse for AgentAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut expertise = None;
        let mut output = None;

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
                _ => {}
            }
        }

        Ok(AgentAttrs { expertise, output })
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
    })
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

    let expanded = quote! {
        #[async_trait::async_trait]
        impl #impl_generics #crate_path::agent::Agent for #struct_name #ty_generics #where_clause {
            type Output = #output_type;

            fn expertise(&self) -> &str {
                #expertise
            }

            async fn execute(&self, intent: String) -> Result<Self::Output, #crate_path::agent::AgentError> {
                use #crate_path::agent::impls::ClaudeCodeAgent;

                // Create internal ClaudeCodeAgent
                let agent = ClaudeCodeAgent::new();

                // Execute and get response
                let response = agent.execute(intent).await?;

                // Extract JSON from the response
                let json_str = #crate_path::extract_json(&response)
                    .map_err(|e| #crate_path::agent::AgentError::ParseError(e.to_string()))?;

                // Deserialize into output type
                serde_json::from_str(&json_str)
                    .map_err(|e| #crate_path::agent::AgentError::ParseError(e.to_string()))
            }
        }
    };

    TokenStream::from(expanded)
}
