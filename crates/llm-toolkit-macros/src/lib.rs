use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Meta, parse_macro_input, punctuated::Punctuated};

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
                vec![llm_toolkit::prompt::PromptPart::Text(json_str)]
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
                vec![llm_toolkit::prompt::PromptPart::Text(json_str)]
            }
        }
    }
}

/// Generate schema-only representation for a struct
fn generate_schema_only_parts(
    struct_name: &str,
    struct_docs: &str,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
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
        vec![llm_toolkit::prompt::PromptPart::Text(#schema_str.to_string())]
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
                impl #impl_generics llm_toolkit::prompt::ToPrompt for #enum_name #ty_generics #where_clause {
                    fn to_prompt_parts(&self) -> Vec<llm_toolkit::prompt::PromptPart> {
                        vec![llm_toolkit::prompt::PromptPart::Text(#prompt_string.to_string())]
                    }

                    fn to_prompt(&self) -> String {
                        #prompt_string.to_string()
                    }
                }
            };

            TokenStream::from(expanded)
        }
        Data::Struct(data_struct) => {
            // Parse struct-level prompt attributes for template and mode
            let mut template_attr = None;
            let mut mode_attr = None;

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
                                Meta::NameValue(nv) if nv.path.is_ident("mode") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value
                                        && let syn::Lit::Str(lit_str) = expr_lit.lit
                                    {
                                        mode_attr = Some(lit_str.value());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            let name = input.ident;
            let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

            // Extract struct name and doc comment for use in schema generation
            let struct_docs = extract_doc_comments(&input.attrs);

            // Check if this is a mode-based struct (mode attribute present)
            let is_mode_based =
                mode_attr.is_some() || (template_attr.is_none() && struct_docs.contains("mode"));

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
                    if attr.path().is_ident("derive") {
                        if let Ok(meta_list) = attr.meta.require_list() {
                            let tokens_str = meta_list.tokens.to_string();
                            tokens_str.contains("Default")
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                });

                // Generate schema-only parts
                let schema_parts =
                    generate_schema_only_parts(&struct_name_str, &struct_docs, fields);

                // Generate example parts
                let example_parts = generate_example_only_parts(fields, has_default);

                quote! {
                    impl #impl_generics llm_toolkit::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt_parts_with_mode(&self, mode: &str) -> Vec<llm_toolkit::prompt::PromptPart> {
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
                                    parts.push(llm_toolkit::prompt::PromptPart::Text("\n### Example".to_string()));
                                    parts.push(llm_toolkit::prompt::PromptPart::Text(
                                        format!("Here is an example of a valid `{}` object:", #struct_name_str)
                                    ));

                                    // Add example
                                    let example_parts = #example_parts;
                                    parts.extend(example_parts);

                                    parts
                                }
                            }
                        }

                        fn to_prompt_parts(&self) -> Vec<llm_toolkit::prompt::PromptPart> {
                            self.to_prompt_parts_with_mode("full")
                        }

                        fn to_prompt(&self) -> String {
                            self.to_prompt_parts()
                                .into_iter()
                                .filter_map(|part| match part {
                                    llm_toolkit::prompt::PromptPart::Text(text) => Some(text),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    }
                }
            } else if let Some(template_str) = template_attr {
                // Use template-based approach if template is provided
                // Collect image fields separately for to_prompt_parts()
                let fields = if let syn::Fields::Named(fields) = &data_struct.fields {
                    &fields.named
                } else {
                    panic!(
                        "Template prompt generation is only supported for structs with named fields."
                    );
                };

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

                quote! {
                    impl #impl_generics llm_toolkit::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt_parts(&self) -> Vec<llm_toolkit::prompt::PromptPart> {
                            let mut parts = Vec::new();

                            // Add image parts first
                            #(#image_field_parts)*

                            // Add the rendered template as text
                            let text = llm_toolkit::prompt::render_prompt(#template_str, self).unwrap_or_else(|e| {
                                format!("Failed to render prompt: {}", e)
                            });
                            if !text.is_empty() {
                                parts.push(llm_toolkit::prompt::PromptPart::Text(text));
                            }

                            parts
                        }

                        fn to_prompt(&self) -> String {
                            llm_toolkit::prompt::render_prompt(#template_str, self).unwrap_or_else(|e| {
                                format!("Failed to render prompt: {}", e)
                            })
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
                    impl #impl_generics llm_toolkit::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt_parts(&self) -> Vec<llm_toolkit::prompt::PromptPart> {
                            let mut parts = Vec::new();

                            // Add image parts first
                            #(#image_field_parts)*

                            // Collect text parts and add as a single text prompt part
                            let mut text_parts = Vec::new();
                            #(#text_field_parts)*

                            if !text_parts.is_empty() {
                                parts.push(llm_toolkit::prompt::PromptPart::Text(text_parts.join("\n")));
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

                    let text = llm_toolkit::prompt::render_prompt(#template_str, self)
                        .map_err(|e| llm_toolkit::prompt::PromptSetError::RenderFailed {
                            target: #target_name.to_string(),
                            source: e,
                        })?;

                    if !text.is_empty() {
                        parts.push(llm_toolkit::prompt::PromptPart::Text(text));
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
                        parts.push(llm_toolkit::prompt::PromptPart::Text(text_parts.join("\n")));
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
            Err(llm_toolkit::prompt::PromptSetError::TargetNotFound {
                target: target.to_string(),
                available,
            })
        }
    });

    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics llm_toolkit::prompt::ToPromptSet for #struct_name #ty_generics #where_clause {
            fn to_prompt_parts_for(&self, target: &str) -> Result<Vec<llm_toolkit::prompt::PromptPart>, llm_toolkit::prompt::PromptSetError> {
                match target {
                    #(#match_arms)*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
