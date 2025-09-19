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
}

/// Parse #[prompt(...)] attributes for struct fields
fn parse_field_prompt_attrs(attrs: &[syn::Attribute]) -> FieldPromptAttrs {
    let mut result = FieldPromptAttrs::default();
    
    for attr in attrs {
        if attr.path().is_ident("prompt") {
            // Try to parse as meta list #[prompt(key = value, ...)]
            if let Ok(meta_list) = attr.meta.require_list() {
                // Parse the tokens inside the parentheses
                if let Ok(metas) = meta_list.parse_args_with(
                    Punctuated::<Meta, syn::Token![,]>::parse_terminated
                ) {
                    for meta in metas {
                        match meta {
                            Meta::Path(path) if path.is_ident("skip") => {
                                result.skip = true;
                            }
                            Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value {
                                    result.rename = Some(lit_str.value());
                                }
                            }
                            Meta::NameValue(nv) if nv.path.is_ident("format_with") => {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit_str),
                                    ..
                                }) = nv.value {
                                    result.format_with = Some(lit_str.value());
                                }
                            }
                            _ => {}
                        }
                    }
                } else if meta_list.tokens.to_string() == "skip" {
                    // Handle simple #[prompt(skip)] case
                    result.skip = true;
                }
            }
        }
    }
    
    result
}

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
                    fn to_prompt(&self) -> String {
                        #prompt_string.to_string()
                    }
                }
            };

            TokenStream::from(expanded)
        }
        Data::Struct(data_struct) => {
            // Check if there's a #[prompt(template = "...")] attribute
            let template_attr = input
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("prompt"))
                .and_then(|attr| {
                    // Try to parse the attribute arguments
                    attr.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
                        .ok()
                        .and_then(|metas| {
                            metas.into_iter().find_map(|meta| match meta {
                                Meta::NameValue(nv) if nv.path.is_ident("template") => {
                                    if let syn::Expr::Lit(expr_lit) = nv.value {
                                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                                            Some(lit_str.value())
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            })
                        })
                });

            let name = input.ident;
            let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

            let expanded = if let Some(template_str) = template_attr {
                // Use template-based approach if template is provided
                quote! {
                    impl #impl_generics llm_toolkit::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt(&self) -> String {
                            llm_toolkit::prompt::render_prompt(#template_str, self).unwrap_or_else(|e| {
                                format!("Failed to render prompt: {}", e)
                            })
                        }
                    }
                }
            } else {
                // Use default key-value format if no template is provided
                let fields = if let syn::Fields::Named(fields) = &data_struct.fields {
                    &fields.named
                } else {
                    panic!("Default prompt generation is only supported for structs with named fields.");
                };

                let field_prompts: Vec<_> = fields.iter()
                    .filter_map(|f| {
                        let field_name = f.ident.as_ref().unwrap();
                        let attrs = parse_field_prompt_attrs(&f.attrs);
                        
                        // Skip if #[prompt(skip)] is present
                        if attrs.skip {
                            return None;
                        }
                        
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
                            let func_path: syn::Path = syn::parse_str(&format_with).expect(&format!("Invalid function path: {}", format_with));
                            quote! { #func_path(&self.#field_name) }
                        } else {
                            quote! { self.#field_name.to_prompt() }
                        };
                        
                        Some(quote! {
                            format!("{}: {}", #key, #value_expr)
                        })
                    })
                    .collect();

                quote! {
                    impl #impl_generics llm_toolkit::prompt::ToPrompt for #name #ty_generics #where_clause {
                        fn to_prompt(&self) -> String {
                            let mut parts = Vec::new();
                            #(
                                parts.push(#field_prompts);
                            )*
                            parts.join("\n")
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
