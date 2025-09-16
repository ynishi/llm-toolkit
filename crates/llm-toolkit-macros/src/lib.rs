use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Meta, parse_macro_input, punctuated::Punctuated};

#[proc_macro_derive(ToPrompt, attributes(prompt))]
pub fn to_prompt_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let attr = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("prompt"))
        .expect("`#[derive(ToPrompt)]` requires a `#[prompt(...)]` attribute.");

    // `syn::Attribute::parse_args_with` を使って属性をパースする
    let name_value = attr
        .parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        .expect("Failed to parse `prompt` attribute arguments")
        .into_iter()
        .find_map(|meta| match meta {
            Meta::NameValue(nv) if nv.path.is_ident("template") => Some(nv),
            _ => None,
        })
        .expect("`#[prompt(...)]` must contain `template = \"...\"`");

    let template_str = if let syn::Expr::Lit(expr_lit) = name_value.value {
        if let syn::Lit::Str(lit_str) = expr_lit.lit {
            lit_str.value()
        } else {
            panic!("'template' attribute value must be a string literal.");
        }
    } else {
        panic!("'template' attribute must have a literal value.");
    };

    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics llm_toolkit::prompt::ToPrompt for #name #ty_generics #where_clause {
            fn to_prompt(&self) -> String {
                llm_toolkit::prompt::render_prompt(#template_str, self).unwrap_or_else(|e| {
                    format!("Failed to render prompt: {}", e)
                })
            }
        }
    };

    TokenStream::from(expanded)
}
