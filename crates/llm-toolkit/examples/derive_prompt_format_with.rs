use llm_toolkit::ToPrompt;

// Custom formatting functions
fn format_currency(amount: &f64) -> String {
    format!("${:.2}", amount)
}

fn format_percentage(value: &f64) -> String {
    format!("{:.1}%", value * 100.0)
}

fn format_tags(tags: &[String]) -> String {
    if tags.is_empty() {
        "None".to_string()
    } else {
        tags.join(", ")
    }
}

#[derive(ToPrompt)]
#[allow(dead_code)]
struct Product {
    /// Product name
    name: String,

    /// Price
    #[prompt(format_with = "format_currency")]
    price: f64,

    /// Discount rate
    #[prompt(rename = "discount", format_with = "format_percentage")]
    discount_rate: f64,

    /// Product tags
    #[prompt(format_with = "format_tags")]
    tags: Vec<String>,

    #[prompt(skip)]
    internal_sku: String,
}

fn main() {
    let product = Product {
        name: "Rust Programming Book".to_string(),
        price: 49.99,
        discount_rate: 0.15,
        tags: vec![
            "programming".to_string(),
            "rust".to_string(),
            "education".to_string(),
        ],
        internal_sku: "SKU-12345".to_string(),
    };

    println!("{}", product.to_prompt());
    println!();
    println!("// Note: The internal_sku field was skipped");
    println!("// The price was formatted with a custom currency formatter");
    println!("// The discount_rate was renamed to 'discount' and formatted as a percentage");
    println!("// The tags were formatted with a custom list formatter");
}
