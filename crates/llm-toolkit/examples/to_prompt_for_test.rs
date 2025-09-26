#[cfg(feature = "derive")]
mod test {
    use llm_toolkit::prompt::{ToPrompt as ToPromptTrait, ToPromptFor as ToPromptForTrait};
    use llm_toolkit::{ToPrompt, ToPromptFor};
    use serde::Serialize;

    // Target type for ToPromptFor
    #[derive(Serialize)]
    struct Target {
        id: String,
    }

    // Test struct with ToPromptFor implementation
    #[derive(Serialize, ToPrompt, ToPromptFor, Default)]
    #[prompt_for(
        target = "Target",
        template = "Struct {name} with value {value} for {self}"
    )]
    struct ConfigForTarget {
        name: String,
        value: i32,
    }

    // Test with mode specifiers
    #[derive(Serialize, ToPrompt, ToPromptFor, Default)]
    #[prompt_for(
        target = "Target",
        template = "Schema: {self:schema_only}\n\nExample: {self:example_only}"
    )]
    #[prompt(mode = "full")]
    struct ConfigWithModes {
        field1: String,
        field2: u32,
    }

    pub fn run_test() {
        // Test basic ToPromptFor
        let config = ConfigForTarget {
            name: "test".to_string(),
            value: 42,
        };

        let target = Target {
            id: "target-1".to_string(),
        };

        let prompt = config.to_prompt_for_with_mode(&target, "full");
        println!("Basic ToPromptFor output:\n{}\n", prompt);

        // Test with mode specifiers
        let config_modes = ConfigWithModes {
            field1: "hello".to_string(),
            field2: 123,
        };

        let prompt_modes = config_modes.to_prompt_for_with_mode(&target, "full");
        println!("ToPromptFor with modes output:\n{}\n", prompt_modes);
    }
}

#[cfg(feature = "derive")]
fn main() {
    test::run_test();
}

#[cfg(not(feature = "derive"))]
fn main() {
    println!("This example requires the 'derive' feature to be enabled.");
}
