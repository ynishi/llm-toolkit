#[cfg(feature = "derive")]
mod tests {
    use llm_toolkit::ToPrompt;

    // Custom formatting function in the crate root (needed for macro to find it)
    pub fn format_temperature(temp: &f32) -> String {
        format!("{:.1}°C", temp)
    }

    pub fn format_percentage(value: &f32) -> String {
        format!("{:.0}%", value * 100.0)
    }

    pub fn format_list(items: &Vec<String>) -> String {
        items.join(", ")
    }

    pub fn uppercase_format(s: &String) -> String {
        s.to_uppercase()
    }

    #[test]
    fn test_format_with_attribute() {
        #[derive(ToPrompt)]
        struct WeatherReport {
            /// Current temperature
            #[prompt(format_with = "super::tests::format_temperature")]
            temperature: f32,

            /// Humidity level
            #[prompt(format_with = "super::tests::format_percentage")]
            humidity: f32,

            /// Weather conditions
            #[prompt(rename = "conditions", format_with = "super::tests::format_list")]
            weather_conditions: Vec<String>,

            // Normal field without custom formatting
            location: String,

            #[prompt(skip)]
            internal_id: u32,
        }

        let report = WeatherReport {
            temperature: 23.456,
            humidity: 0.65,
            weather_conditions: vec!["Partly cloudy".to_string(), "Light wind".to_string()],
            location: "Tokyo".to_string(),
            internal_id: 12345,
        };

        let prompt = report.to_prompt();

        // Check that custom formatters were applied
        assert!(
            prompt.contains("Current temperature: 23.5°C"),
            "Temperature should be formatted with format_temperature: {}",
            prompt
        );
        assert!(
            prompt.contains("Humidity level: 65%"),
            "Humidity should be formatted with format_percentage: {}",
            prompt
        );
        assert!(
            prompt.contains("conditions: Partly cloudy, Light wind"),
            "Weather conditions should use renamed key and custom formatter: {}",
            prompt
        );
        assert!(
            prompt.contains("location: Tokyo"),
            "Location should use default to_prompt: {}",
            prompt
        );
        assert!(
            !prompt.contains("internal_id"),
            "internal_id should be skipped: {}",
            prompt
        );
    }

    #[test]
    fn test_all_attributes_together() {
        // This test ensures that skip, rename, and format_with work together
        #[derive(ToPrompt)]
        struct TestStruct {
            #[prompt(skip)]
            ignored: String,

            #[prompt(rename = "custom_name")]
            renamed_field: String,

            #[prompt(format_with = "super::tests::uppercase_format")]
            formatted_field: String,

            #[prompt(rename = "special", format_with = "super::tests::uppercase_format")]
            both_rename_and_format: String,
        }

        let test = TestStruct {
            ignored: "should not appear".to_string(),
            renamed_field: "renamed value".to_string(),
            formatted_field: "hello".to_string(),
            both_rename_and_format: "world".to_string(),
        };

        let prompt = test.to_prompt();

        assert!(!prompt.contains("should not appear"));
        assert!(!prompt.contains("ignored"));
        assert!(prompt.contains("custom_name: renamed value"));
        assert!(prompt.contains("formatted_field: HELLO"));
        assert!(prompt.contains("special: WORLD"));
    }
}
