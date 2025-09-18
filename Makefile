.PHONY: preflight publish test-example-derive-prompt-enum

# Test derive_prompt_enum example
test-example-derive-prompt-enum:
	@echo "Testing derive_prompt_enum example..."
	cargo run --example derive_prompt_enum --package llm-toolkit --features="derive"

# Run checks for all workspace members
preflight: test-example-derive-prompt-enum
	@echo "Running preflight checks for the entire workspace..."
	cargo fmt --all
	cargo clippy --all-targets -- -D warnings
	cargo test --all-targets

# Publish all workspace members in the correct order
publish: preflight
	@echo "--- Running dry-run for llm-toolkit-macros ---"
	cargo publish -p llm-toolkit-macros --dry-run

	@echo "\n--- Running dry-run for llm-toolkit ---"
	cargo publish -p llm-toolkit --dry-run

	@echo "\n✅ Dry-run successful. Proceeding with actual publish...\n"

	@echo "--- Publishing llm-toolkit-macros ---"
	cargo publish -p llm-toolkit-macros

	@echo "\n--- Waiting 10 seconds for crates.io index to update... "
	sleep 10

	@echo "\n--- Publishing llm-toolkit ---"
	cargo publish -p llm-toolkit

	@echo "\n✅ Successfully published all crates."
