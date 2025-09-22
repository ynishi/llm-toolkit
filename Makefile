.PHONY: preflight publish test-example-derive-prompt-enum

# Test derive_prompt_enum example
test-example-derive-prompt-enum:
	@echo "Testing derive_prompt_enum example..."
	cargo run --example derive_prompt_enum --package llm-toolkit --features="derive"

# Run checks for all workspace members
preflight: test-example-derive-prompt-enum
	@echo "Running preflight checks for the entire workspace..."
	cargo fmt --all
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test --all-targets --all-features 

# Publish all workspace members in the correct order with atomic dry-run -> publish sequences
publish: preflight
	@echo "\n🚀 Starting sequential publish process...\n"

	@echo "--- Step 1: Publishing llm-toolkit-macros ---"
	@echo "  Running dry-run for llm-toolkit-macros..."
	cargo publish -p llm-toolkit-macros --dry-run --allow-dirty

	@echo "  ✓ Dry-run successful for llm-toolkit-macros"
	@echo "  Publishing llm-toolkit-macros to crates.io..."
	cargo publish -p llm-toolkit-macros --allow-dirty

	@echo "\n✅ llm-toolkit-macros published successfully!"
	@echo "\n⏳ Waiting 10 seconds for crates.io index to update..."
	sleep 10

	@echo "\n--- Step 2: Publishing llm-toolkit ---"
	@echo "  Running dry-run for llm-toolkit..."
	cargo publish -p llm-toolkit --dry-run --allow-dirty

	@echo "  ✓ Dry-run successful for llm-toolkit"
	@echo "  Publishing llm-toolkit to crates.io..."
	cargo publish -p llm-toolkit --allow-dirty

	@echo "\n✅ llm-toolkit published successfully!"
	@echo "\n🎉 All crates have been successfully published to crates.io!"
