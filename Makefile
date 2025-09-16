.PHONY: preflight publish

# Run checks for all workspace members
preflight:
	@echo "Running preflight checks for the entire workspace..."
	cargo fmt --all
	cargo clippy --all-targets -- -D warnings
	cargo test --all-targets

# Publish all workspace members in the correct order
publish: preflight
	@echo "--- Publishing llm-toolkit-macros ---"
	cargo publish -p llm-toolkit-macros

	@echo "\n--- Waiting 10 seconds for crates.io index to update... "
	sleep 10

	@echo "\n--- Publishing llm-toolkit ---"
	cargo publish -p llm-toolkit

	@echo "\n✅ Successfully published all crates."
