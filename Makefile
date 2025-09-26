.PHONY: preflight publish test-example-derive-prompt-enum

# 実行したいExampleの名前をリストとして変数に定義
EXAMPLES := \
	derive_prompt_enum \
	derive_prompt_for \
	derive_prompt_format_with \
	derive_prompt_set \
	derive_prompt \
	examples_section_test \
	multimodal_prompt \
	to_prompt_for_test

# test-examplesターゲットで、リストをループ処理する
test-examples:
	@for name in $(EXAMPLES); do \
		echo "Running $$name example..."; \
		cargo run --example $$name --package llm-toolkit --features="derive"; \
	done

# Run checks for all workspace members
preflight: test-examples
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
