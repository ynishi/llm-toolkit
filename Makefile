.PHONY: preflight publish test-examples test-examples-offline

# 全Example（外部API依存含む）
EXAMPLES := \
	derive_prompt_enum \
	derive_prompt_for \
	derive_prompt_format_with \
	derive_prompt_set \
	derive_prompt \
	examples_section_test \
	multimodal_prompt \
	to_prompt_for_test \
	external_template \
	define_intent_comprehensive \
	test_define_intent \
	orchestrator_with_mock \
	orchestrator_fast_path_e2e \
	orchestrator_get_typed_output \
	check_claude \
	agent_derive \
	agent_auto_json_test \
	agent_with_toprompt_schema \
	agent_retry_test \
	agent_backend_switching \
	agent_model_selection \
	agent_attribute \
	execution_profile_demo \
	agent_with_profile \
	agent_string_output_test \
	type_marker_schema_test \
	test_type_output \
	orchestrator_streaming

# 外部API依存なしのExample（E2Eテストとして実行可能）
OFFLINE_EXAMPLES := $(filter-out orchestrator_streaming,$(EXAMPLES))

# 全Exampleを実行（外部API依存含む）
test-examples:
	@for name in $(EXAMPLES); do \
		echo "Running $$name example..."; \
		cargo run --example $$name --package llm-toolkit --features="derive agent"; \
	done

# 外部API依存なしのExampleのみを実行（E2Eテスト）
test-examples-offline:
	@echo "🧪 Running offline examples (E2E tests - no external API dependencies)..."
	@for name in $(OFFLINE_EXAMPLES); do \
		echo ""; \
		echo "▶ Running $$name..."; \
		cargo run --example $$name --package llm-toolkit --features="derive agent" || exit 1; \
	done
	@echo ""
	@echo "✅ All offline examples passed!"

# Run checks for all workspace members
preflight: test-examples-offline
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
