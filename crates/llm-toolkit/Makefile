.PHONY: preflight

preflight:
	cargo fmt
	cargo clippy --allow-dirty --allow-staged --fix --quiet
	cargo build --quiet
	cargo test
