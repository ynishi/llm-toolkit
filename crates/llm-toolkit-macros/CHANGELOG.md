# Changelog

All notable changes to `llm-toolkit-macros` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deprecated

- **`#[derive(Agent)]` macro is now deprecated** and will be removed in v0.60.0 (Q2 2025)
  - **Critical Issue**: This macro does NOT inject expertise into prompts, meaning the LLM never sees your expertise definition
  - **Migration**: Remove `#[derive(Agent)]` and use only `#[agent(...)]` attribute macro
  - See migration guide in macro documentation for details

### Added

- Comprehensive deprecation documentation with migration guide for `#[derive(Agent)]`
- Clear warning messages when using deprecated macro

## [0.58.0] - 2024-12-11

### Changed

- Documented the difference between `#[derive(Agent)]` and `#[agent(...)]` macros
- Updated README with deprecation warnings and migration guides

## Previous Versions

See git history for changes prior to this CHANGELOG.
