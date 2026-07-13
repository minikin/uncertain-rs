# Default recipe - shows available commands
default:
    @just --list

# Run the server
run:
    cargo run

# Format code (applies changes)
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Run linting
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Run tests (default features)
test:
    cargo test --all-targets

# Run tests with the parallel feature enabled
test-parallel:
    cargo test --all-targets --features parallel

# Run doc tests
test-doc:
    cargo test --doc --all-features

# Security audit - check for known vulnerabilities in dependencies
audit:
    cargo audit

# License / bans / sources / advisories policy (see deny.toml)
deny:
    cargo deny check

# Dogfood: score this crate's own coverage/complexity, gated on regressions vs crap_baseline.json
crap:
    mkdir -p target
    cargo llvm-cov --all-features --workspace --lcov --output-path target/lcov.info
    cargo crap --lcov target/lcov.info --workspace --baseline crap_baseline.json --fail-regression

# Regenerate crap_baseline.json after an intentional complexity/coverage improvement
crap-update-baseline:
    mkdir -p target
    cargo llvm-cov --all-features --workspace --lcov --output-path target/lcov.info
    cargo crap --lcov target/lcov.info --workspace --format json --sort file --output crap_baseline.json

# Pre-commit gate. Exception: changes touching only .md/.yml files may skip this and dev-mutants-diff.
dev: fmt-check lint test test-parallel test-doc audit deny crap

# Mutation testing on files changed vs main (re-runs dev first)
dev-mutants-diff: dev
    mkdir -p target
    git diff origin/main...HEAD -- '*.rs' > target/mutants-diff.patch
    cargo mutants --in-diff target/mutants-diff.patch

# Full mutation-testing run across the whole crate (slow)
mutants:
    cargo mutants

# Run the criterion benchmark suite
bench:
    cargo bench

# Clean build artifacts (including nextest cache)
clean:
    cargo clean
    rm -rf target/nextest
