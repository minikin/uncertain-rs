# Default recipe - shows available commands
default:
    @just --list

# Run the server
run:
    cargo run

# Format code
fmt:
    cargo fmt

# Run linting
lint:
    cargo clippy -- -D warnings

# Run tests
test:
    cargo test

# Security audit - check for known vulnerabilities in dependencies
audit:
    cargo audit

# Development workflow (format + lint + test + audit)
dev: fmt lint test audit

# Clean build artifacts (including nextest cache)
clean:
    cargo clean
    rm -rf target/nextest
