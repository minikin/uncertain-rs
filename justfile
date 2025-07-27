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

# Development workflow (format + lint + test)
dev: fmt lint test

# Clean build artifacts (including nextest cache)
clean:
    cargo clean
    rm -rf target/nextest
