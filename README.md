# uncertain-rs

[![CI](https://github.com/minikin/uncertain-rs/workflows/CI/badge.svg)](https://github.com/minikin/uncertain-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/minikin/uncertain-rs/graph/badge.svg?token=IKjgVbFAQk)](https://codecov.io/gh/minikin/uncertain-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library for uncertainty-aware programming, implementing the approach from
"Uncertain<T>: A First-Order Type for Uncertain Data" by Bornholt, Mytkowicz, and McKinley.

- [uncertain-rs](#uncertain-rs)
  - [Core Concept: Evidence-Based Conditionals](#core-concept-evidence-based-conditionals)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Advanced Features](#advanced-features)
    - [Graph Optimization](#graph-optimization)
  - [Development Workflow](#development-workflow)
    - [Security](#security)
  - [Contributing](#contributing)
  - [License](#license)

## Core Concept: Evidence-Based Conditionals

Instead of treating uncertain data as exact values (which leads to bugs),
this library uses evidence-based conditionals that account for uncertainty:

```rust
use uncertain_rs::Uncertain;

// Create uncertain values from probability distributions
let speed = Uncertain::normal(55.2, 5.0); // GPS reading with ±5 mph error

// Evidence-based conditional (returns Uncertain<bool>)
let speeding_evidence = speed.gt(60.0);

// Convert evidence to decision with confidence level
if speeding_evidence.probability_exceeds(0.95) {
    // Only act if 95% confident
    println!("Issue speeding ticket");
}
```

## Features

- **Evidence-based conditionals**: Comparisons return evidence, not boolean facts
- **Uncertainty propagation**: Arithmetic operations preserve uncertainty
- **Lazy evaluation**: Computation graphs built lazily for efficiency
- **Graph optimization**: Common subexpression elimination and caching for performance
- **SPRT hypothesis testing**: Sequential Probability Ratio Test for optimal sampling
- **Rich distributions**: Normal, uniform, exponential, binomial, categorical, etc.
- **Statistical analysis**: Mean, std dev, confidence intervals, CDF, etc.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
uncertain-rs = "0.1.0"
```

## Quick Start

```rust
use uncertain_rs::Uncertain;

fn main() {
    // Create uncertain values
    let x = Uncertain::normal(5.0, 1.0);
    let y = Uncertain::normal(3.0, 0.5);

    // Perform arithmetic operations
    let sum = x.clone() + y.clone();
    let product = x * y;

    // Sample from the distributions
    println!("Sum sample: {}", sum.sample());
    println!("Product sample: {}", product.sample());

    // Statistical analysis
    println!("Sum mean: {}", sum.expected_value(1000));
    println!("Sum std dev: {}", sum.standard_deviation(1000));
}
```

For more examples, see the [examples directory](examples).

## Advanced Features

### Graph Optimization

The library includes a computation graph optimizer that can eliminate common subexpressions and improve performance:

```rust
use uncertain_rs::{Uncertain, computation::GraphOptimizer};

// Create an expression with common subexpressions
let x = Uncertain::normal(2.0, 0.1);
let y = Uncertain::normal(3.0, 0.1);
let z = Uncertain::normal(1.0, 0.1);

// Expression: (x + y) * (x + y) + (x + y) * z
// The subexpression (x + y) appears 3 times
let sum = x.clone() + y.clone();
let expr = (sum.clone() * sum.clone()) + (sum * z);

// Apply optimization to eliminate common subexpressions
let mut optimizer = GraphOptimizer::new();
let optimized = optimizer.eliminate_common_subexpressions(expr.into_computation_node());

// The optimized graph reuses the (x + y) subexpression
println!("Cache size: {}", optimizer.subexpression_cache.len());
```

## Development Workflow

We use [just](https://github.com/casey/just) as a task runner. Available commands:

- `just fmt` - Format code
- `just lint` - Run clippy linting
- `just test` - Run tests
- `just audit` - Security audit (check for vulnerabilities)
- `just dev` - Run the full development workflow (format + lint + test + audit)

### Security

This project takes security seriously. We run `cargo audit` to check for known vulnerabilities in dependencies:

- **CI**: Automated security audits run on every push and PR
- **Local**: Run `just audit` or `cargo audit` before submitting changes
- **Installation**: If you don't have `cargo-audit`, run `cargo install cargo-audit`

The security audit checks all dependencies against the [RustSec Advisory Database](https://rustsec.org/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
