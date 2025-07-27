# uncertain-rs

[![CI](https://github.com/minikin/uncertain-rs/workflows/CI/badge.svg)](https://github.com/minikin/uncertain-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library for uncertainty-aware programming, implementing the approach from
"Uncertain<T>: A First-Order Type for Uncertain Data" by Bornholt, Mytkowicz, and McKinley.

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
    let sum = x + y;
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
