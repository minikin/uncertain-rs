# uncertain-rs

[![CI](https://github.com/minikin/uncertain-rs/workflows/CI/badge.svg)](https://github.com/minikin/uncertain-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/minikin/uncertain-rs/graph/badge.svg?token=IKjgVbFAQk)](https://codecov.io/gh/minikin/uncertain-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library for uncertainty-aware programming, implementing the approach from
["Uncertain<T>: A First-Order Type for Uncertain Data" by Bornholt, Mytkowicz, and McKinley.](https://www.microsoft.com/en-us/research/publication/uncertaint-a-first-order-type-for-uncertain-data-2/)

- [uncertain-rs](#uncertain-rs)
  - [Core Concept: Evidence-Based Conditionals](#core-concept-evidence-based-conditionals)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Error Handling](#error-handling)
  - [Reproducible Sampling](#reproducible-sampling)
  - [Advanced Features](#advanced-features)
    - [Parallel Sampling](#parallel-sampling)
    - [Graph Optimization](#graph-optimization)
  - [Development Workflow](#development-workflow)
  - [Contributing](#contributing)
  - [License](#license)

## Core Concept: Evidence-Based Conditionals

Instead of treating uncertain data as exact values (which leads to bugs),
this library uses evidence-based conditionals that account for uncertainty:

```rust
use uncertain_rs::Uncertain;

// Create uncertain values from probability distributions
let speed = Uncertain::normal(55.2, 5.0).unwrap(); // GPS reading with ±5 mph error

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
- **Reproducible sampling**: Seed a `ChaCha8Rng` for bitwise-identical results across runs
- **Parallel sampling** (optional): Multi-threaded sample generation using rayon

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
uncertain-rs = "0.2.0"

# Enable parallel sampling (optional)
# uncertain-rs = { version = "0.2.0", features = ["parallel"] }
```

## Quick Start

```rust
use uncertain_rs::Uncertain;

fn main() {
    // Create uncertain values
    let x = Uncertain::normal(5.0, 1.0).unwrap();
    let y = Uncertain::normal(3.0, 0.5).unwrap();

    // Perform arithmetic operations
    let sum = x.clone() + y.clone();
    let product = x * y;

    // Sample from the distributions
    println!("Sum sample: {}", sum.sample());
    println!("Product sample: {}", product.sample());

    // Statistical analysis
    println!("Sum mean: {}", sum.expected_value(1000).unwrap());
    println!("Sum std dev: {}", sum.standard_deviation(1000).unwrap());
}
```

For more examples, see the [examples directory](examples).

## Error Handling

The library uses a custom `UncertainError` type for type-safe error handling. Every
distribution constructor validates its parameters and returns `Result<Uncertain<T>,
UncertainError>`:

```rust
use uncertain_rs::{Uncertain, UncertainError};

let result = Uncertain::normal(0.0, -1.0); // negative std_dev
match result {
    Err(UncertainError::InvalidParameter { parameter, value, constraint }) => {
        println!("Invalid '{parameter}': {value} {constraint}");
    }
    Err(e) => println!("Error: {}", e),
    Ok(dist) => { /* use dist */ }
}
```

See [`examples/error_handling.rs`](examples/error_handling.rs) for a complete walkthrough,
and [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) if you're upgrading from a pre-0.3 release.

## Reproducible Sampling

`sample()`/`take_samples()` draw from real thread-local randomness — different every run,
by design. For reproducible results (debugging, tests, papers), use `sample_with`/
`take_samples_with` with a seeded `ChaCha8Rng`:

```rust
use uncertain_rs::Uncertain;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

let normal = Uncertain::normal(0.0, 1.0).unwrap();

let mut rng = ChaCha8Rng::seed_from_u64(42);
let samples = normal.take_samples_with(&mut rng, 1000);

// Same seed, same crate version -> bitwise-identical stream, on any platform.
let mut rng2 = ChaCha8Rng::seed_from_u64(42);
assert_eq!(samples, normal.take_samples_with(&mut rng2, 1000));
```

This works for composed expressions too (arithmetic, comparisons, evidence) — the whole
expression tree draws from the one seeded RNG for the duration of the call. With the
`parallel` feature, `take_samples_with_par(seed, count)` gives the same reproducibility
independent of thread count (each sample index always derives the same sub-seed,
regardless of which thread happens to compute it).

## Advanced Features

### Parallel Sampling

Enable the `parallel` feature to unlock multi-threaded sample generation for significant performance improvements with large sample counts:

```rust
use uncertain_rs::Uncertain;

let normal = Uncertain::normal(0.0, 1.0).unwrap();

// Sequential sampling
let samples = normal.take_samples(100_000);

// Parallel sampling (requires 'parallel' feature)
let samples_par = normal.take_samples_par(100_000); // ~2-4x faster on multi-core systems

// Parallel + caching for f64 distributions
let gamma = Uncertain::gamma(2.0, 1.0).unwrap();
let cached = gamma.take_samples_cached_par(100_000); // Fast generation + reuse
```

**When to use parallel sampling:**

- Large sample counts (typically > 1,000)
- Expensive sampling operations (complex transformations, costly distributions)
- Multi-core systems available for parallelization
- Monte Carlo simulations and statistical analysis

See the [parallel sampling example](examples/parallel_sampling.rs) for benchmarks and use cases.

### Graph Optimization

The library includes a computation graph optimizer that can eliminate common subexpressions and improve performance:

```rust
use uncertain_rs::{
    computation::{ComputationNode, GraphOptimizer},
    operations::arithmetic::BinaryOperation,
};

// GraphOptimizer works on ComputationNode directly — it isn't wired into `Uncertain<T>`'s
// arithmetic operators, so a graph built through `+`/`*` on `Uncertain<T>` has to be
// reconstructed at the ComputationNode level to optimize it (see `examples/optimizations/`
// for the full pattern building this up from `Uncertain::normal`).
let x = ComputationNode::leaf(|| 2.0);
let y = ComputationNode::leaf(|| 3.0);
let z = ComputationNode::leaf(|| 1.0);

// Expression: (x + y) * (x + y) + (x + y) * z
// The subexpression (x + y) appears 3 times
let sum = ComputationNode::binary_op(x, y, BinaryOperation::Add);
let expr = ComputationNode::binary_op(
    ComputationNode::binary_op(sum.clone(), sum.clone(), BinaryOperation::Mul),
    ComputationNode::binary_op(sum.clone(), z, BinaryOperation::Mul),
    BinaryOperation::Add,
);

// Apply optimization to eliminate common subexpressions
let mut optimizer = GraphOptimizer::new();
let optimized = optimizer.eliminate_common_subexpressions(expr);

// The optimized graph reuses the (x + y) subexpression
println!("Cache size: {}", optimizer.cache_size());
println!("Result: {}", optimized.evaluate_fresh());
```

## Development Workflow

We use [just](https://github.com/casey/just) as a task runner. Available commands:

- `just fmt` - Format code
- `just lint` - Run clippy linting
- `just test` - Run tests
- `just audit` - Security audit (check for vulnerabilities)
- `just dev` - Run the full development workflow (format + lint + test + audit)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
