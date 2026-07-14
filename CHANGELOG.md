# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Custom `UncertainError` type for improved error handling
- `Result` type alias for convenience (`std::result::Result<T, UncertainError>`)
- Structured error types with detailed information:
  - `EmptyComponents` - When creating mixture with no components
  - `WeightCountMismatch` - When weights count doesn't match components count
  - `EmptyData` - When creating empirical distribution with no data
  - `EmptyProbabilities` - When creating categorical distribution with no probabilities
  - `InvalidParameter` - When a parameter violates constraints
  - `NonFiniteParameter` - When a parameter is NaN or infinite
  - `InvalidWeights` - When weights are invalid
  - `InvalidSampleCount` - When sample count is invalid
  - `InvalidQuantile` - When quantile is out of [0, 1] range
  - `InvalidConfidence` - When confidence level is out of (0, 1) range
  - `InvalidBandwidth` - When bandwidth is non-positive
- Helper methods for creating errors programmatically
- New example: `error_handling.rs` demonstrating proper error handling patterns
- Migration guide (`MIGRATION_GUIDE.md`) for upgrading from 0.2.x
- Reproducible sampling: `Uncertain::sample_with`/`take_samples_with` take a seeded
  `ChaCha8Rng` (new `rand_chacha` dependency) and produce bitwise-identical results
  across runs and platforms for the same seed; `take_samples_with_par` (parallel
  feature) gives the same guarantee independent of thread count. `sample()`/
  `take_samples()` are unchanged (still thread-local randomness).

### Changed

- **BREAKING**: Error return types changed from `Result<T, &'static str>` to `Result<T, UncertainError>`
- **BREAKING**: Distribution constructors (`normal`, `uniform`, `exponential`, `log_normal`,
  `beta`, `gamma`, `bernoulli`, `binomial`, `poisson`, `geometric`) now validate their
  parameters and return `Result<Uncertain<T>, UncertainError>` instead of constructing an
  `Uncertain<T>` infallibly. See `MIGRATION_GUIDE.md`.
- Error messages now include structured data (e.g., expected vs actual counts)
- Improved error messages with more context and helpful information

### Removed

- **BREAKING**: `Uncertain<T>` no longer implements `PartialEq`/`PartialOrd`. The removed
  impls drew one fresh sample per side and compared those samples, which isn't a
  meaningful fact about a distribution and silently broke both traits' contracts (`a == a`
  could be `false`). Use the evidence-based `Comparison` trait (`a.gt(threshold)`,
  `a.lt_uncertain(&b)`, etc.) instead. See `MIGRATION_GUIDE.md`.

## [0.2.0] - 2024-09-24

### Added

- Computation graph optimizations
- Common subexpression elimination
- Identity operation elimination
- Constant folding
- Graph visualization and profiling tools
- Adaptive sampling strategies
- Lazy statistics evaluation
- Comprehensive examples (GPS navigation, medical diagnosis, climate modeling, sensor processing)

### Changed

- Improved performance through graph optimizations
- Better caching strategies

## [0.1.0] - Initial Release

### Added

- Core `Uncertain<T>` type
- Evidence-based conditionals
- Probability distributions (normal, uniform, exponential, beta, gamma, binomial, etc.)
- Statistical analysis methods
- SPRT hypothesis testing
- Arithmetic operations with uncertainty propagation
- Comprehensive test suite
- Documentation and examples
