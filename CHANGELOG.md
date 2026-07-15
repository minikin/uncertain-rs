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
- `UncertainError::UnsupportedNode` - returned by `ComputationNode` evaluation methods
  when a graph node isn't evaluable in the requested domain (e.g. a boolean `BinaryOp`,
  for which no operation is defined).
- `ComputationNode::constant` - creates a leaf whose value is structurally known to be
  constant, for callers building computation graphs directly with the low-level API.
  This is the only way to make a node eligible for the optimizer's identity
  elimination/constant folding passes; `Uncertain::point` now uses it internally.

### Fixed

- `GraphOptimizer`'s constant-folding and identity-elimination passes (`x + 0`, `x * 1`,
  `x * 0`, etc.) no longer decide whether a node is constant by sampling it 3-4 times and
  comparing the results. That approach could silently miscompile a low-entropy
  distribution into a constant purely by chance (e.g. `bernoulli(0.99)` had roughly a
  96% chance of sampling the same value 4 times in a row) and fold it away, changing the
  distribution the optimized graph represents. Constancy is now decided structurally: a
  node is constant only if it was built via the new `ComputationNode::constant`
  (transitively, `Uncertain::point`) or is itself a folded result of already-constant
  operands — never by observing sampled behavior.

### Changed

- **BREAKING**: Error return types changed from `Result<T, &'static str>` to `Result<T, UncertainError>`
- **BREAKING**: Distribution constructors (`normal`, `uniform`, `exponential`, `log_normal`,
  `beta`, `gamma`, `bernoulli`, `binomial`, `poisson`, `geometric`) now validate their
  parameters and return `Result<Uncertain<T>, UncertainError>` instead of constructing an
  `Uncertain<T>` infallibly. See `MIGRATION_GUIDE.md`.
- **BREAKING**: Statistics entry points in `src/statistics.rs` now validate their
  parameters and return `Result` instead of silently producing `NaN`/a clamped value:
  every method taking `sample_count: usize` (`mode`, `histogram`, `entropy`,
  `expected_value`, `expected_value_adaptive`, `variance`, `standard_deviation`,
  `skewness`, `kurtosis`, `confidence_interval`, `cdf`, `quantile`,
  `interquartile_range`, `median_absolute_deviation`, `pdf_kde`, `log_likelihood`,
  `correlation`, `compute_stats_batch`, `lazy_stats`/`stats`, `LazyStats::new`,
  `AdaptiveLazyStats::new`) rejects `sample_count == 0` (`InvalidSampleCount`);
  `quantile` additionally rejects `q` outside `[0, 1]` (`InvalidQuantile`);
  `confidence_interval` rejects `confidence` outside `(0, 1)` (`InvalidConfidence`);
  `pdf_kde`/`log_likelihood` reject non-positive `bandwidth` (`InvalidBandwidth`). See
  `MIGRATION_GUIDE.md`.
- Error messages now include structured data (e.g., expected vs actual counts)
- Improved error messages with more context and helpful information
- Distribution sampling (`normal`, `uniform`, `exponential`, `log_normal`, `beta`, `gamma`,
  `bernoulli`, `binomial`, `poisson`, `geometric`) now uses `rand_distr` (new dependency)
  instead of hand-rolled algorithms. The previous normal sampler clamped its Box-Muller
  uniforms to `[0.001, 0.999]`, making `|z| > ~3.09` unreachable and biasing every
  downstream statistic; `rand_distr`'s Ziggurat method has no such clamp and is
  ~65% faster (measured: 272µs -> 94µs per 1000 samples). `poisson` now rejects
  `lambda > ~1.844e19` (`Poisson::MAX_LAMBDA`, a real limit of the sampling algorithm) —
  see `MIGRATION_GUIDE.md`. No other observable behavior change; degenerate cases
  (`std_dev`/`scale`/`lambda` of `0`) are still supported exactly as before.
- **BREAKING**: `ComputationNode::evaluate`, `evaluate_arithmetic`, and `evaluate_bool` now
  return `Result<_, UncertainError>` instead of panicking on unsupported node/domain
  combinations (e.g. a `BinaryOp` passed to `evaluate`, or a boolean `BinaryOp` passed to
  `evaluate_bool`). `evaluate_arithmetic` additionally now handles `Conditional` nodes
  directly (it absorbs the former `evaluate_conditional_with_arithmetic`, which is removed).
  `evaluate_fresh` keeps its infallible `-> T` signature. See `MIGRATION_GUIDE.md`.

### Removed

- **BREAKING**: `Uncertain<T>` no longer implements `PartialEq`/`PartialOrd`. The removed
  impls drew one fresh sample per side and compared those samples, which isn't a
  meaningful fact about a distribution and silently broke both traits' contracts (`a == a`
  could be `false`). Use the evidence-based `Comparison` trait (`a.gt(threshold)`,
  `a.lt_uncertain(&b)`, etc.) instead. See `MIGRATION_GUIDE.md`.
- **BREAKING**: `ComputationNode::evaluate_conditional_with_arithmetic` is removed; its
  logic is now part of `evaluate_arithmetic` (see above).

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
