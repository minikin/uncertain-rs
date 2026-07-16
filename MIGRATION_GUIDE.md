# Migration Guide: 0.2.x → 0.3.0

## New: correct, faster sampling (no migration needed for almost everyone)

Distribution sampling now uses `rand_distr` (new dependency) instead of hand-rolled
algorithms. This fixes a real correctness bug: the previous normal sampler clamped its
Box-Muller uniforms to `[0.001, 0.999]`, making `|z| > ~3.09` unreachable — every
downstream statistic derived from `normal`/`log_normal` samples was subtly biased away
from the true tails. It's also faster (measured: ~65% faster for normal sampling, 272µs
-> 94µs per 1000 samples).

Every distribution's degenerate-case behavior (`std_dev`/`scale`/`lambda` of `0`
degenerating to a point mass, as documented since 0.3.0's constructor validation) is
unchanged. The one new constraint: `poisson(lambda)` now rejects
`lambda > Poisson::MAX_LAMBDA` (~1.844e19) — a real limit of the sampling algorithm
rather than an arbitrary choice, and one no realistic use case should ever approach.

## New: reproducible sampling (no migration needed)

This is additive, not breaking — nothing to change in existing code. `sample()`/
`take_samples()` are unchanged (thread-local randomness, as before). New:
`sample_with`/`take_samples_with` take a `&mut ChaCha8Rng` (from the new `rand_chacha`
dependency) and produce bitwise-identical results for identically-seeded RNGs, across
runs and platforms. With the `parallel` feature, `take_samples_with_par(seed, count)`
gives the same guarantee independent of thread count. See the
[Reproducible Sampling](README.md#reproducible-sampling) section of the README.

## `Uncertain<T>` no longer implements `PartialEq`/`PartialOrd`

`a == b`, `a < b`, `a.partial_cmp(&b)` on two `Uncertain<T>` values no longer compile.

### Why

The removed impls drew one fresh sample from each side and compared those samples —
which isn't a meaningful fact about a *distribution*, and silently broke the contract
both traits promise: `a == a` could evaluate to `false`, and repeated comparisons of the
same pair could disagree with each other from one call to the next.

### How to update

Use the evidence-based comparison API instead — it returns `Uncertain<bool>` (evidence
built from independent samples) rather than a single-draw `bool`/`Ordering`, and pairs
with `probability_exceeds` to turn that evidence into a decision:

```rust
// Before (0.2.x) — compiled, but silently unsound
// if a < b { ... }

// After (0.3.0)
use uncertain_rs::Uncertain;
let a = Uncertain::normal(10.0, 1.0).unwrap();
let b = Uncertain::normal(12.0, 1.0).unwrap();
let evidence = a.lt_uncertain(&b);
if evidence.probability_exceeds(0.95) {
    // 95%+ confident a < b
}
```

For comparing against a fixed scalar threshold, use `a.gt(threshold)` / `a.lt(threshold)`
/ `a.eq_value(threshold)`, etc. (already available since 0.2.0, unaffected by this change).

If you genuinely want a single fresh sample from each side compared directly — rare, and
usually a sign you want the evidence-based API above instead — do it explicitly:
`a.sample() == b.sample()`, which makes the single-draw, non-reproducible nature of the
comparison visible at the call site instead of hiding behind `==`.

## Distribution constructors now return `Result`

Every distribution constructor validates its parameters and returns
`Result<Uncertain<T>, UncertainError>` instead of constructing an `Uncertain<T>`
infallibly. This affects:

`normal`, `uniform`, `exponential`, `log_normal`, `beta`, `gamma`, `bernoulli`,
`binomial`, `poisson`, `geometric`.

(`mixture`, `empirical`, `categorical`, and `point` were already `Result`-returning or
infallible-by-design in 0.2.x and are unchanged.)

### Why

Before 0.3.0, invalid parameters were silently accepted and produced garbage or `NaN`:

```rust
// 0.2.x — compiles, but silently broken
let dist = Uncertain::normal(0.0, -1.0);   // negative std_dev: undefined behavior
let dist = Uncertain::exponential(0.0);    // rate = 0: every sample is `inf`
```

0.3.0 catches these at construction time instead of at some unpredictable point deep in a
sampling loop.

### How to update

Add `.unwrap()` if you already know the parameters are valid (e.g. compile-time
constants), or handle the `Result` with `?` / a `match` if the parameters come from
user input, config, or any other untrusted source:

```rust
// Before (0.2.x)
let speed = Uncertain::normal(55.2, 5.0);

// After (0.3.0) — known-valid parameters
let speed = Uncertain::normal(55.2, 5.0).unwrap();

// After (0.3.0) — parameters from an untrusted source
fn build_sensor_model(mean: f64, std_dev: f64) -> Result<Uncertain<f64>, UncertainError> {
    let speed = Uncertain::normal(mean, std_dev)?;
    Ok(speed)
}
```

See [`examples/error_handling.rs`](examples/error_handling.rs) for a complete walkthrough
of validation, error matching, and propagation with `?`.

### Validation rules

| Constructor | Constraint | `0`/boundary behavior |
|---|---|---|
| `normal(mean, std_dev)` | both finite; `std_dev >= 0` | `std_dev = 0` → point mass at `mean` |
| `uniform(min, max)` | both finite; `min <= max` | `min == max` → point mass |
| `exponential(rate)` | finite; `rate > 0` | `rate = 0` → **rejected** (would yield `inf`) |
| `log_normal(mu, sigma)` | both finite; `sigma >= 0` | `sigma = 0` → point mass at `exp(mu)` |
| `beta(alpha, beta)` | both finite; `> 0` | `0` → **rejected** (undefined) |
| `gamma(shape, scale)` | both finite; `shape > 0`, `scale >= 0` | `scale = 0` → point mass at `0` |
| `bernoulli(p)` | finite; `p ∈ [0, 1]` | inclusive bounds allowed |
| `binomial(trials, p)` | `p` finite; `p ∈ [0, 1]` | inclusive bounds allowed |
| `poisson(lambda)` | finite; `lambda >= 0` | `lambda = 0` → point mass at `0` |
| `geometric(p)` | finite; `p ∈ (0, 1]` | `p = 0` → **rejected** (would never terminate) |

A rejected parameter returns `UncertainError::NonFiniteParameter { parameter, value }` (NaN
or infinite input) or `UncertainError::InvalidParameter { parameter, value, constraint }`
(finite but out of range) — never a panic.

## Statistics methods now return `Result`

Every statistics method that takes a `sample_count: usize` — `mode`, `histogram`,
`entropy`, `expected_value`, `expected_value_adaptive`, `variance`,
`standard_deviation`, `skewness`, `kurtosis`, `confidence_interval`, `cdf`, `quantile`,
`interquartile_range`, `median_absolute_deviation`, `pdf_kde`, `log_likelihood`,
`correlation`, `compute_stats_batch`, `lazy_stats`/`stats`, `LazyStats::new`, and
`AdaptiveLazyStats::new` — now returns `Result<_, UncertainError>` instead of an
infallible value. `quantile` additionally validates `q`, `confidence_interval` validates
`confidence`, and `pdf_kde`/`log_likelihood` validate `bandwidth`.

### Why

Before 0.3.0, `sample_count = 0` silently produced `NaN`, `0.0`, or an empty
collection depending on the method — a division by zero or an empty iterator that
happened to have a harmless-looking default, rather than a signal that something was
wrong. Likewise, `quantile(1.5, ...)` or `confidence_interval(-1.0, ...)` silently
clamped to a boundary value instead of rejecting an out-of-range request:

```rust
// 0.2.x — compiles, but silently wrong
let mean = Uncertain::normal(0.0, 1.0).unwrap().expected_value(0);       // NaN
let median = Uncertain::normal(0.0, 1.0).unwrap().quantile(1.5, 1000);   // silently clamped
```

### How to update

Add `.unwrap()` for compile-time-constant, known-valid arguments, or propagate with `?`
when `sample_count`/`q`/`confidence`/`bandwidth` come from user input:

```rust
// Before (0.2.x)
let mean = normal.expected_value(1000);

// After (0.3.0) — known-valid argument
let mean = normal.expected_value(1000).unwrap();

// After (0.3.0) — argument from an untrusted source
fn summarize(dist: &Uncertain<f64>, sample_count: usize) -> Result<f64, UncertainError> {
    let mean = dist.expected_value(sample_count)?;
    Ok(mean)
}
```

`LazyStats`'s own `mean`/`variance`/`std_dev`/`samples` accessors are unaffected — the
`sample_count` they use is validated once when the `LazyStats` is constructed
(`normal.lazy_stats(1000)?`), so the accessors themselves stay infallible. Its
`quantile`/`confidence_interval` methods (which take `q`/`confidence` directly) do
return `Result`, same as the `Uncertain` variants.

### Validation rules

| Parameter | Constraint | Error |
|---|---|---|
| `sample_count: usize` (all listed methods) | `> 0` | `InvalidSampleCount` |
| `q: f64` (`quantile`) | `∈ [0, 1]` | `InvalidQuantile` |
| `confidence: f64` (`confidence_interval`) | `∈ (0, 1)` | `InvalidConfidence` |
| `bandwidth: f64` (`pdf_kde`, `log_likelihood`) | `> 0` | `InvalidBandwidth` |

Valid inputs are unaffected: every `Ok(...)` value is bit-identical to the corresponding
0.2.x return value.

## `ComputationNode` evaluation methods now return `Result` instead of panicking

`ComputationNode::evaluate`, `evaluate_arithmetic`, and `evaluate_bool` (low-level graph
evaluation, not part of the typical `Uncertain<T>` usage surface) return
`Result<_, UncertainError>` instead of panicking on node shapes they can't handle.

### Why

Each of these three methods is total over some, but not all, `ComputationNode` variants —
which variants depends on `T`'s trait bounds at each method (see below). Before 0.3.0, the
unsupported cases were a `panic!`; a library consumer walking a graph and picking the wrong
method for a node shape could crash instead of getting a typed, recoverable error.

### How to update

If you call these methods directly (most code doesn't — `Uncertain<T>`'s combinator API
builds and evaluates graphs internally and is unaffected), handle the `Result`:

```rust
// Before (0.2.x) — panicked on a BinaryOp node
// let value = node.evaluate(&mut context);

// After (0.3.0)
let value = node.evaluate(&mut context)?;
```

- `evaluate` (`T: Shareable`) is total for `Leaf`/`UnaryOp`; `BinaryOp`/`Conditional` return
  `Err(UnsupportedNode)` — this is a structural limit (both require trait bounds `Shareable`
  alone doesn't provide), not a temporary gap.
- `evaluate_arithmetic` (`T: Arithmetic`) is now the total dispatcher for the arithmetic
  domain, including `Conditional` (it absorbs the former
  `evaluate_conditional_with_arithmetic`, which is removed — call `evaluate_arithmetic`
  instead).
- `evaluate_bool` is total except for `BinaryOp`, which has no defined `bool` operation.
- `evaluate_fresh` is unaffected: it keeps its infallible `-> T` signature.

`GraphProfiler::get_stats` is unaffected by this section — it already returned `Option` and
no longer has an internal (unreachable) `.expect()`.

## `GraphOptimizer::subexpression_cache` is no longer a public field

### Why

The field's type changed to a private, collision-safe key internally (see the CHANGELOG's
"Fixed" entry on `GraphOptimizer`'s subexpression cache), so it's no longer meaningful for
callers to reach into directly.

### How to update

If you only checked its size (as the README's optimizer example did), use the new
accessor:

```rust
// Before (0.2.x)
// let size = optimizer.subexpression_cache.len();

// After (0.3.0)
let size = optimizer.cache_size();
```

`GraphOptimizer::cse_hits()` is new — the number of times a cached subexpression was
reused instead of rebuilt.
