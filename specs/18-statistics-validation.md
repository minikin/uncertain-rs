# Spec 18 — Statistics Entry-Point Validation

**Status:** Pending | **Effort:** High | **Module:** `src/statistics.rs`

## Context

Split out from [Spec 02](02-validated-constructors.md) during implementation: auditing the
ripple showed 21 methods across `src/statistics.rs` take a `sample_count: usize` parameter
that silently produces `NaN`/nonsense at `0` (e.g. dividing by `sample_count as f64`), and
`quantile`, `confidence_interval`, `pdf_kde`, `log_likelihood` accept unconstrained
`q`/`confidence`/`bandwidth` parameters — `pdf_kde(x, n, 0.0)` and a quantile request of
`1.5` both produce garbage rather than the `InvalidBandwidth`/`InvalidQuantile` errors that
already exist in `src/error.rs` for exactly this purpose. This is a comparably large,
independently-shippable change from Spec 02: 42+ call sites across tests, examples, and
benches reference just these four methods, before counting every other `sample_count`
consumer.

Two nearly-identical method pairs exist and both need this: `LazyStats::quantile` /
`Uncertain::quantile`, and `LazyStats::confidence_interval` / `Uncertain::confidence_interval`.

## Scope and Invariants

1. `quantile` (both `LazyStats` and `Uncertain`) validates `q ∈ [0, 1]`
   (`InvalidQuantile` otherwise) and returns `Result<f64>`.
2. `confidence_interval` (both variants) validates `confidence ∈ (0, 1)`
   (`InvalidConfidence` otherwise) and returns `Result<(f64, f64)>`.
3. `pdf_kde` and `log_likelihood` validate `bandwidth > 0` (`InvalidBandwidth` otherwise)
   and return `Result<f64>`.
4. Every method taking `sample_count: usize` validates `sample_count > 0`
   (`InvalidSampleCount` otherwise, using the existing `reason: &'static str` field to say
   why — e.g. "cannot compute a mean from zero samples") and returns `Result<...>` instead
   of silently producing `NaN`/`0.0`. This includes at minimum: `mean`, `variance`,
   `std_dev`, `expected_value`, `expected_value_adaptive`, `skewness`, `kurtosis`, `cdf`,
   `interquartile_range`, `median_absolute_deviation`, `correlation`, `mode`, `histogram`,
   `entropy`, `compute_stats_batch`, `lazy_stats`/`stats` (audit the full 21-method list at
   implementation time; some may already delegate to a method this spec fixes, needing
   only propagation, not its own check).
5. `LazyStats`/`AdaptiveLazyStats`/`ProgressiveStats` constructors (`new`) validate
   `sample_count > 0` once at construction so the accessor methods that read from them
   don't need to re-check on every call, where the internal structure allows it; document
   which approach (construction-time vs. call-time validation) applies to which type and
   why.
6. Error messages state the parameter name, the received value, and the constraint,
   consistent with Spec 02's `InvalidParameter`/`NonFiniteParameter` style.
7. `MIGRATION_GUIDE.md` gains a section for this change; it may be the same 0.2→0.3
   migration document Spec 02 created, appended to rather than duplicated.
8. Nothing in this spec changes the numeric result for any already-valid input — only
   previously-undefined/NaN-producing inputs start returning `Err` instead.

## Acceptance Tests

- **Given** `pdf_kde(x, n, 0.0)`, **then** `Err(InvalidBandwidth)`; **given** a quantile
  request of `1.5`, **then** `Err(InvalidQuantile)`; **given** `confidence_interval(0.0,
  n)` or `confidence_interval(1.0, n)`, **then** `Err(InvalidConfidence)`.
- **Given** any `sample_count`-taking method called with `sample_count = 0`, **then**
  `Err(InvalidSampleCount)`, never `NaN`.
- **Given** valid inputs matching pre-spec behavior, **when** any changed method is called,
  **then** the `Ok(...)` value is bit-identical to the pre-spec return value (no silent
  numeric change).
- **Given** the crate source, **when** grepped, **then** `InvalidQuantile`,
  `InvalidConfidence`, `InvalidBandwidth`, `InvalidSampleCount` each have a constructing
  site outside `error.rs` and its tests.
- **Given** `just dev`, **when** run, **then** it's green across default and `parallel`
  features, including every test/example/bench call site updated for the new `Result`
  signatures.
