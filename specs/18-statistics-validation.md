# Spec 18 — Statistics Entry-Point Validation

**Status:** Implemented | **Effort:** High | **Module:** `src/statistics.rs`

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

## Implementation notes (deviations and interpretation calls)

- **Validation helpers.** Four private free functions in `src/statistics.rs` —
  `validate_sample_count`, `validate_quantile`, `validate_confidence`,
  `validate_bandwidth` — each called at the top of the relevant method, before any
  sampling/cache work. This keeps the actual computation and its cache interaction
  (`cache::stats_cache().get_or_compute_*`, whose closures are unchanged) exactly as
  before for valid inputs, satisfying invariant 8: invalid inputs short-circuit before
  reaching the closure at all, so the `Ok(...)` path is bit-identical to the pre-spec
  return value.
- **Construction-time vs. call-time validation (invariant 5).** `LazyStats::new` and
  `AdaptiveLazyStats::new` validate once, at construction:
  - `LazyStats::new(uncertain, sample_count)` now returns `Result<Self, UncertainError>`,
    validating `sample_count > 0` once. Its own `mean`/`variance`/`std_dev`/`samples`
    accessors stay infallible (`-> f64`/`-> Vec<T>`) — they read the already-validated
    stored `sample_count`, so a per-call re-check would be redundant. `quantile`/
    `confidence_interval` still validate per call (invariants 1–2 apply to *both*
    `LazyStats` and `Uncertain` variants), since `q`/`confidence` are call-time
    parameters, not part of the stored state.
  - `AdaptiveLazyStats::new(uncertain, config)` validates `config.min_samples > 0` once.
    This isn't a literal `sample_count: usize` parameter (it's a field on the
    `AdaptiveSampling` config struct), but `min_samples` plays exactly that role for the
    adaptive convergence loop — with `min_samples == 0`, the loop's exponential growth
    (`sample_count * growth_factor`) never advances past zero and convergence never
    terminates. `Uncertain::adaptive_lazy_stats`/`expected_value_adaptive` propagate this
    via `?` (the latter needs no explicit check of its own: its first loop iteration
    calls `self.expected_value(config.min_samples)`, which already validates and errors).
  - Every other listed method (`mode`, `histogram`, `entropy`, `expected_value`,
    `variance`, `standard_deviation`, `skewness`, `kurtosis`, `confidence_interval`,
    `cdf`, `quantile`, `interquartile_range`, `median_absolute_deviation`, `pdf_kde`,
    `log_likelihood`, `correlation`, `compute_stats_batch`, `lazy_stats`/`stats`) takes
    `sample_count` as a genuine per-call parameter (no stored state to validate once
    instead), so each validates at call time. Several (`standard_deviation`,
    `interquartile_range`, `median_absolute_deviation`, `log_likelihood`, `entropy`)
    validate only by propagating `?` from a method they delegate to
    (`variance`/`quantile`/`quantile`/`pdf_kde`/`histogram` respectively), per invariant
    4's parenthetical allowance.
- **Dead-code cleanup enabled by validation.** Once `sample_count > 0` is guaranteed at
  entry, `Uncertain::take_samples(sample_count)` (an infinite iterator `.take(count)`)
  can never return an empty `Vec`. This made the pre-existing `samples.is_empty()` guard
  in both `LazyStats::quantile` and `Uncertain::quantile` truly unreachable (previously
  it was reachable only via the since-removed silent-clamp path for invalid inputs), so
  both were deleted rather than left as untestable dead branches — the same pattern as
  the `GraphProfiler::get_stats` cleanup in Spec 06.
- **Tests whose premise the validation invalidates.** Several existing tests
  (`test_mode_empty_samples`, `test_histogram_empty`, and six `..._clamp` tests for
  `quantile`/`confidence_interval` on both `LazyStats` and `Uncertain`) were written
  *for* the old silent/clamping behavior on invalid inputs (`sample_count = 0`,
  `q`/`confidence` outside their valid range) — several carried an explicit comment
  flagging them as "not yet validated -- see spec 18". These are replaced with tests
  asserting the specific `Err` variant instead of the old clamped/empty value.
- **CRAP regression, accepted and rebaselined.** Every touched method's CRAP score rose
  (CC +1 to +2, from the added validation branch/branches), the same pattern as Spec 06.
  All reached 100% line coverage in this change (including two new tests for the
  previously-untested `validate_bandwidth`/`validate_quantile` error paths and two for
  the single-sample/exact-integer-position branches the dead-code cleanup above exposed
  as needing direct coverage), so the increase is purely complexity, not a coverage gap.
  `crap_baseline.json` regenerated via `just crap-update-baseline`.
- **Benchmarks.** Not a stated invariant for this spec (unlike Spec 05/06), but checked
  anyway: `statistical_operations`/`interval_operations`/`pdf_operations` criterion
  groups show no consistent regression — a couple of the tiniest cached-path
  benchmarks (~30-40ns baseline) flip between "regressed" and "improved" by double-digit
  percentages across consecutive reruns, i.e. pure measurement noise on a duration where
  a couple of nanoseconds of branch overhead is a large relative percentage; the
  "first_run" (uncached, real sampling) benchmarks — the ones that reflect actual
  workload cost — show no regression.
