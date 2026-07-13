# Spec 09 — Consistent Variance/Estimator Policy

**Status:** Pending | **Effort:** Low | **Module:** `src/statistics.rs`

## Context

The crate mixes estimators: `Uncertain::variance` and `LazyStats::variance` compute
population variance (÷ n) (`src/statistics.rs:710`, `:94`) while
`ProgressiveStats::variance` computes sample variance (÷ n−1) using the numerically
unstable `sum_squares − n·mean²` formula (`:212`) — despite "numerically stable" claims
elsewhere in the docs. Users comparing the two get silently different numbers, and the
catastrophic-cancellation formula loses precision for large means.

## Scope and Invariants

1. One policy, crate-wide: **sample variance (÷ n−1) is the default** everywhere a
   variance/std-dev is estimated from samples (it is the unbiased estimator users expect
   from a Monte-Carlo library). A `population_variance()` sibling is provided where the
   biased estimator is legitimately wanted.
2. All variance computations use a numerically stable algorithm (Welford / one-pass
   updates for the progressive path, two-pass or Welford for the batch path). The
   `sum_squares − n·mean²` form is removed.
3. Skewness/kurtosis/std-dev definitions are audited for the same n vs n−1 consistency
   and documented (state exactly which estimator each returns).
4. Docs on every estimator method state the formula and the estimator type.
5. n = 0 and n = 1 edge cases return a documented, non-NaN result (error per Spec 02's
   `InvalidSampleCount`, or a documented `0.0`/`None` — one choice, applied uniformly).

## Acceptance Tests

- **Given** the same fixed-seed sample set, **when** `variance`, `LazyStats::variance`,
  and `ProgressiveStats::variance` are computed, **then** all three agree to floating-point
  tolerance.
- **Given** samples `[1.0, 2.0, 3.0]`, **then** `variance == 1.0` (n−1) and
  `population_variance == 2/3`.
- **Given** samples with a huge common offset (e.g. `1e9 + small noise`), **when**
  progressive variance is computed, **then** the result matches the two-pass computation
  to relative tolerance (no catastrophic cancellation).
- **Given** 0 or 1 samples, **when** variance is requested, **then** the documented
  behavior occurs and it is not `NaN`.
- **Given** the source, **when** grepped, **then** no `sum_squares - n * mean * mean`
  pattern remains.
