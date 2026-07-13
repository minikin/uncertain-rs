# Spec 11 — Property-Based Testing

**Status:** Pending | **Effort:** Medium | **Module:** `tests/`, dev-dependencies

## Context

For a sampling/statistics library, invariants are natural properties, yet there is no
`proptest`/`quickcheck` anywhere — only example-based tests with hand-picked values.
Property tests would have caught several review findings mechanically (NaN paths,
estimator inconsistencies, optimizer miscompiles).

## Scope and Invariants

1. `proptest` is added as a dev-dependency; property tests live in `tests/properties/`
   (or a `props_` prefixed integration test per area).
2. Properties use seeded sampling (Spec 04) so failures reproduce; `proptest` regressions
   files are committed.
3. Minimum property set:
   - **Distributions:** samples lie in the support (beta ∈ [0,1], exponential ≥ 0,
     bernoulli ∈ {t,f}, poisson/binomial/geometric are valid integers); constructors
     reject invalid params for all invalid inputs (pairs with Spec 02).
   - **Statistics:** `cdf` is monotone nondecreasing with range ⊆ [0,1]; quantiles are
     monotone in the quantile argument and `quantile(cdf(x)) ≈ x` on continuous support;
     `variance ≥ 0`; `std_dev == sqrt(variance)`; CI lower ≤ mean ≤ CI upper; IQR ≥ 0;
     mode ∈ samples.
   - **Arithmetic:** for independent X, Y with fixed seeds — `mean(X+Y) ≈ mean(X)+mean(Y)`,
     `mean(cX) ≈ c·mean(X)`; `point(a) op point(b)` equals the scalar result exactly.
   - **Logical/comparison:** `P(A and B) ≤ min(P(A), P(B))`; `P(not A) ≈ 1 − P(A)`;
     `probability ∈ [0, 1]` always.
   - **Optimizer:** optimized vs unoptimized graph, same seed ⇒ identical samples
     (pairs with Specs 07/08).
   - **Hypothesis:** posterior from `bayesian_update` sums to 1 and contains no NaN for
     arbitrary valid priors/evidence.
4. Property tests run in CI as part of the standard test job; case counts tuned so the
   suite stays under a reasonable budget (e.g. 256 cases default, more via env for
   nightly/scheduled runs).

## Acceptance Tests

- **Given** `cargo test`, **when** run, **then** the property suites execute and pass on
  stable with both default and `parallel` features.
- **Given** a deliberately introduced violation (e.g. re-adding the Box–Muller clamp or
  the ÷n variance in one spot), **when** the property suite runs, **then** at least one
  property fails — demonstrated once in the PR description, then reverted.
- **Given** a property failure, **when** re-run, **then** the same failing case replays
  from the committed regression file.
- **Given** CI, **then** property tests are part of the required test job, not an optional
  workflow.
