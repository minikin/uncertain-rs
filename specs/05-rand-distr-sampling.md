# Spec 05 — Correct Sampling via `rand_distr`

**Status:** Pending | **Effort:** Medium | **Module:** `src/distributions.rs`

## Context

Samplers are hand-rolled. The normal sampler clamps its Box–Muller uniforms to
`[0.001, 0.999]` (`src/distributions.rs:208`), which truncates the tails (|z| ≳ 3.09 is
unreachable) and biases every downstream statistic; it also discards the second Box–Muller
output. Other distributions (gamma, beta, poisson, binomial) use naive algorithms whose
quality/performance is unverified. `rand_distr` provides vetted, faster implementations
(Ziggurat normal, Marsaglia–Tsang gamma, etc.).

## Scope and Invariants

1. All continuous/discrete samplers are backed by `rand_distr` distributions; hand-rolled
   sampling code is deleted. The tail clamp disappears.
2. `rand_distr`'s parameter errors map onto Spec 02's validation errors at construction
   time (validation stays in this crate's error vocabulary).
3. Composes with Spec 04: `rand_distr` distributions sample from the injected RNG.
4. Sampled quality is verified statistically, not eyeballed: with a fixed seed and large n,
   moment checks (mean/variance/skewness) against closed-form values within analytic
   tolerance, plus a tail check for the normal.
5. Public constructor signatures do not change beyond what Spec 02 already changed.
6. Benchmarks in `benches/performance_benchmarks.rs` still cover sampling; a before/after
   criterion comparison is recorded in the PR description (expect normal sampling to get
   faster).

## Acceptance Tests

- **Given** `normal(0, 1)` with a fixed seed, **when** 10⁷ samples are drawn, **then** at
  least one sample has `|z| > 4` (tails restored) and the empirical mean/std are within
  analytic tolerance of 0/1.
- **Given** `gamma(k, θ)` and `beta(α, β)` for several parameter sets (including k < 1),
  **when** sampled at large n with a fixed seed, **then** empirical mean and variance match
  closed forms within tolerance.
- **Given** `poisson(λ)` and `binomial(n, p)`, **then** all samples are integers in the
  support and empirical means match `λ` / `np` within tolerance.
- **Given** the crate source, **when** grepped, **then** no Box–Muller implementation and
  no uniform clamp remain in `src/distributions.rs`.
- **Given** `Cargo.toml`, **then** `rand_distr` is a regular dependency compatible with the
  resolved `rand` version (post Spec 01 reconciliation).
