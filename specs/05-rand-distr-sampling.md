# Spec 05 — Correct Sampling via `rand_distr`

**Status:** Implemented | **Effort:** Medium | **Module:** `src/distributions.rs`

## Context

Samplers were hand-rolled. The normal sampler clamped its Box–Muller uniforms to
`[0.001, 0.999]`, which truncated the tails (`|z| ≳ 3.09` was unreachable) and biased
every downstream statistic; it also discarded the second Box–Muller output. Other
distributions (gamma, beta, poisson, binomial) used naive algorithms whose
quality/performance was unverified. `rand_distr` provides vetted, faster implementations
(Ziggurat normal, Marsaglia–Tsang gamma, etc.).

## Scope and Invariants

1. All continuous/discrete samplers are backed by `rand_distr` distributions
   (`Normal`, `LogNormal`, `Uniform`, `Exp`, `Beta`, `Gamma`, `Bernoulli`, `Binomial`,
   `Poisson`, `Geometric`); hand-rolled sampling code is deleted. The tail clamp is gone.
   (`mixture`/`empirical`/`categorical` are unaffected — they're custom
   weighted-selection logic, not standard distributions `rand_distr` models.)
2. **Deviation, discovered at implementation time:** three of `rand_distr`'s constructors
   are *stricter* than the degenerate-case behavior this crate committed to in Spec 02,
   requiring special-casing rather than a direct pass-through:
   - `Gamma::new` requires `scale > 0` strictly (rejects `0`). `scale == 0` is handled
     before ever calling `rand_distr`: returns a point mass at `0` directly.
   - `Poisson::new` requires `lambda > 0` strictly (rejects `0`), same treatment:
     `lambda == 0` returns a point mass at `0` directly.
   - `Poisson::new` also rejects `lambda > Poisson::MAX_LAMBDA` (~1.844e19) — a real
     numerical limit of the sampling algorithm with no equivalent in the old hand-rolled
     version (which had no upper bound at all, though nothing realistic ever approached
     one). Mapped onto this crate's own `InvalidParameter` error rather than leaking
     `rand_distr`'s error type, per item 2 of the original scope.
   - Everywhere else (`Normal`, `Uniform` via `new_inclusive`, `Exp`, `LogNormal`,
     `Beta`, `Bernoulli`, `Binomial`, `Geometric`), this crate's existing Spec 02
     validation is a strict subset of what `rand_distr` itself accepts, so construction
     after validation always succeeds — confirmed by reading each constructor's source,
     not assumed.
3. **Convention preserved across the algorithm swap:** `rand_distr::Geometric` counts
   *failures before the first success* (0-indexed, `k >= 0`); this crate documents
   *trials until first success* (1-indexed, `>= 1`), matching its pre-`rand_distr`
   behavior. The sampling closure adds `1` to `rand_distr`'s result to preserve that
   convention rather than silently changing it.
4. `binomial`/`poisson`/`geometric` stay generic over `T: From<u32> + ...` (unchanged from
   Spec 02) despite `rand_distr::Binomial`/`Geometric` returning `u64` and `Poisson<f64>`
   returning `f64`: binomial's count is bounded by its own `u32` trial count (never
   truncates); poisson rounds its `f64` result before narrowing; geometric's `u64`
   failure count is narrowed the same way. All three accept the same practical
   precision this crate already had.
5. Composes with Spec 04: every `rand_distr::Distribution::sample` call goes through
   `crate::rng::with_rng`, so it transparently participates in `sample_with`/
   `take_samples_with`'s seeded override with no special-casing needed.
6. Sampled quality is verified statistically: existing moment tests (mean/variance
   against closed forms) continue to pass against the new implementation unmodified,
   plus a new dedicated tail-restoration test using a fixed seed (`take_samples_with`,
   10,000,000 samples) asserting at least one `|z| > 4` — impossible under the old clamp,
   routine under `rand_distr`'s Ziggurat method.
7. Public constructor signatures unchanged beyond Spec 02 (only the one new
   `Poisson::MAX_LAMBDA` upper-bound check, item 2 above).
8. Benchmark comparison (measured, not estimated): `expected_value_first_run`
   (1000 fresh `normal(0,1)` samples) went from **272.4µs to 94.1µs, a 65.5% improvement**
   — `cargo bench --bench performance_benchmarks -- expected_value --sample-size 20`,
   before and after this spec's changes, same machine, same run.

## Acceptance Tests

- **Given** `normal(0, 1)` with a fixed seed, **when** 10,000,000 samples are drawn via
  `take_samples_with`, **then** at least one sample has `|z| > 4` (tails restored). ✅
  (`test_normal_tails_are_not_truncated`)
- **Given** `gamma(k, θ)` and `beta(α, β)` for several parameter sets (including `k < 1`),
  **when** sampled at large n, **then** empirical mean and variance match closed forms
  within tolerance. ✅ (`test_gamma_distribution_moments_shape_at_least_one`,
  `test_gamma_distribution_moments_shape_below_one`, `test_beta_distribution_moments`)
- **Given** `poisson(λ)` and `binomial(n, p)`, **then** all samples are integers in the
  support and empirical means match `λ`/`np` within tolerance. ✅
  (`test_poisson_distribution`, `test_binomial_distribution`, plus degenerate-case and
  `MAX_LAMBDA`-boundary tests)
- **Given** the crate source, **when** grepped, **then** no Box–Muller implementation and
  no uniform clamp remain in `src/distributions.rs`. ✅
- **Given** `Cargo.toml`, **then** `rand_distr` is a regular dependency compatible with
  the resolved `rand` version. ✅ (`rand_distr = "0.6.0"`, resolves to the same
  `rand 0.10.2` already in the tree — confirmed via `cargo tree -p rand_distr`)
- **Given** the full test suite, **when** run (default and `parallel` features), **then**
  all pass, including every pre-existing degenerate-case test
  (`test_gamma_allows_zero_scale_degenerate`, `test_poisson_allows_zero_lambda_degenerate`,
  `test_geometric_allows_probability_one`, etc.) unmodified. ✅
