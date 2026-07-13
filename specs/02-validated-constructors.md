# Spec 02 — Validated Constructors (Breaking)

**Status:** Implemented | **Effort:** High | **Module:** `src/distributions.rs`, `src/error.rs`

## Context

The error enum was half dead code. Only `mixture`, `empirical`, and `categorical` returned
`Result`; the numeric constructors (`normal`, `uniform`, `exponential`, `log_normal`,
`beta`, `gamma`, `bernoulli`, `binomial`, `poisson`, `geometric`) performed zero validation.
`Uncertain::normal(0.0, -1.0)`, `exponential(0.0)` (→ `inf`) silently produced garbage/NaN
even though `InvalidParameter`, `NonFiniteParameter` existed in `src/error.rs` and were
used nowhere outside it. `CHANGELOG.md` already announced this breaking change and
references a `MIGRATION_GUIDE.md` and `examples/error_handling.rs` that did not exist.

## Scope and Invariants

1. Every distribution constructor returns `Result<Uncertain<T>>` and validates parameters
   before constructing a sampler. All `f64` parameters must be finite
   (`NonFiniteParameter` otherwise). Beyond finiteness, two policies were applied
   consistently:
   - **Strict-positive** where the boundary is mathematically undefined or breaks the
     algorithm: `exponential` (`rate > 0`), `beta` (`alpha`, `beta > 0`), `gamma`
     (`shape > 0`), `geometric` (`probability ∈ (0, 1]` — `0` never terminates the
     sampling loop).
   - **Degenerate-allowed** where the boundary is a well-defined point-mass distribution
     (and, for `uniform`/`normal(std_dev=0)`, an existing pre-spec test already relied on
     the degenerate case working): `normal` (`std_dev >= 0`), `uniform` (`min <= max`),
     `log_normal` (`sigma >= 0`, via `normal`), `gamma` (`scale >= 0`), `poisson`
     (`lambda >= 0`). `bernoulli`/`binomial` (`probability ∈ [0, 1]`, inclusive — existing
     tests exercise both bounds).
   - **Deviation from the original draft:** the draft specified `normal: std_dev > 0 (or
     >= 0 with 0 degenerating to point — pick one)` and `uniform: low < high` strictly.
     The strict form for `uniform` would have broken an existing test
     (`test_uniform_edge_cases`, `uniform(5.0, 5.0)`); `>=`/`<=` was chosen for both to
     match that already-legitimate degenerate behavior, and the same reasoning was
     extended to `gamma`'s `scale` and `poisson`'s `lambda` (both algorithmically well
     defined and safe at their zero boundary — verified with dedicated tests
     `test_gamma_allows_zero_scale_degenerate`, `test_poisson_allows_zero_lambda_degenerate`).
2. **Deviation from the original draft:** item 2 of the original draft ("statistics entry
   points validate their inputs: `pdf_kde`, quantile, confidence-interval, sample-count")
   is **out of scope for this spec**. Auditing the ripple showed 21 methods across
   `src/statistics.rs` take a `sample_count: usize` parameter, and `quantile`/
   `confidence_interval`/`pdf_kde`/`log_likelihood` alone have 42 call sites across tests,
   examples, and benches — a comparably large, independently-shippable change touching an
   entirely different module. Split out as [Spec 18](18-statistics-validation.md). The
   `InvalidQuantile`, `InvalidConfidence`, `InvalidBandwidth`, `InvalidSampleCount` error
   variants remain intentionally unconstructed until Spec 18 lands (not deleted now, since
   they're the right shape for exactly that spec — see the amended item 4 below).
3. Error messages state the parameter name, the received value, and the constraint
   (`UncertainError::InvalidParameter { parameter, value, constraint }` /
   `NonFiniteParameter { parameter, value }`).
4. **Amended:** every `UncertainError` variant used by distribution construction
   (`EmptyComponents`, `WeightCountMismatch`, `EmptyData`, `EmptyProbabilities`,
   `InvalidParameter`, `NonFiniteParameter`) now has a constructing site outside
   `error.rs`/its tests. `InvalidWeights` remains unconstructed and unused anywhere in the
   crate (not just statistics) — kept for now since Spec 18 or a future spec may need it
   for weight-vector validation in `mixture`/`categorical`; revisit if still unused once
   Spec 18 lands. `InvalidQuantile`/`InvalidConfidence`/`InvalidBandwidth`/
   `InvalidSampleCount` are deferred to Spec 18 per item 2 above.
5. `.expect(...)` inside the `empirical` and `categorical` sampling closures is eliminated:
   `empirical` uses `data.choose(&mut rng()).cloned().unwrap_or_else(|| data[0].clone())`
   and `categorical` uses an equivalent structurally-safe fallback — both provably
   non-panicking given the non-empty check already performed at construction time, without
   the word "expect" implying a real possible panic.
6. `MIGRATION_GUIDE.md` (0.2 → 0.3) and `examples/error_handling.rs` are created, making
   the existing CHANGELOG references true. README snippets are updated to the `Result` API.
7. Version bumps to 0.3.0 territory in CHANGELOG terms; actual release is out of scope.

## Acceptance Tests

- **Given** `Uncertain::normal(0.0, -1.0)`, **when** constructed, **then** it returns
  `Err(UncertainError::InvalidParameter { .. })` naming `std_dev`. ✅
- **Given** `Uncertain::normal(f64::NAN, 1.0)`, **then** `Err(NonFiniteParameter)`. ✅
- **Given** `Uncertain::exponential(0.0)`, **then** `Err`, and no sampler ever yields `inf`
  from a successfully constructed exponential. ✅
- **Given** valid parameters for every constructor, **when** constructed and sampled 1 000
  times, **then** all samples are finite and within the distribution's support. ✅ (full
  test suite green, default and `parallel` features, plus new degenerate-case tests)
- **Given** the crate source, **when** grepped, **then** every `UncertainError` variant
  used by distribution construction has a constructing site outside `error.rs` and its
  tests. ✅ (see amended item 4 for the two variants intentionally still deferred)
- **Given** the repo, **then** `MIGRATION_GUIDE.md` and `examples/error_handling.rs` exist,
  the example compiles and runs via `cargo run --example error_handling`, and all README
  doc snippets compile. ✅
- **Given** `just dev`, **when** run, **then** it's green: fmt, clippy (`-D warnings`),
  tests (default + `parallel`), doc tests, audit, deny, and crap regression check.
