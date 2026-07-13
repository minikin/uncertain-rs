# Spec 02 — Validated Constructors (Breaking)

**Status:** Pending | **Effort:** High | **Module:** `src/distributions.rs`, `src/error.rs`, `src/statistics.rs`

## Context

The error enum is half dead code. Only `mixture`, `empirical`, and `categorical` return
`Result`; the numeric constructors (`normal`, `uniform`, `exponential`, `log_normal`,
`beta`, `gamma`, `bernoulli`, `binomial`, `poisson`, `geometric`) perform zero validation.
`Uncertain::normal(0.0, -1.0)`, `exponential(0.0)` (→ `inf`), and `pdf_kde(x, n, 0.0)`
silently produce garbage/NaN even though `InvalidParameter`, `NonFiniteParameter`,
`InvalidBandwidth`, `InvalidQuantile`, `InvalidConfidence`, `InvalidSampleCount` exist in
`src/error.rs` and are used nowhere outside it. `CHANGELOG.md` already announces this
breaking change and references a `MIGRATION_GUIDE.md` and `examples/error_handling.rs`
that do not exist.

## Scope and Invariants

1. Every distribution constructor returns `Result<Uncertain<T>>` and validates parameters
   before constructing a sampler:
   - all `f64` parameters must be finite (`NonFiniteParameter` otherwise);
   - `normal`: `std_dev > 0` (or `>= 0` with `0` degenerating to `point` — pick one and
     document it); `uniform`: `low < high`; `exponential`: `rate > 0`; `log_normal`:
     `scale > 0`; `beta`/`gamma`: shape/rate `> 0`; `bernoulli`: `p ∈ [0,1]`;
     `binomial`: `p ∈ [0,1]`; `poisson`: `lambda > 0`; `geometric`: `p ∈ (0,1]`.
2. Statistics entry points validate their inputs: `pdf_kde` rejects `bandwidth <= 0`
   (`InvalidBandwidth`), quantile functions reject values outside `[0,1]`
   (`InvalidQuantile`), confidence-interval functions reject confidence outside `(0,1)`
   (`InvalidConfidence`), sample-count parameters of `0` where statistically meaningless
   return `InvalidSampleCount` instead of NaN.
3. Error messages state the parameter name, the received value, and the constraint.
4. No error variant in `src/error.rs` remains unconstructed by library code; any variant
   that still has no producer after this spec is deleted.
5. `.expect(...)` inside the `empirical` (`src/distributions.rs:123`) and `categorical`
   (`:179`) sampling closures is eliminated: validation at construction time makes the
   sampler infallible, and the closures encode that without `expect`.
6. `MIGRATION_GUIDE.md` (0.2 → 0.3) and `examples/error_handling.rs` are created, making
   the existing CHANGELOG references true. README snippets are updated to the `Result` API.
7. Version bumps to 0.3.0 territory in CHANGELOG terms; actual release is out of scope.

## Acceptance Tests

- **Given** `Uncertain::normal(0.0, -1.0)`, **when** constructed, **then** it returns
  `Err(UncertainError::InvalidParameter { .. })` naming `std_dev`.
- **Given** `Uncertain::normal(f64::NAN, 1.0)`, **then** `Err(NonFiniteParameter)`.
- **Given** `Uncertain::exponential(0.0)`, **then** `Err`, and no sampler ever yields `inf`
  from a successfully constructed exponential.
- **Given** valid parameters for every constructor, **when** constructed and sampled 1 000
  times, **then** all samples are finite and within the distribution's support.
- **Given** `pdf_kde(x, n, 0.0)`, **then** `Err(InvalidBandwidth)`; **given** a quantile
  request of `1.5`, **then** `Err(InvalidQuantile)`.
- **Given** the crate source, **when** grepped, **then** every `UncertainError` variant has
  at least one constructing site outside `error.rs` and its tests.
- **Given** the repo, **then** `MIGRATION_GUIDE.md` and `examples/error_handling.rs` exist,
  the example compiles and runs via `cargo run --example error_handling`, and all README
  doc snippets compile.
