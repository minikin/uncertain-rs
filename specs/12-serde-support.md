# Spec 12 — Serde Support (feature `serde`)

**Status:** Pending | **Effort:** Low | **Module:** `src/error.rs`, `src/statistics.rs`, `src/distributions.rs`, `Cargo.toml`

## Context

Nothing in the crate serializes. `Uncertain<T>` itself holds an `Arc<dyn Fn() -> T>` and
cannot (and should not) be serialized — but its *inputs* and *outputs* can: distribution
parameter descriptors, computed statistics (histograms, CIs, summaries), and errors.
Serialization enables config-driven models, persisted results, and cross-service use —
table stakes for a "top-tier" crate.

## Scope and Invariants

1. A `serde` cargo feature (off by default) gates all of it; `default = []` unchanged.
2. A serializable `DistributionSpec` enum mirrors every constructor
   (`Normal { mean, std_dev }`, `Uniform { low, high }`, …, including `Mixture`,
   `Empirical`, `Categorical`), with `TryFrom<DistributionSpec> for Uncertain<f64>` (and
   the appropriate typed variants) performing Spec 02 validation — round-tripping a spec
   through JSON and constructing from it behaves identically to calling the constructor.
3. Result types derive `Serialize` (+ `Deserialize` where meaningful): confidence
   intervals, histogram buckets, summary-statistics structs, hypothesis-test outcomes, and
   `UncertainError` (Serialize only).
4. No serde types leak into the default feature: `cargo check` with default features
   compiles without serde in the tree (verify with `cargo tree`).
5. Doc examples show JSON round-trips; `#[serde(deny_unknown_fields)]` on specs so config
   typos fail loudly.

## Acceptance Tests

- **Given** `DistributionSpec::Normal { mean: 0.0, std_dev: 1.0 }`, **when** serialized to
  JSON and back and converted via `TryFrom`, **then** sampling with a fixed seed matches
  `Uncertain::normal(0.0, 1.0)` with the same seed exactly.
- **Given** a JSON spec with an invalid parameter (`std_dev: -1`), **when** converted,
  **then** the Spec 02 validation error is returned.
- **Given** a JSON spec with an unknown field, **when** deserialized, **then** it errors.
- **Given** `cargo check` (no features), **then** serde is absent from `cargo tree`;
  **given** `cargo test --features serde`, **then** all round-trip tests pass.
- **Given** a computed histogram/CI, **when** serialized, **then** the JSON is stable and
  documented (field names asserted in a snapshot test).
