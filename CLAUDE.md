# uncertain-rs Development Workflow

## Quality Gates (Pre-Commit Requirements)

Before any commit:

1. **`just dev`** — fmt-check, clippy (`-D warnings`), tests (default + `parallel` features),
   doc tests, `cargo audit`, `cargo deny check`, and a `cargo-crap` coverage/complexity
   regression check against `crap_baseline.json`.
2. **`just dev-mutants-diff`** — mutation testing on changed Rust files only (re-runs
   `just dev` first).

Never commit with a failing dogfood run or surviving mutants. Exception: changes touching
only `.md`/`.yml` files may skip both.

If a change intentionally improves coverage or reduces complexity, regenerate the baseline
with `just crap-update-baseline` and commit the updated `crap_baseline.json` alongside it.

## Specification as Law

All feature specs live in `specs/` using Given/When/Then acceptance criteria (see
`specs/README.md` for the index and suggested implementation order). **Never modify a spec
file without explicit permission from the maintainer.** When scope changes emerge during
implementation, propose the change and wait for approval before editing the spec.

## Essential Commands

```bash
cargo build --all-targets
cargo test --all-targets
cargo test --all-targets --features parallel
cargo test <test_name>
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
just dev
just dev-mutants-diff
just crap                   # regression check against crap_baseline.json
just crap-update-baseline   # regenerate the baseline after an intentional improvement
```

## Architecture

- **`src/uncertain.rs`** — the `Uncertain<T>` type: an `Arc<dyn Fn() -> T>` sampler paired
  with a lazy `ComputationNode` graph.
- **`src/distributions.rs`** — constructors (`normal`, `uniform`, `exponential`, `beta`,
  `bernoulli`, `mixture`, `empirical`, `categorical`, …).
- **`src/operations/`** — `Arithmetic`, `Comparison` (evidence-based, returns
  `Uncertain<bool>`), `LogicalOps` traits.
- **`src/statistics.rs`** — mean/variance/CDF/quantile/CI/entropy/KDE and the
  `LazyStats`/`ProgressiveStats`/`AdaptiveLazyStats` incremental estimators.
- **`src/hypothesis.rs`** — SPRT (`evaluate_hypothesis`), Bayesian updates, multiple
  hypothesis testing.
- **`src/computation.rs`** — the `ComputationNode` graph, `GraphOptimizer` (CSE, constant
  folding), `GraphVisualizer` (DOT), `GraphProfiler`.
- **`src/cache.rs`** — global TTL caches keyed by `(uuid, sample_count)`.
- **`src/error.rs`** — `UncertainError` (`thiserror`), crate `Result<T>` alias.

## MSRV & Toolchain

MSRV is declared once, in `Cargo.toml`'s `rust-version` — CI's `msrv` job reads it from
there rather than duplicating the value. `rust-toolchain.toml` pins the local dev toolchain
to `stable` with `rustfmt`/`clippy`.
