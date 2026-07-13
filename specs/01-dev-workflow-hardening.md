# Spec 01 — Dev-Workflow Hardening

**Status:** Implemented | **Effort:** Medium | **Module:** repo root, `justfile`, `.github/workflows/`

## Context

Quality enforcement lived only in CI and optional cargo aliases. `Cargo.toml` had no
`rust-version` (MSRV 1.88.0 was declared only in `.github/workflows/ci.yml`), no `[lints]`
table, and the repo had no `rust-toolchain.toml` or `rustfmt.toml`. There was no mutation
testing, no `cargo-deny`, no `cargo-semver-checks`, and no coverage/complexity gate.
cargo-crap's own model: a single `just dev` gate plus `just dev-mutants-diff`, both
mandatory before any commit, with the tool dogfooding itself.

## Scope and Invariants

1. `Cargo.toml` gains `rust-version = "1.88.0"`; the CI `msrv` job reads the manifest value
   via `grep`, not a duplicated env var.
2. `Cargo.toml` gains `[lints.clippy] all = { level = "warn", priority = -1 }` — formalizes
   the lint level `cargo clippy` already applies. **Deviation from the original draft:**
   turning on `clippy::pedantic`/`clippy::nursery` here was found to produce ~150
   pre-existing findings unrelated to this spec's scope (see `just clippy-pedantic-check`
   dry run). Rather than silently expand scope or ship a red gate, pedantic/nursery stay
   opt-in via the existing `.cargo/config.toml` aliases, and the cleanup is tracked as its
   own backlog item: [Spec 17](17-clippy-pedantic-cleanup.md).
3. `rust-toolchain.toml` pins `channel = "stable"` with `rustfmt`, `clippy` components.
4. `rustfmt.toml` added (`edition = "2024"`, a no-op on current formatting — verified with
   `cargo fmt --all -- --check` before/after).
5. `deny.toml` + `cargo-deny` CI job (`EmbarkStudios/cargo-deny-action@v2`): license
   allowlist, deny unknown registries/git sources, advisories. Running this surfaced a real
   finding — `RUSTSEC-2026-0204` in `crossbeam-epoch 0.9.18` (transitive via `rayon`/
   `criterion`) — fixed in this branch via `cargo update -p crossbeam-epoch` (→ 0.9.20).
6. `.cargo-mutants.toml` added (excludes `examples/**`, `benches/**`); `just
   dev-mutants-diff` runs `cargo mutants --in-diff` against `origin/main` on changed Rust
   files only.
7. `cargo-semver-checks` CI job (`obi1kenobi/cargo-semver-checks-action@v2`) runs on PRs
   against the latest published version.
8. **cargo-crap dogfooding**, mirroring cargo-crap's own workflow. **Deviation from the
   original draft:** an absolute `--fail-above --threshold 30` gate was tried first and
   found 9/293 functions already exceed threshold — gating on that today would block all
   work immediately on pre-existing debt unrelated to whatever a contributor is touching.
   Switched to the **regression gate** cargo-crap's own README recommends: `crap_baseline.json`
   is committed at the repo root (`--format json --sort file`); `just crap` /
   CI's `crap` job run `--baseline crap_baseline.json --fail-regression` — new work must not
   make any function's score worse, but paying down the existing 9 offenders is separate,
   deliberate work (not blocked here). `just crap-update-baseline` regenerates the baseline
   when a change intentionally improves coverage/complexity.
9. `just dev` = fmt-check + clippy (`-D warnings`) + tests (default and `--features
   parallel`) + doc tests + `cargo audit` + `cargo deny check` + `just crap`. Green `just
   dev` is the commit gate.
10. `CLAUDE.md` added at repo root: quality gates, the "specification as law" rule for
    `specs/`, essential commands, and an architecture map (one paragraph per `src/` module).
11. The `rand` manifest/lockfile mismatch flagged during review was already resolved by a
    prior commit (`38dc401`, `rand` 0.9.2 → `^0.10.0`) before this spec started; `Cargo.lock`
    already resolves `rand 0.10.2`. No action needed here beyond the crossbeam-epoch bump
    in item 5.
12. Nothing in this spec changes library behavior; the public API is untouched.

## Acceptance Tests

- **Given** a fresh checkout, **when** `cargo metadata` is inspected, **then** the package
  reports `rust_version = "1.88.0"` and a `[lints]` table is present in `Cargo.toml`. ✅
- **Given** the pinned toolchain, **when** `just dev` runs, **then** fmt-check, clippy,
  tests (both feature sets), doc tests, audit, deny, and the crap regression check all
  pass. ✅ (verified locally; `just crap` reports `0 regressed`)
- **Given** a diff that introduces an easily-mutable logic error surface, **when** `just
  dev-mutants-diff` runs, **then** cargo-mutants executes only against files changed
  relative to `origin/main` and reports caught/missed mutants.
- **Given** CI on a PR, **when** the workflow completes, **then** jobs for fmt+clippy,
  tests, MSRV (reading `rust-version` from the manifest), cargo-deny,
  cargo-semver-checks, and the crap regression gate have all executed.
- **Given** `Cargo.lock`, **when** `cargo tree -i rand` runs, **then** the resolved `rand`
  version satisfies the manifest requirement. ✅ (already `0.10.2`)
- **Given** `cargo deny check`, **then** it exits 0 with no unresolved advisories. ✅
  (crossbeam-epoch bumped to 0.9.20)
- **Given** the repo root, **when** listed, **then** `rust-toolchain.toml`, `rustfmt.toml`,
  `deny.toml`, `.cargo-mutants.toml`, `crap_baseline.json`, and `CLAUDE.md` exist. ✅
