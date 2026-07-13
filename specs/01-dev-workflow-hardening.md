# Spec 01 — Dev-Workflow Hardening

**Status:** Pending | **Effort:** Medium | **Module:** repo root, `justfile`, `.github/workflows/`

## Context

Quality enforcement currently lives only in CI and optional cargo aliases. `Cargo.toml` has
no `rust-version` (MSRV 1.88.0 is declared only in `.github/workflows/ci.yml:11`), no
`[lints]` table (cast allows are scattered as module-level `#![allow]` in `cache.rs:1`,
`statistics.rs:1`, `distributions.rs:1`), and the repo has no `rust-toolchain.toml` or
`rustfmt.toml`. There is no mutation testing, no `cargo-deny`, no `cargo-semver-checks`.
cargo-crap's model: a single `just dev` gate plus `just dev-mutants-diff`, both mandatory
before any commit.

## Scope and Invariants

1. `Cargo.toml` gains `rust-version = "1.88.0"`; the CI MSRV job reads the manifest value
   (single source of truth), not a duplicated env var.
2. `Cargo.toml` gains a `[lints.rust]` / `[lints.clippy]` table: `warn` on `clippy::all`,
   `clippy::pedantic`, `clippy::nursery`; deny `warnings` stays in CI, not the manifest.
   Scattered module-level `#![allow]` moves into the `[lints]` table or is narrowed to
   item level with a justification comment.
3. `rust-toolchain.toml` pins the stable channel with `rustfmt`, `clippy` components.
4. `rustfmt.toml` added (explicit defaults at minimum, so formatting is versioned).
5. `deny.toml` + `cargo-deny` CI job: license allowlist, no yanked crates, advisories
   (replaces or supplements `cargo-audit`).
6. `.cargo-mutants.toml` added; `just mutants-diff` recipe runs `cargo mutants --in-diff`
   against `main` on changed Rust files only.
7. `cargo-semver-checks` CI job runs on PRs against the latest published version.
8. `just dev` = fmt-check + clippy (`-D warnings`, pedantic per `[lints]`) + all tests
   (default and `--features parallel`) + doc tests. Green `just dev` is the commit gate.
9. `CLAUDE.md` added at repo root documenting: the quality gates, the "specification as
   law" rule for `specs/`, essential commands, and architecture map (one paragraph per
   `src/` module).
10. The stale `rand` requirement/lockfile mismatch is reconciled: `Cargo.toml` requires
    `rand = "0.10.0"` while `Cargo.lock` pins 0.9.2. After this spec, `cargo update` is
    clean and the lockfile satisfies the manifest.
11. Nothing in this spec changes library behavior; the public API is untouched.

## Acceptance Tests

- **Given** a fresh checkout, **when** `cargo metadata` is inspected, **then** the package
  reports `rust_version = "1.88.0"` and a `[lints]` table is present in `Cargo.toml`.
- **Given** the pinned toolchain, **when** `just dev` runs, **then** fmt-check, clippy with
  the `[lints]` configuration, tests (both feature sets), and doc tests all pass.
- **Given** a diff that introduces an easily-mutable logic error surface, **when**
  `just mutants-diff` runs, **then** cargo-mutants executes only against files changed
  relative to `main` and reports caught/missed mutants.
- **Given** CI on a PR, **when** the workflow completes, **then** jobs for fmt+clippy,
  tests, MSRV (reading `rust-version` from the manifest), cargo-deny, and
  cargo-semver-checks have all executed.
- **Given** `Cargo.lock`, **when** `cargo tree -i rand` runs, **then** the resolved `rand`
  version satisfies the manifest requirement.
- **Given** the repo root, **when** listed, **then** `rust-toolchain.toml`, `rustfmt.toml`,
  `deny.toml`, `.cargo-mutants.toml`, and `CLAUDE.md` exist.
