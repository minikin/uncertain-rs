# Spec 17 — Clippy Pedantic/Nursery Cleanup

**Status:** Pending | **Effort:** Medium | **Module:** all of `src/`

## Context

Discovered while implementing [Spec 01](01-dev-workflow-hardening.md): running
`cargo clippy --all-targets --all-features -- -D clippy::all -D clippy::pedantic -D
clippy::nursery` against the current source produces ~150 findings (missing `const fn`,
`use Self`, `unnecessary structure name repetition`, `suboptimal_flops` on manual
mul-add patterns, `option_if_let_else`, etc.). None are bugs — mostly idiom/ergonomics —
but there are too many to fix incidentally inside an infra spec. `.cargo/config.toml`
already exposes `clippy-pedantic`/`clippy-strict` aliases for running this manually.

## Scope and Invariants

1. Fix pedantic/nursery findings file by file (or in a small number of grouped PRs), each
   independently reviewable — this is not a single mechanical sweep given `suboptimal_flops`
   changes (`mul_add`) affect floating-point rounding and deserve a second look each.
2. Once clean, `Cargo.toml`'s `[lints.clippy]` table gains `pedantic = { level = "warn",
   priority = -1 }` and `nursery = { level = "warn", priority = -1 }`, and CI's default
   `-D warnings` gate starts enforcing them (no more opt-in aliases needed).
3. Any lint that's a poor fit for this codebase (e.g. a pedantic lint that actively hurts
   readability in statistical code) is allowed at `[lints.clippy]` level with a one-line
   justification — not silenced ad hoc per call site.
4. No behavior change from mechanical fixes (`const fn`, `Self` substitution); `suboptimal_flops`
   (`mul_add`) changes get a numerical-equivalence check (the property tests from
   [Spec 11](11-property-based-testing.md), if landed first) since `mul_add` uses fused
   multiply-add and can change the last bit of a float result.

## Acceptance Tests

- **Given** the full crate, **when** `cargo clippy --all-targets --all-features -- -D
  clippy::all -D clippy::pedantic -D clippy::nursery` runs, **then** it exits 0.
- **Given** `Cargo.toml`, **then** `[lints.clippy]` includes `pedantic` and `nursery` at
  `warn`, and CI's default lint job (no special flags) fails on a newly introduced pedantic
  violation — demonstrated once, then reverted.
- **Given** the statistics test suite, **when** run before/after any `mul_add` rewrite,
  **then** results agree within the existing tolerances (no unexpected precision
  regressions).
- **Given** `.cargo/config.toml`, **then** the `clippy-pedantic`/`clippy-strict` aliases may
  be removed or kept as redundant convenience aliases — no longer load-bearing.
