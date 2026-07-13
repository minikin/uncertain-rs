# Spec 15 — Docs & Meta Cleanup

**Status:** Pending | **Effort:** Low | **Module:** repo root, `src/lib.rs`, `README.md`, `CHANGELOG.md`

## Context

Several trust-eroding inconsistencies: `CHANGELOG.md`'s `[Unreleased]` announces a
breaking change referencing `MIGRATION_GUIDE.md` and `examples/error_handling.rs` —
neither exists (Spec 02 creates them). `SECURITY.md` is the uncustomized GitHub template
(placeholder versions "5.1.x / 4.0.x" for a 0.2.0 crate, boilerplate "Use this section to
tell people…"). `CONTRIBUTING.md` doesn't exist though README invites contributions.
`lib.rs` lacks `#![warn(missing_docs)]` and `#![forbid(unsafe_code)]`. The README
optimization example accesses the `pub` implementation field
`optimizer.subexpression_cache` (`src/computation.rs:531`; Spec 08 privatizes it). Some
`# Panics` sections are boilerplate ("should not happen under normal circumstances",
`src/distributions.rs:41-42,105-107`).

## Scope and Invariants

1. `SECURITY.md` rewritten for this crate: actual supported versions, a real disclosure
   channel (GitHub private vulnerability reporting), response expectations.
2. `CONTRIBUTING.md` created: dev setup, `just dev` gate, spec-driven workflow (points at
   `specs/README.md`), commit/PR conventions, MSRV policy.
3. `CHANGELOG.md` reconciled with reality: `[Unreleased]` describes what is actually
   unreleased; dangling file references either exist (post Spec 02) or are removed;
   version links resolve.
4. `src/lib.rs` gains `#![forbid(unsafe_code)]` and `#![warn(missing_docs)]`; every
   resulting `missing_docs` warning is fixed.
5. Boilerplate `# Panics` sections are replaced with precise conditions — or deleted when
   the path can no longer panic (post Specs 02/06).
6. README examples are all compile-checked (doc-tested include or CI example run); the
   optimization example uses only public API.
7. `docs.rs` metadata in `Cargo.toml` (`[package.metadata.docs.rs]`) builds with
   `--all-features` so `parallel`/`serde` APIs are visible.

## Acceptance Tests

- **Given** `cargo doc --all-features`, **when** built with `missing_docs` warnings denied,
  **then** zero warnings.
- **Given** the repo, **then** `CONTRIBUTING.md` exists and `SECURITY.md` contains no
  GitHub-template placeholder text and lists versions consistent with `Cargo.toml`.
- **Given** every README code block, **when** executed as a doc test or example, **then**
  it compiles and runs against the current API.
- **Given** `CHANGELOG.md`, **then** every referenced file exists in the repo and the
  `[Unreleased]` section matches `git log` since the last tag.
- **Given** the source, **when** grepped for "should not happen", **then** no `# Panics`
  boilerplate remains.
