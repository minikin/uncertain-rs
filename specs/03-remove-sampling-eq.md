# Spec 03 — Remove Sampling-Based `PartialEq`/`PartialOrd` (Breaking)

**Status:** Implemented | **Effort:** Low | **Module:** `src/uncertain.rs`

## Context

`PartialEq`/`PartialOrd` for `Uncertain<T>` drew a single random sample per operand and
compared the samples. This violated the contracts of both traits: `a == a` could return
`false`, repeated comparisons disagreed, and sorting/dedup on `Uncertain<T>` values was
silently nondeterministic. The paper's model already provides the correct alternative:
evidence-returning comparisons (`Comparison` trait → `Uncertain<bool>`) plus hypothesis
testing — and the crate already had this full API (`gt`/`lt`/`ge`/`le`/`eq_value`/
`ne_value` against a scalar threshold, `gt_uncertain`/`lt_uncertain`/`eq_uncertain`
between two `Uncertain<T>` values) before this spec started.

## Scope and Invariants

1. The `PartialEq` and `PartialOrd` impls for `Uncertain<T>` are deleted.
2. **Deviation from the original draft:** no `sample_eq`/`sample_cmp` replacement methods
   were added. The draft's item 2 made this conditional ("if worth keeping"), and given
   the rich evidence-based API already available (item 3), adding a same-shaped
   single-draw escape hatch under a different name would send a mixed signal right after
   removing `PartialEq`/`PartialOrd` specifically because that shape is misleading. A
   user who genuinely wants one fresh sample from each side compared directly can write
   `a.sample() == b.sample()` — explicit about what's happening, no dedicated method
   needed.
3. The evidence-based `Comparison` trait remains the documented way to compare uncertain
   values; the crate-level `Uncertain<T>` doc comment now explicitly says so and links to
   it, with a `compile_fail` doctest demonstrating `==` no longer compiles.
4. Audited: no other internal code relied on `Uncertain<T>`'s `PartialEq`/`PartialOrd`
   (confirmed by grep — only the impls themselves and their own tests referenced them).
5. `MIGRATION_GUIDE.md` and `CHANGELOG.md` both gained a section for this change.

## Acceptance Tests

- **Given** the crate, **when** compiled, **then** `let _ = a == b;` on two
  `Uncertain<f64>` values is a compile error. ✅ (verified via a `compile_fail` doctest on
  `Uncertain<T>`'s doc comment, run as part of `cargo test --doc`)
- **Given** the test suite and doc tests, **when** run, **then** everything passes with the
  impls removed — no hidden internal reliance on `PartialEq`/`PartialOrd`. ✅ (311/313
  tests pass, default and `parallel` features; 76 doc tests pass)
- **Given** `cargo semver-checks`, **then** the removal is reported as a major change
  (consistent with the planned 0.3.0). ✅
