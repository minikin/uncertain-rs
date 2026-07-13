# Spec 03 — Remove Sampling-Based `PartialEq`/`PartialOrd` (Breaking)

**Status:** Pending | **Effort:** Low | **Module:** `src/uncertain.rs`

## Context

`PartialEq`/`PartialOrd` for `Uncertain<T>` (`src/uncertain.rs:375-408`) draw a single
random sample per operand and compare the samples. This violates the contracts of both
traits: `a == a` can return `false`, repeated comparisons disagree, and sorting/dedup on
`Uncertain<T>` values is silently nondeterministic. The paper's model already provides the
correct alternative: evidence-returning comparisons (`Comparison` trait → `Uncertain<bool>`)
plus hypothesis testing.

## Scope and Invariants

1. The `PartialEq` and `PartialOrd` impls for `Uncertain<T>` are deleted.
2. If single-draw comparison is worth keeping, it is exposed as explicit, honestly-named
   methods (e.g. `sample_eq(&other) -> bool`, `sample_cmp(&other) -> Option<Ordering>`)
   whose docs state that each call draws one fresh sample per operand.
3. The evidence-based `Comparison` trait remains the documented way to compare uncertain
   values; README/docs steer users there.
4. All internal uses of `==`/`<` on `Uncertain<T>` (if any) are migrated.
5. MIGRATION_GUIDE.md gains a section for this change.

## Acceptance Tests

- **Given** the crate, **when** compiled, **then** `let _ = a == b;` on two
  `Uncertain<f64>` values is a compile error.
- **Given** `sample_eq`/`sample_cmp` (if kept), **when** called twice on the same pair of
  nondegenerate distributions, **then** results may differ and the doc comment says so.
- **Given** the test suite and doc tests, **when** run, **then** everything passes with the
  impls removed — no hidden internal reliance on `PartialEq`/`PartialOrd`.
- **Given** `cargo semver-checks`, **then** the removal is reported as a major change
  (consistent with the planned 0.3.0).
