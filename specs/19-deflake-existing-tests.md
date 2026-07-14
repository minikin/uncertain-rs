# Spec 19 — Deflake Existing Test Suite with Seeded RNG

**Status:** Pending | **Effort:** High | **Module:** `src/*.rs` test modules, `tests/*.rs`, `benches/*.rs`

## Context

Split out from [Spec 04](04-seedable-rng.md) during implementation: item 5 of the
original draft ("every statistical assertion in unit/integration tests switches to a
fixed-seed RNG") is a large, mostly-mechanical retrofit across hundreds of existing tests
that already pass today (loosely, via tolerance checks over unseeded draws) — a
comparably large, independently-shippable effort from building the seeding
infrastructure itself, which is why Spec 04 shipped a focused new test suite
(`tests/seeded_rng_tests.rs`) proving the seeding guarantees instead of touching every
pre-existing test.

## Scope and Invariants

1. Audit every `#[test]` in `src/*.rs` test modules and `tests/*.rs` that samples a
   distribution and asserts a statistical property (mean/variance/proportion/range within
   a tolerance) with **no** fixed seed.
2. Where a test's *point* is exercising exact behavior (e.g. moment-matching against a
   closed form, boundary behavior), switch it to `sample_with`/`take_samples_with` with a
   fixed seed chosen so the test is deterministic — same numeric result every run, not
   just "passes with high probability."
3. Where a test's point is genuinely statistical (e.g. "roughly follows a normal
   distribution"), keep the tolerance-based assertion but seed it anyway so a failure is
   reproducible (a flake today can't be reproduced locally; a seeded flake can be, even if
   the assertion itself stays probabilistic).
4. Tests intentionally exercising the *unseeded* path (e.g. spec 04's own
   `sample_without_explicit_rng_is_unaffected_by_seeding_infrastructure`) are explicitly
   exempted and documented as such.
5. No change to what each test is actually verifying — this is a mechanical
   determinism-hardening pass, not a test-logic rewrite. Any test whose assertion looks
   wrong while doing this pass gets flagged, not silently "fixed" here.

## Acceptance Tests

- **Given** the full test suite, **when** run 20 times in a loop, **then** there are zero
  statistical flakes (the original Spec 04 acceptance criterion, now fully in scope here).
- **Given** a test that previously used unseeded sampling, **when** re-run twice, **then**
  it asserts the exact same numeric result both times (for tests converted to exact-value
  assertions) or is annotated with why it remains tolerance-based.
- **Given** `just dev` and `just dev-mutants-diff`, **when** run, **then** they're green —
  no coverage regression from the conversion.
