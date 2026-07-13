# Spec 14 — Benchmarks & Coverage Gates in CI

**Status:** Pending | **Effort:** Medium | **Module:** `.github/workflows/`, `benches/`, `justfile`

## Context

`benches/performance_benchmarks.rs` (criterion, 386 lines) is never executed in CI, so
performance regressions land silently — exactly the failure mode Specs 05/06/08 must
guard against. Coverage is uploaded to Codecov on push but nothing gates on it: a PR can
drop coverage arbitrarily.

## Scope and Invariants

1. A CI bench job compiles and runs the criterion suite on PRs (reduced sample size /
   `--quick` is fine); results are compared against the base branch (critcmp,
   criterion-compare-action, or Bencher) and a summary lands in the PR (comment or job
   summary).
2. Regression policy: >10 % slowdown on core benches (sampling, graph evaluation,
   statistics) fails the job or flags prominently — threshold recorded in the workflow and
   tunable.
3. Bench noise control documented (fixed sample counts, warm-up; accept that shared
   runners are noisy — the gate flags, a human decides).
4. Coverage job gains a threshold: overall line coverage below a floor (start at the
   current measured value, don't aspire) fails; the floor lives in one place
   (`codecov.yml` or workflow env) and may only be raised.
5. Coverage runs on PRs too, not only pushes.
6. `just bench` runs the suite locally; `just bench-compare` compares against `main`.
7. Benches compile in the normal test gate (`cargo check --benches` or
   `cargo test --benches --no-run`) so they can't silently rot.

## Acceptance Tests

- **Given** a PR, **when** CI completes, **then** a bench-comparison summary is visible
  and the coverage check reports pass/fail against the floor.
- **Given** a synthetic 10× slowdown in a core bench (a `sleep` in the benched closure),
  **when** the bench job runs, **then** the regression is flagged/failed — demonstrated
  once, then reverted.
- **Given** a PR deleting a well-covered module's tests, **when** coverage runs, **then**
  the job fails the floor.
- **Given** a syntax error introduced only in `benches/`, **when** the normal test gate
  runs, **then** it fails (benches are compiled).
- **Given** `just bench` locally, **then** the criterion suite runs to completion.
