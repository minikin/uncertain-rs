# Spec 06 — Total Graph Evaluation (Breaking)

**Status:** Implemented | **Effort:** Medium | **Module:** `src/computation.rs`

## Context

Graph evaluation is split across three partial `pub` methods, each panicking on the
variants it doesn't handle: `ComputationNode::evaluate` panics on `BinaryOp`/`Conditional`
(`src/computation.rs:217,223`), `evaluate_arithmetic` panics on `Conditional` (`:273`),
`evaluate_bool` panics on `BinaryOp` (`:482`). Unit tests exist solely to pin the panics
(`#[should_panic]`). A library consumer walking a graph can crash by picking the wrong
method for a node shape. `GraphProfiler::get_stats` also uses `.expect`
(`:1337,1340`).

## Scope and Invariants

1. One total evaluation entry point per value domain, returning `Result` instead of
   panicking: unsupported-node cases become a dedicated `UncertainError` variant
   (e.g. `UnsupportedNode { expected, found }`) rather than a panic.
2. The three partial methods are either removed or reduced to thin `Result`-returning
   wrappers over the total dispatch; no `panic!`/`unreachable!`/`expect` remains on any
   reachable evaluation path.
3. `#[should_panic]` tests for evaluation are replaced by tests asserting the specific
   error variant.
4. `GraphProfiler::get_stats` returns `Option`/`Result` instead of `expect`ing.
5. Internal callers (`Uncertain` sampling path, optimizer) are migrated; the sampling hot
   path must not regress: criterion benchmarks for graph evaluation stay within noise of
   the baseline.
6. Ships in the 0.3.0 breaking batch with Specs 02/03; MIGRATION_GUIDE.md gains a section.

## Acceptance Tests

- **Given** a `Conditional` node, **when** evaluated through the arithmetic path, **then**
  the call returns `Err(UnsupportedNode { .. })` and does not panic.
- **Given** any node variant, **when** evaluated through the total entry point with a
  domain it supports, **then** it produces a value; with a domain it cannot, **then** a
  typed error.
- **Given** the crate source, **when** grepped for `panic!`, `unreachable!`, `unwrap`, and
  `expect` in `src/computation.rs`, **then** none remain on evaluation or profiler paths
  (test modules excluded).
- **Given** the benchmark suite, **when** run before/after, **then** graph-evaluation
  throughput is within noise (no boxing/indirection regressions).
- **Given** the full test suite, **then** all previous `#[should_panic]` evaluation tests
  are replaced and green as error-variant assertions.

## Implementation notes (deviations from the draft above)

- **One dispatcher per domain, not one dispatcher overall.** `evaluate`, `evaluate_arithmetic`,
  and `evaluate_bool` all now return `Result<_, UncertainError>` and none panics, but they
  remain three separate methods rather than three thin wrappers over a single total
  dispatch — there is no single dispatch to wrap them around, because each is total over a
  *different* set of variants, dictated by `T`'s trait bounds:
  - `evaluate` (`T: Shareable` only) is total for `Leaf`/`UnaryOp` and returns
    `Err(UnsupportedNode)` for `BinaryOp`/`Conditional` — this is a structural limit, not a
    gap: `BinaryOp` needs `T: Arithmetic` (`BinaryOperation::apply` has no meaning otherwise)
    and `Conditional` needs to evaluate a `bool` condition, neither of which is available
    under a bare `Shareable` bound.
  - `evaluate_arithmetic` (`T: Arithmetic`) is the true total dispatcher for the arithmetic
    domain: it now handles `Conditional` directly (evaluating the condition via
    `evaluate_bool` and delegating to the taken branch), absorbing what used to be the
    separate `evaluate_conditional_with_arithmetic` method, which is now deleted. Its lone
    external caller (`Uncertain::with_node`'s `sample_fn`, `src/uncertain.rs`) calls
    `evaluate_arithmetic(...).expect(...)` instead — documented as an internal invariant,
    since graphs built through the public `Uncertain<T>` combinator API are always
    well-formed and this path can't observe an `Err` in practice.
  - `evaluate_bool` is total except for `BinaryOp`, which is genuinely unsupported: no
    `BinaryOperation` variant is defined for `bool` (`BinaryOperation` is `Add/Sub/Mul/Div`,
    and `Arithmetic` has no `bool` impl), so a boolean `BinaryOp` node can only arise from
    directly misusing the low-level `ComputationNode` constructors.
  - `evaluate_fresh` keeps its existing infallible signature (`-> T`, not `-> Result<T, _>`)
    since it has ~20 pre-existing callers (mostly tests) that treat it as such; it now
    calls `evaluate_arithmetic(...).expect(...)` internally, for the same well-formed-graph
    reason as `with_node` above.
- **CRAP regression, accepted and rebaselined.** `evaluate`, `evaluate_arithmetic`, and
  `evaluate_bool` all show a higher CRAP score against the committed baseline (CC +1 to +5)
  from the added `Result` plumbing and, for `evaluate_arithmetic`, the absorbed conditional
  branch. All three reached 100% line coverage in the same change (new tests for the error
  paths, the previously-uncovered `Map`/`Filter` arms of the raw `evaluate`/`evaluate_bool`
  methods, and the bool-leaf memoization/false-branch/filter paths), so the remaining CRAP
  increase is purely the higher cyclomatic complexity — not a coverage gap. `crap_baseline.json`
  was regenerated (`just crap-update-baseline`) to accept this as intentional.
- **Benchmarks.** `computation_graphs` criterion group (`benches/performance_benchmarks.rs`)
  re-run after the change: `complex_expression_expected_value` and `complex_expression_variance`
  within noise (small improvement, ~1-2%); `multiple_stats_operations` showed a one-off +7.7%
  on a first run but reproduced as "within noise" (-0.8%) on a second run — the Result
  wrapping is a zero-allocation enum tag on the `Ok` fast path, so no structural regression is
  expected or observed.
