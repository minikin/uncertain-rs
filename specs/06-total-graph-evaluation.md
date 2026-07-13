# Spec 06 — Total Graph Evaluation (Breaking)

**Status:** Pending | **Effort:** Medium | **Module:** `src/computation.rs`

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
