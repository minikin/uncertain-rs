# Spec 07 — Sound Constant Folding

**Status:** Pending | **Effort:** Medium | **Module:** `src/computation.rs` (GraphOptimizer)

## Context

`GraphOptimizer::is_constant` / `is_constant_zero` / `is_constant_one` decide whether a
node is constant by sampling it 3–4 times and comparing (`src/computation.rs:901,915,
1131,1146`). A genuinely random node can return equal samples by chance (guaranteed for
low-entropy discrete distributions, e.g. `bernoulli(0.99)` → four `true`s with p ≈ 0.96)
and be folded into a constant — a silent miscompilation that changes the semantics of the
user's model. An optimizer must never alter distributional semantics.

## Scope and Invariants

1. Constancy is decided structurally, never statistically: a node is constant iff it is a
   `point`/known-constant leaf or an operation whose operands are all constant. Sampling
   is removed from `is_constant*` entirely.
2. Identity rewrites (`x + 0`, `x * 1`, `x * 0`) fire only when the operand is a
   structural constant with exactly that value.
3. Semantics-preservation invariant, stated in the optimizer docs: for every rewrite rule,
   optimized and unoptimized graphs define identical distributions (not merely close).
4. If a distributional-equivalence check is ever desired as a heuristic, it must be opt-in,
   clearly named (`assume_constant_after_n_samples`), and off by default — out of scope
   for this spec.
5. Existing optimizer tests that relied on sampled constancy are rewritten structurally.
6. Composes with Spec 04: with seeded RNGs, an equivalence property test (same seed,
   optimize vs. not) becomes exact and is added.

## Acceptance Tests

- **Given** `bernoulli(0.99)` mapped to `f64`, **when** the optimizer runs, **then** the
  node is NOT treated as constant, and downstream `probability_exceeds` estimates match
  the unoptimized graph.
- **Given** `point(0.0) + x`, **when** optimized, **then** it folds to `x`; **given**
  `point(2.0) * point(3.0)`, **then** it folds to a constant `6.0` node.
- **Given** `normal(0, 1e-12)` (tiny but nonzero variance), **when** optimized, **then**
  it is not folded.
- **Given** any example graph in `examples/optimizations/`, **when** run optimized vs.
  unoptimized with the same seed (Spec 04), **then** sampled outputs are identical.
- **Given** the source of `GraphOptimizer`, **when** grepped, **then** no `sample()` call
  participates in any `is_constant*` decision.
