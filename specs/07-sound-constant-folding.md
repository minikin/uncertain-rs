# Spec 07 — Sound Constant Folding

**Status:** Implemented | **Effort:** Medium | **Module:** `src/computation.rs` (GraphOptimizer)

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

## Implementation notes

- **`constant_value: Option<T>`, not just a bool.** `ComputationNode::Leaf` gained a
  `constant_value: Option<T>` field (not merely an `is_constant: bool`) precisely so the
  last acceptance test holds *literally*: `is_constant_zero`/`is_constant_one`/the
  constant-folding match arms compare `constant_value` directly and never call `sample`
  at all — not even once after confirming constancy. A bool-flag version was tried first
  and still called `sample()` once to read the value for the zero/one comparison; storing
  the value directly removes that call entirely, at the cost of one extra (small,
  `Option<T>`) field per leaf.
- **The only way to get `constant_value: Some(_)`** is `ComputationNode::constant(value)`
  (new `pub` constructor) or the optimizer's own folded results, which are always
  rebuilt via `ComputationNode::constant` too — so a folded subexpression is eligible for
  further folding one level up, exactly like a literal. `ComputationNode::leaf(...)`
  (the pre-existing general constructor) always sets `constant_value: None`, including
  for a closure that happens to be deterministic (e.g. always returns `0.0`) — constancy
  is a structural fact about *how a node was built*, never an inference from behavior.
- **`Uncertain::point` now builds via `ComputationNode::constant`** instead of the
  generic `Uncertain::new`, so every `point(v)` is structurally constant to the optimizer
  without changing `point`'s public signature or behavior.
- **`GraphOptimizer` is not wired into `Uncertain<T>`'s arithmetic operators** (`+`, `*`,
  etc. in `src/operations/arithmetic.rs` build a graph via `Uncertain::with_node` and
  never call `GraphOptimizer::optimize`) — discovered while implementing this spec, and
  unchanged by it: `GraphOptimizer` remains an opt-in utility a caller invokes manually
  (as `examples/optimizations/*.rs` already did). This affects how the acceptance tests
  are exercised: the `bernoulli(0.99)`-style test uses a deterministic low-entropy
  closure (`ComputationNode::leaf(|| if rand::random::<f64>() < 0.99 {1.0} else {0.0})`)
  and asserts `is_constant_zero`/`is_constant_one` structurally rather than routing
  through `Uncertain::bernoulli` + `probability_exceeds` end-to-end, since there's no
  automatic optimization pass in that path to exercise. The `point(0.0) + x` /
  `point(2.0) * point(3.0)` / same-seed-equivalence tests do use the real public
  `Uncertain<T>` API (`Uncertain::point`, `Uncertain::normal`, `sample_with`), reaching
  into the `pub(crate) node` field and calling `GraphOptimizer::new().optimize(...)`
  manually, since that's the only way this functionality is invoked today. Whether to
  wire optimization into the arithmetic operators automatically is a separate, unscoped
  design decision — flagged here for Spec 08 (effective CSE) to pick up, since CSE has
  the same "never automatically invoked" characteristic.
- **`examples/optimizations/graph_optimization.rs`** had its literal `0`/`1` operands
  (`zero_node`/`one_node`) switched from `ComputationNode::leaf(...)` to
  `ComputationNode::constant(...)` — with the old sampling-based check removed, the demo
  would otherwise silently stop demonstrating identity elimination/constant folding
  (the operations just wouldn't fire) while still printing the same correct final number
  from evaluation. `examples/optimizations/common_subexpression_optimization.rs` only
  exercises CSE (not identity/constant folding) and needed no change.
- Ten pre-existing functions' CRAP scores *improved* as a side effect (mostly the
  `check_*_identities` helpers, simplified by moving the `Leaf`-pattern-match out of the
  call sites and into `is_constant_zero`/`is_constant_one` themselves); `crap_baseline.json`
  regenerated to capture this, plus the new `ComputationNode::constant` function and the
  removal of the old sample-based `is_constant`/`is_constant_bool` helpers (no longer
  needed now that callers pattern-match `constant_value` directly).
