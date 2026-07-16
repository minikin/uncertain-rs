# Spec 08 — Effective Common-Subexpression Elimination

**Status:** Implemented | **Effort:** Medium | **Module:** `src/computation.rs` (GraphOptimizer)

## Context

CSE is largely a no-op: `structural_hash` includes each leaf's `uuid`
(`src/computation.rs:424`), so two independently built but identical expressions never
unify — only literal `.clone()`s do. `tests/cache_optimization_tests.rs:151` carries a
TODO acknowledging this. Worse, `subexpression_cache` maps a bare `u64` hash to
`Box<dyn Any>` and `downcast_ref`s on lookup: a hash collision silently returns a wrong
node or `None`. Note an important semantic subtlety: two *distinct* `normal(0,1)` leaves
are independent random variables — unifying them changes correlation semantics; two
*clones* of one leaf are the same variable. CSE must respect that distinction.

## Scope and Invariants

1. Correlation-preserving unification rule, documented explicitly: nodes may be unified
   iff they are provably the *same* random variable (same leaf identity) or are fully
   deterministic given already-unified inputs (e.g. `map` with the same function pointer
   over the same unified input). Independent identically-distributed leaves are never
   unified — the uuid stays load-bearing for leaf identity, by design.
2. Within that rule, CSE actually fires: deterministic composite nodes with structurally
   equal (post-unification) inputs are deduplicated, e.g. `let s = &a + &b;` used in two
   branches, or `(a.clone() + b.clone())` built twice from the same leaves.
3. The cache becomes collision-safe: keys carry full structural identity (not a bare
   `u64`), and values are typed — a lookup can never return a node of the wrong type.
   `Box<dyn Any>` + hash-only keying is removed or made sound with full equality checks
   on collision.
4. The TODO test at `tests/cache_optimization_tests.rs:151` is resolved: it now asserts
   the unification that the rule permits.
5. `GraphOptimizer.subexpression_cache` stops being a `pub` field (README currently
   pokes at it); a method like `cse_hits()`/`stats()` exposes what's needed.
6. Semantics-preservation: for any graph, optimized and unoptimized results are identical
   under a fixed seed (extends Spec 07's property test to CSE rewrites).

## Acceptance Tests

- **Given** two clones of one `normal(0,1)` leaf combined as `x.clone() + x.clone()`,
  **when** CSE runs, **then** the operands unify (one evaluation) and, with a fixed seed,
  every sample of the sum is exactly `2 ×` the leaf's sample.
- **Given** two independently constructed `normal(0,1)` leaves added together, **when**
  CSE runs, **then** they do NOT unify and the sum's variance ≈ 2 (not 4).
- **Given** the same deterministic subexpression built twice from the same leaves,
  **when** optimized, **then** the graph contains one shared node (assert via node count
  or profiler), replacing the current TODO test.
- **Given** two structurally different nodes engineered to share a 64-bit hash (or a
  simulated collision via test hook), **when** looked up, **then** the cache never
  returns the wrong node.
- **Given** the README optimization example, **then** it compiles using only public
  methods — no direct field access.

## Implementation notes

- **The cache is keyed by a collision-safe `StructuralKey`, not a bare `u64`.**
  `GraphOptimizer.subexpression_cache` is now `HashMap<StructuralKey, Box<dyn Any + Send
  + Sync>>`, where `StructuralKey` (a private, non-generic enum mirroring
  `ComputationNode`'s shape: `Leaf(Uuid)` / `Binary(BinaryOperation, ..)` /
  `Unary(UnaryOpIdentity, ..)` / `Conditional(..)`) derives `PartialEq`/`Eq`/`Hash`. A
  `std::collections::HashMap` always resolves same-bucket entries via the key's `Eq`
  before returning one, so this makes the "never returns the wrong node" guarantee
  unconditional — it holds regardless of the quality of `StructuralKey`'s own `Hash`
  impl, not merely "unlikely to collide." `test_structural_key_lookup_is_correct_even_under_a_forced_hash_collision`
  proves this directly with a custom `Hasher` that maps every key to the same bucket.
- **A real, pre-existing hash collision — not just a theoretical one.** Before this
  spec, `structural_hash`'s `hash_structure` for `UnaryOp` hashed only the string
  `"unary"` and the operand — never the closure itself. Two `UnaryOp` nodes over the
  same operand with *different* closures (e.g. `x.clone().map(|v| v + 1.0)` vs.
  `x.clone().map(|v| v * 10.0)`) therefore always collided, by construction, not by
  chance. Combined with the old bare-`u64`-keyed cache, looking up the second node
  after caching the first would silently return the first node's (wrong) result — a
  real miscompilation, not a hypothetical one. Fixed by identifying a `UnaryOperation`'s
  closure via `Arc` pointer address (`unary_operation_identity`) in both
  `structural_key` (the cache's key) and `hash_structure` (the public
  `structural_hash()`, fixed for the same reason even though the cache no longer relies
  on it — it's still a public method documented for caching use).
  `test_cse_does_not_confuse_unary_ops_with_different_closures` locks in the fix.
- **Correlation-preserving unification needed no new logic, only new tests and a fixed
  cache.** The "same leaf id ⇒ same variable" rule was already implicit in
  `structural_hash`/`eliminate_common_subexpressions` hashing each `Leaf`'s `uuid` —
  clones (shared id) always unified, independently-built leaves (distinct ids) never
  did. This spec makes the rule explicit and tested:
  `test_cse_unifies_clones_of_the_same_leaf` (via `SampleContext` memoization: `x.clone()
  + x.clone()` evaluates to exactly `2x` in one context) and
  `test_cse_never_unifies_independently_constructed_leaves` (two independent
  `normal(0,1)`s sum to variance ≈ 2, not 4, checked over 50,000 samples with a 0.4
  tolerance).
- **The TODO test is resolved, not deleted.** `tests/cache_optimization_tests.rs`'s
  `test_structural_hash_consistency` (formerly annotated "TODO: … We'd need node-level
  sharing for true CSE") is reworded: two independently-built leaves hashing
  differently is the *correct*, by-design outcome (Scope invariant 1), not a limitation
  to fix — unifying them would be wrong the moment either leaf became non-constant. The
  TODO wording is removed; the test now documents why the assertion holds.
- **`subexpression_cache` is private; `cache_size()`/`cse_hits()` replace direct field
  access.** `cse_hits` is a new counter incremented on every cache hit inside
  `eliminate_common_subexpressions`. The README's Graph Optimization example
  (previously non-compiling — it called `expr.into_computation_node()`, a method that
  doesn't exist anywhere in the crate, and read the now-private field directly) is
  rewritten to build the graph via `ComputationNode::leaf`/`binary_op` directly
  (matching `examples/optimizations/common_subexpression_optimization.rs`'s existing
  pattern, since `GraphOptimizer` still isn't wired into `Uncertain<T>`'s operators —
  see Spec 07's implementation notes) and calls `optimizer.cache_size()`.
- **`BinaryOperation` gained `#[derive(Debug)]`** — needed for `StructuralKey`'s own
  `Debug` derive (used only in test assertions); harmless, since it's a plain
  value-less enum (`Add`/`Sub`/`Mul`/`Div`).
