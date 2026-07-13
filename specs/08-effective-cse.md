# Spec 08 — Effective Common-Subexpression Elimination

**Status:** Pending | **Effort:** Medium | **Module:** `src/computation.rs` (GraphOptimizer)

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
