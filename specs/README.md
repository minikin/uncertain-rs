# uncertain-rs — Specifications

Spec-driven development, modeled after [cargo-crap](https://github.com/minikin/cargo-crap):
each spec is acceptance criteria ("specification as law"). **Never modify a spec file
without explicit permission from the maintainer.** Implementation is done spec-by-spec;
a spec is closed only when every acceptance test passes and `just dev` is green.

## Index

| #   | Spec                                                             | Priority | Effort | Breaking   | Status                |
| --- | ---------------------------------------------------------------- | -------- | ------ | ---------- | --------------------- |
| 01  | [Dev-workflow hardening](01-dev-workflow-hardening.md)           | P0       | Medium | no         | Implemented           |
| 02  | [Validated constructors](02-validated-constructors.md)           | P0       | High   | **yes**    | Implemented           |
| 03  | [Remove sampling-based Eq/Ord](03-remove-sampling-eq.md)         | P0       | Low    | **yes**    | Implemented           |
| 04  | [Seedable RNG & reproducibility](04-seedable-rng.md)             | P0       | High   | no         | Implemented           |
| 05  | [Correct sampling via rand_distr](05-rand-distr-sampling.md)     | P0       | Medium | no         | Implemented           |
| 06  | [Total graph evaluation](06-total-graph-evaluation.md)           | P0       | Medium | **yes**    | Implemented           |
| 07  | [Sound constant folding](07-sound-constant-folding.md)           | P0       | Medium | no         | Implemented           |
| 08  | [Effective CSE](08-effective-cse.md)                             | P1       | Medium | no         | Implemented           |
| 09  | [Consistent variance policy](09-variance-policy.md)              | P1       | Low    | behavioral | Pending               |
| 10  | [Bounded caches, no NaN paths](10-bounded-caches.md)             | P1       | Medium | no         | Partially implemented |
| 11  | [Property-based testing](11-property-based-testing.md)           | P1       | Medium | no         | Pending               |
| 12  | [Serde support](12-serde-support.md)                             | P2       | Low    | no         | Pending               |
| 13  | [More distributions](13-more-distributions.md)                   | P2       | Medium | no         | Pending               |
| 14  | [Benchmarks & coverage gates in CI](14-bench-regression-ci.md)   | P1       | Medium | no         | Pending               |
| 15  | [Docs & meta cleanup](15-docs-and-meta.md)                       | P1       | Low    | no         | Pending               |
| 16  | [Release automation](16-release-automation.md)                   | P2       | Low    | no         | Pending               |
| 17  | [Clippy pedantic/nursery cleanup](17-clippy-pedantic-cleanup.md) | P2       | Medium | no         | Pending               |
| 18  | [Statistics entry-point validation](18-statistics-validation.md) | P0       | High   | **yes**    | Implemented           |
| 19  | [Deflake existing test suite](19-deflake-existing-tests.md)      | P1       | High   | no         | Pending               |

## Suggested order

1. **01** first — it installs the quality gates every later spec is verified against.
   _(Implemented; see the spec for two deliberate deviations from the original draft:
   pedantic/nursery lints stay opt-in — split out as Spec 17 — and cargo-crap uses a
   regression gate against a committed baseline rather than an absolute threshold, since
   9 functions already exceed it today.)_
2. **04 → 05** (seeding, then correct samplers) — reproducibility unblocks deflaked tests
   used by every other spec. _(Both implemented. 04 used `rand_chacha::ChaCha8Rng`
   directly instead of `rand::rngs::StdRng` — the latter is explicitly documented as
   non-portable in this `rand` version, which would have broken the determinism
   guarantee. Retrofitting every existing statistical test to a fixed seed was split out
   to Spec 19, since it's a large mechanical effort of its own kind, distinct from
   building the seeding infrastructure. 05 found `rand_distr`'s `Gamma`/`Poisson`
   constructors stricter than this crate's committed degenerate-case behavior
   (`scale`/`lambda` of `0`), handled via special-casing before ever calling
   `rand_distr`; measured a 65.5% speedup for normal sampling.)_
3. **02, 03, 06, 18** — the breaking API changes, batched into one 0.3.0 release.
   _(All four implemented. 02's statistics-entry-point-validation half was split out to
   18 — see 02's spec for why: a comparably large, independently-shippable ripple through
   a different module. 06 kept `evaluate`/`evaluate_arithmetic`/`evaluate_bool` as three
   separate `Result`-returning dispatchers, one total per value domain, rather than one
   dispatcher wrapped three ways — see 06's spec's implementation notes for why a single
   dispatcher isn't meaningful across domains with different trait bounds. 18 validates
   at construction time for `LazyStats`/`AdaptiveLazyStats` (sample count fixed once,
   reused by every accessor) and at call time everywhere else (sample
   count/quantile/confidence/bandwidth are genuine per-call parameters) — see 18's spec
   for the full per-method breakdown.)_
4. **07 → 08** — optimizer correctness before optimizer effectiveness.
   _(Both implemented. 07: constancy decided via a `constant_value: Option<T>` field on
   `Leaf` — set only by the new `ComputationNode::constant`/`Uncertain::point` — rather
   than the old sample-3x-and-compare check, which could silently fold a low-entropy
   distribution (e.g. `bernoulli(0.99)`) into a constant. Discovered along the way:
   `GraphOptimizer` isn't wired into `Uncertain<T>`'s arithmetic operators at all — it's
   an opt-in utility invoked manually, unchanged by 07 but relevant to 08. 08: the
   subexpression cache is now keyed by a collision-safe `StructuralKey` (not a bare
   `u64`), fixing a real pre-existing bug where two `UnaryOp`s over the same operand
   with different closures hashed identically (the old hash never looked at the
   closure) and could silently return one closure's cached result for the other's
   lookup. `subexpression_cache` is now private, with `cache_size()`/`cse_hits()`
   accessors; the README's optimizer example, previously non-compiling, is fixed. The
   `GraphOptimizer`-not-wired-in gap from 07 is unchanged — still out of scope.)_
5. **09, 10, 11, 14, 15, 17, 19** in any order.
6. **12, 13, 16** — feature/polish tail.

## Spec format

Each spec follows the cargo-crap structure:

```
# Spec NN — Title
**Status:** Pending | Implemented | **Effort:** Low/Medium/High | **Module:** src/...

## Context          — why, with concrete file:line evidence
## Scope and Invariants — the design boundaries; what MUST hold
## Acceptance Tests — Given/When/Then scenarios; the definition of done
```
