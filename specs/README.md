# uncertain-rs — Specifications

Spec-driven development, modeled after [cargo-crap](https://github.com/minikin/cargo-crap):
each spec is acceptance criteria ("specification as law"). **Never modify a spec file
without explicit permission from the maintainer.** Implementation is done spec-by-spec;
a spec is closed only when every acceptance test passes and `just dev` is green.

## Index

| #  | Spec | Priority | Effort | Breaking | Status |
|----|------|----------|--------|----------|--------|
| 01 | [Dev-workflow hardening](01-dev-workflow-hardening.md) | P0 | Medium | no | Implemented |
| 02 | [Validated constructors](02-validated-constructors.md) | P0 | High | **yes** | Implemented |
| 03 | [Remove sampling-based Eq/Ord](03-remove-sampling-eq.md) | P0 | Low | **yes** | Implemented |
| 04 | [Seedable RNG & reproducibility](04-seedable-rng.md) | P0 | High | no | Implemented |
| 05 | [Correct sampling via rand_distr](05-rand-distr-sampling.md) | P0 | Medium | no | Pending |
| 06 | [Total graph evaluation](06-total-graph-evaluation.md) | P0 | Medium | **yes** | Pending |
| 07 | [Sound constant folding](07-sound-constant-folding.md) | P0 | Medium | no | Pending |
| 08 | [Effective CSE](08-effective-cse.md) | P1 | Medium | no | Pending |
| 09 | [Consistent variance policy](09-variance-policy.md) | P1 | Low | behavioral | Pending |
| 10 | [Bounded caches, no NaN paths](10-bounded-caches.md) | P1 | Medium | no | Partially implemented |
| 11 | [Property-based testing](11-property-based-testing.md) | P1 | Medium | no | Pending |
| 12 | [Serde support](12-serde-support.md) | P2 | Low | no | Pending |
| 13 | [More distributions](13-more-distributions.md) | P2 | Medium | no | Pending |
| 14 | [Benchmarks & coverage gates in CI](14-bench-regression-ci.md) | P1 | Medium | no | Pending |
| 15 | [Docs & meta cleanup](15-docs-and-meta.md) | P1 | Low | no | Pending |
| 16 | [Release automation](16-release-automation.md) | P2 | Low | no | Pending |
| 17 | [Clippy pedantic/nursery cleanup](17-clippy-pedantic-cleanup.md) | P2 | Medium | no | Pending |
| 18 | [Statistics entry-point validation](18-statistics-validation.md) | P0 | High | **yes** | Implemented |
| 19 | [Deflake existing test suite](19-deflake-existing-tests.md) | P1 | High | no | Pending |

## Suggested order

1. **01** first — it installs the quality gates every later spec is verified against.
   *(Implemented; see the spec for two deliberate deviations from the original draft:
   pedantic/nursery lints stay opt-in — split out as Spec 17 — and cargo-crap uses a
   regression gate against a committed baseline rather than an absolute threshold, since
   9 functions already exceed it today.)*
2. **04 → 05** (seeding, then correct samplers) — reproducibility unblocks deflaked tests
   used by every other spec. *(04 implemented; used `rand_chacha::ChaCha8Rng` directly
   instead of `rand::rngs::StdRng` — the latter is explicitly documented as non-portable
   in this `rand` version, which would have broken the determinism guarantee. Retrofitting
   every existing statistical test to a fixed seed was split out to Spec 19, since it's a
   large mechanical effort of its own kind, distinct from building the seeding
   infrastructure.)*
3. **02, 03, 06, 18** — the breaking API changes, batched into one 0.3.0 release.
   *(02, 03, 18 implemented. 02's statistics-entry-point-validation half was split out to
   18 — see 02's spec for why: a comparably large, independently-shippable ripple through
   a different module. 18 validates at construction time for `LazyStats`/
   `AdaptiveLazyStats` (sample count fixed once, reused by every accessor) and at call
   time everywhere else (sample count/quantile/confidence/bandwidth are genuine per-call
   parameters) — see 18's spec for the full per-method breakdown.)*
4. **07 → 08** — optimizer correctness before optimizer effectiveness.
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
