# uncertain-rs — Specifications

Spec-driven development, modeled after [cargo-crap](https://github.com/minikin/cargo-crap):
each spec is acceptance criteria ("specification as law"). **Never modify a spec file
without explicit permission from the maintainer.** Implementation is done spec-by-spec;
a spec is closed only when every acceptance test passes and `just dev` is green.

## Index

| #  | Spec | Priority | Effort | Breaking | Status |
|----|------|----------|--------|----------|--------|
| 01 | [Dev-workflow hardening](01-dev-workflow-hardening.md) | P0 | Medium | no | Pending |
| 02 | [Validated constructors](02-validated-constructors.md) | P0 | High | **yes** | Pending |
| 03 | [Remove sampling-based Eq/Ord](03-remove-sampling-eq.md) | P0 | Low | **yes** | Pending |
| 04 | [Seedable RNG & reproducibility](04-seedable-rng.md) | P0 | High | no | Pending |
| 05 | [Correct sampling via rand_distr](05-rand-distr-sampling.md) | P0 | Medium | no | Pending |
| 06 | [Total graph evaluation](06-total-graph-evaluation.md) | P0 | Medium | **yes** | Pending |
| 07 | [Sound constant folding](07-sound-constant-folding.md) | P0 | Medium | no | Pending |
| 08 | [Effective CSE](08-effective-cse.md) | P1 | Medium | no | Pending |
| 09 | [Consistent variance policy](09-variance-policy.md) | P1 | Low | behavioral | Pending |
| 10 | [Bounded caches, no NaN paths](10-bounded-caches.md) | P1 | Medium | no | Pending |
| 11 | [Property-based testing](11-property-based-testing.md) | P1 | Medium | no | Pending |
| 12 | [Serde support](12-serde-support.md) | P2 | Low | no | Pending |
| 13 | [More distributions](13-more-distributions.md) | P2 | Medium | no | Pending |
| 14 | [Benchmarks & coverage gates in CI](14-bench-regression-ci.md) | P1 | Medium | no | Pending |
| 15 | [Docs & meta cleanup](15-docs-and-meta.md) | P1 | Low | no | Pending |
| 16 | [Release automation](16-release-automation.md) | P2 | Low | no | Pending |

## Suggested order

1. **01** first — it installs the quality gates every later spec is verified against.
2. **04 → 05** (seeding, then correct samplers) — reproducibility unblocks deflaked tests
   used by every other spec.
3. **02, 03, 06** — the breaking API changes, batched into one 0.3.0 release.
4. **07 → 08** — optimizer correctness before optimizer effectiveness.
5. **09, 10, 11, 14, 15** in any order.
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
