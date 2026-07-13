# Spec 04 — Seedable RNG & Reproducibility

**Status:** Pending | **Effort:** High | **Module:** `src/distributions.rs`, `src/uncertain.rs`, all tests

## Context

Every sampler calls thread-local `rand::random()` / `rand::rng()`. There is no seed API
anywhere in the crate, so no computation is reproducible — a serious gap for a
probabilistic library (debugging, papers, CI). All statistical tests rely on loose
tolerances over unseeded draws and are inherently flaky, multiplied by the
stable/beta/nightly CI matrix.

## Scope and Invariants

1. Sampling becomes RNG-injectable. Design: samplers take `&mut dyn RngCore` (or a generic
   `R: Rng`) internally; the public surface gains
   - `Uncertain::sample_with(&mut rng) -> T` and `take_samples_with(&mut rng, n)`;
   - existing `sample()`/`take_samples(n)` keep today's behavior via the thread-local RNG.
2. A seeded context constructor: every distribution constructor gets a deterministic path
   (e.g. builder or `with_seed(seed)` adapter) such that two `Uncertain` values built from
   the same parameters and seed produce identical sample streams.
3. Determinism invariant: same seed + same operation graph + same sample count ⇒ bitwise
   identical `Vec<T>` from `take_samples_with`, across runs and platforms (use a portable
   PRNG, e.g. `ChaCha8`/`Pcg64` via `rand`'s seedable APIs — not `ThreadRng`).
4. The `parallel` feature preserves reproducibility by deriving per-chunk sub-seeds from
   the master seed (results independent of thread count), or documents explicitly that
   parallel sampling has a per-run stream — pick one; the invariant chosen is tested.
5. Test suite deflaking: every statistical assertion in unit/integration tests switches to
   a fixed-seed RNG; tolerance-based assertions remain only where they test genuine
   statistical properties, and then with seeds making them deterministic.
6. No public API is removed; this is additive (breaking only if constructors change shape,
   which belongs to Spec 02's 0.3.0 batch).

## Acceptance Tests

- **Given** two RNGs seeded identically, **when** `normal(0,1)` is sampled 10 000 times via
  `take_samples_with` on each, **then** the two vectors are exactly equal.
- **Given** a composed computation (`(a + b) * c` with comparisons), **when** evaluated
  twice with the same seed, **then** `probability_exceeds`-style results are identical.
- **Given** different seeds, **when** the same distribution is sampled, **then** the
  streams differ (sanity check that seeding is actually wired through).
- **Given** the `parallel` feature with 1, 2, and 8 threads, **when** the chosen
  reproducibility invariant from item 4 applies, **then** it holds and is asserted by
  `tests/parallel_sampling_tests.rs`.
- **Given** the full test suite, **when** run 20 times in a loop (`just test-repeat` or
  equivalent), **then** there are zero statistical flakes.
- **Given** `sample()` with no explicit RNG, **when** called, **then** behavior is
  unchanged from today (thread-local randomness).
