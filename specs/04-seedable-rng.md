# Spec 04 — Seedable RNG & Reproducibility

**Status:** Implemented | **Effort:** High | **Module:** `src/rng.rs` (new), `src/distributions.rs`, `src/uncertain.rs`

## Context

Every sampler called thread-local `rand::random()` / `rand::rng()`. There was no seed API
anywhere in the crate, so no computation was reproducible — a serious gap for a
probabilistic library (debugging, papers, CI). All statistical tests relied on loose
tolerances over unseeded draws and were inherently flaky, multiplied by the
stable/beta/nightly CI matrix.

## Scope and Invariants

1. **Deviation from the original draft's mechanism** — the draft's item 1 proposed
   changing sampling closures to take `&mut dyn RngCore`/`R: Rng` as an explicit
   parameter. `Uncertain<T>`'s `sample_fn: Arc<dyn Fn() -> T + Send + Sync>` (and every
   combinator — `map`, `flat_map`, `filter` — every arithmetic/comparison/logical operator
   overload, and the whole `ComputationNode` evaluation engine) is built around that
   zero-argument shape; threading an RNG parameter through all of it would mean rewriting
   the crate's core representation, a far larger and riskier change than "add seeding."
   Instead: `src/rng.rs` provides a thread-local RNG **override** — `with_override`
   installs a caller-provided RNG as the thread-local source for the dynamic extent of one
   call (via a panic-safe `Drop` guard), and `with_rng`/`random_f64` (used by every
   distribution constructor in place of `rand::random()`/`rand::rng()`) transparently draw
   from that override when one is installed, falling back to real thread-local randomness
   otherwise. No existing closure signature changed. The public surface still gained
   exactly what item 1 asked for:
   - `Uncertain::sample_with(&mut ChaCha8Rng) -> T` and
     `take_samples_with(&mut ChaCha8Rng, n) -> Vec<T>`;
   - `sample()`/`take_samples(n)` are byte-for-byte unchanged (still thread-local).
2. **Deviation:** no separate "seeded context constructor" was added. Any existing
   `Uncertain<T>` value — built however — becomes reproducible simply by sampling it via
   `sample_with`/`take_samples_with` instead of `sample`/`take_samples`; there's no
   separate construction-time seeding step to design or maintain.
3. **Deviation from the original draft's RNG choice** — the draft suggested
   `ChaCha8`/`Pcg64` via `rand`'s own seedable APIs (i.e. `rand::rngs::StdRng`). Checked
   at implementation time: this `rand` version's own docs state `StdRng` is **"non-portable:
   any future library version may replace the algorithm and results may be
   platform-dependent... even with a fixed seed, output is not portable"** — directly
   contradicting the determinism invariant this spec exists to provide. Used
   `rand_chacha::ChaCha8Rng` directly instead (new dependency) — the same family the draft
   named, but the actual portable/version-stable implementation, not the generic
   alias. `ChaCha8Rng: Clone` lets `sample_with`/`take_samples_with` install a clone of the
   caller's state into the override and write the advanced clone back afterward — this is
   what makes `with_override` unsafe-code-free (no `Box<dyn Rng>` + lifetime erasure
   needed), at the cost of fixing the RNG type rather than being generic over `R: Rng`.
4. `take_samples_with_par(seed: u64, count: usize)` (parallel feature): sample index `i`
   derives its own sub-seed from `seed` via `splitmix64`-style mixing
   (`rng::derive_sub_seed`), then seeds a fresh `ChaCha8Rng` per index — independent of
   which thread computes it or how work is chunked, giving reproducibility independent of
   thread count (the stronger of the draft's two allowed choices, not just "documented as
   per-run").
5. **Scope reduction:** retrofitting every existing statistical test in the crate to a
   fixed seed (the draft's item 5) is large enough on its own — hundreds of existing
   tests, a mechanical but sizable effort of its own kind — to be its own spec rather than
   silently folded into this one. Split out as
   [Spec 19](19-deflake-existing-tests.md). This spec instead ships a focused new test
   suite (`tests/seeded_rng_tests.rs`) that directly proves every acceptance test below.
6. No public API removed; fully additive (a new dependency, new methods).

## Acceptance Tests

- **Given** two RNGs seeded identically, **when** `normal(0,1)` is sampled 10 000 times via
  `take_samples_with` on each, **then** the two vectors are exactly equal. ✅
  (`identically_seeded_rngs_produce_identical_sample_streams`)
- **Given** a composed computation (`(a + b) * c` with comparisons), **when** evaluated
  twice with the same seed, **then** results are identical. ✅
  (`composed_computation_is_deterministic_under_seeding`)
- **Given** different seeds, **when** the same distribution is sampled, **then** the
  streams differ. ✅ (`different_seeds_produce_different_sample_streams`)
- **Given** the `parallel` feature with 1, 2, and 8 threads, **when**
  `take_samples_with_par` runs with the same seed, **then** the result is identical
  regardless of thread count. ✅ (`take_samples_with_par_is_independent_of_thread_count`,
  using explicit `rayon::ThreadPoolBuilder` pools)
- **Given** the new seeded-RNG test suite, **when** run repeatedly, **then** there are zero
  flakes (verified locally: 5 consecutive full runs, default and `parallel` features). Full
  crate-wide deflaking is Spec 19, not this spec.
- **Given** `sample()` with no explicit RNG, **when** called, **then** behavior is
  unchanged from today (thread-local randomness) — and using the seeded API earlier in a
  test doesn't leave a stuck override behind. ✅
  (`sample_without_explicit_rng_is_unaffected_by_seeding_infrastructure`)
