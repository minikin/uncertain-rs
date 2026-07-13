# Spec 10 — Bounded Caches & No Silent NaN Paths

**Status:** Partially implemented | **Effort:** Medium | **Module:** `src/cache.rs`, `src/statistics.rs`, `src/hypothesis.rs`, `src/operations/logical.rs`

## Context

Two global `LazyLock` caches (`src/cache.rs:15-18`) keyed by `(uuid, sample_count)` grow
without bound: entries expire only by TTL and only when `cleanup_*` is called manually —
a long-running process creating many distributions leaks memory. Caching *samples* also
changes observable behavior (a cached vector returned later ≠ a fresh draw), forcing tests
to call `clear_global_caches()`.

**Item 2 partially done** (landed opportunistically on the `spec/02-validated-constructors`
branch, PR #17, while fixing a genuinely broken test —
`test_different_precision_cache_hits` failed even on plain `main` because
`Uncertain::confidence_interval` never consulted the cache it was testing): `expected_value`,
`variance`, `confidence_interval`, and `quantile` on `Uncertain<T>` now call
`cache::stats_cache().get_or_compute_*`, matching the pattern `skewness`/`kurtosis`/`cdf`/
`pdf_kde` already used. `get_or_compute_std_dev` had no caller anywhere (not even in
`standard_deviation`, which derives from the now-cached `variance` and doesn't need its own
entry) and was deleted rather than wired up. Remaining for this spec: the bounded/LRU
eviction (item 1) and the NaN-path guards (items 5–6) below are still open — two silent
NaN paths exist: `probability()` divides by `samples.len()` without an empty guard
(`src/operations/logical.rs:182`), and `bayesian_update` divides by `1.0 − evidence_total`
which can be zero (`src/hypothesis.rs:282`).

## Scope and Invariants

1. Caches get a hard size bound with LRU (or LRU-ish) eviction; eviction happens inline on
   insert — no reliance on manual `cleanup_*` calls. Default capacity is documented and
   configurable.
2. ~~Dead `get_or_compute_*` methods are either wired up (statistics actually consult the
   cache) or deleted — no dead public surface remains.~~ **Done** (see Context): all five
   originally-dead methods are now either wired up (`expected_value`, `variance`,
   `confidence_interval`, `quantile`) or deleted (`std_dev`).
3. Cache policy is documented: what is cached, key, TTL, capacity, and the implication
   that cached sample vectors are reused (or sample-caching is dropped in favor of caching
   derived statistics only — preferred, since it removes the behavioral surprise; pick one
   and document).
4. A scoped/injectable cache (per-`Uncertain` or context object) is preferred over global
   state where feasible without breaking the API; if globals remain, `clear_global_caches`
   stays and its role in tests is documented.
5. `probability()` with zero samples returns the Spec 02 `InvalidSampleCount` error or a
   documented value — never NaN.
6. `bayesian_update` guards `evidence_total ∈ {0, 1}` degenerate cases with a typed error
   or documented clamp — never NaN.
7. Thread-safety unchanged: caches remain `Send + Sync`; a concurrency smoke test hits one
   cache from many threads under `--features parallel`.

## Acceptance Tests

- **Given** a cache with capacity N, **when** N+K distinct entries are inserted, **then**
  memory-resident entries never exceed N and the K oldest/least-recent are gone — without
  any manual cleanup call.
- **Given** a long loop creating 100 000 distinct distributions and computing a cached
  statistic on each, **when** run, **then** process memory plateaus (bounded cache) rather
  than growing linearly.
- **Given** the crate source, **when** grepped, **then** every public cache method has a
  caller in library code or has been removed.
- **Given** `probability()` on an evidence value with `sample_count = 0`, **then** the
  documented non-NaN behavior occurs.
- **Given** `bayesian_update` where evidence probabilities sum to 1.0 exactly, **then**
  the result contains no NaN.
- **Given** 16 threads concurrently reading/writing one cache, **when** the smoke test
  runs under `--features parallel`, **then** no deadlock, no panic, and all results are
  consistent.
