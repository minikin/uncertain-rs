# Spec 10 — Bounded Caches & No Silent NaN Paths

**Status:** Pending | **Effort:** Medium | **Module:** `src/cache.rs`, `src/statistics.rs`, `src/hypothesis.rs`, `src/operations/logical.rs`

## Context

Two global `LazyLock` caches (`src/cache.rs:15-18`) keyed by `(uuid, sample_count)` grow
without bound: entries expire only by TTL and only when `cleanup_*` is called manually —
a long-running process creating many distributions leaks memory. Caching *samples* also
changes observable behavior (a cached vector returned later ≠ a fresh draw), forcing tests
to call `clear_global_caches()`. Half the cache API is dead: `StatisticsCache::
get_or_compute_{expected_value,variance,std_dev,confidence_interval,quantile}` have no
callers; only skewness/kurtosis/cdf/pdf_kde actually use the cache. Separately, two silent
NaN paths exist: `probability()` divides by `samples.len()` without an empty guard
(`src/operations/logical.rs:182`), and `bayesian_update` divides by `1.0 − evidence_total`
which can be zero (`src/hypothesis.rs:282`).

## Scope and Invariants

1. Caches get a hard size bound with LRU (or LRU-ish) eviction; eviction happens inline on
   insert — no reliance on manual `cleanup_*` calls. Default capacity is documented and
   configurable.
2. Dead `get_or_compute_*` methods are either wired up (statistics actually consult the
   cache) or deleted — no dead public surface remains. Decision recorded in the spec PR.
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
