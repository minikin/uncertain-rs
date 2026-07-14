//! Integration tests for seeded, reproducible sampling (`sample_with`/`take_samples_with`).
//!
//! These directly exercise the determinism guarantees from
//! `specs/04-seedable-rng.md`.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use uncertain_rs::Uncertain;

#[test]
fn identically_seeded_rngs_produce_identical_sample_streams() {
    let normal = Uncertain::normal(0.0, 1.0).unwrap();

    let mut rng_a = ChaCha8Rng::seed_from_u64(42);
    let mut rng_b = ChaCha8Rng::seed_from_u64(42);

    let samples_a = normal.take_samples_with(&mut rng_a, 10_000);
    let samples_b = normal.take_samples_with(&mut rng_b, 10_000);

    assert_eq!(samples_a, samples_b);
}

#[test]
fn different_seeds_produce_different_sample_streams() {
    let normal = Uncertain::normal(0.0, 1.0).unwrap();

    let mut rng_a = ChaCha8Rng::seed_from_u64(1);
    let mut rng_b = ChaCha8Rng::seed_from_u64(2);

    let samples_a = normal.take_samples_with(&mut rng_a, 1000);
    let samples_b = normal.take_samples_with(&mut rng_b, 1000);

    assert_ne!(samples_a, samples_b);
}

#[test]
fn seeded_sampling_is_deterministic_across_distribution_kinds() {
    // Sanity check across a few constructors, not just normal.
    for build in [
        (|| Uncertain::uniform(0.0, 100.0)) as fn() -> uncertain_rs::Result<Uncertain<f64>>,
        || Uncertain::exponential(2.0),
        || Uncertain::gamma(2.0, 1.0),
        || Uncertain::beta(2.0, 5.0),
    ] {
        let dist = build().unwrap();
        let mut rng_a = ChaCha8Rng::seed_from_u64(7);
        let mut rng_b = ChaCha8Rng::seed_from_u64(7);
        assert_eq!(
            dist.take_samples_with(&mut rng_a, 500),
            dist.take_samples_with(&mut rng_b, 500)
        );
    }
}

#[test]
fn sample_with_advances_rng_state_across_calls() {
    let normal = Uncertain::normal(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);

    let first = normal.sample_with(&mut rng);
    let second = normal.sample_with(&mut rng);

    // Same underlying rng reference, but its state has advanced, so the two draws
    // must (overwhelmingly likely) differ -- this also guards against a broken
    // implementation that resets the rng to the same seed on every call.
    assert_ne!(first, second);

    // Replaying from the same starting seed reproduces the exact same pair.
    let mut replay = ChaCha8Rng::seed_from_u64(5);
    assert_eq!(first, normal.sample_with(&mut replay));
    assert_eq!(second, normal.sample_with(&mut replay));
}

#[test]
fn composed_computation_is_deterministic_under_seeding() {
    // (a + b) * c, then compare against a threshold -- exercises the arithmetic
    // operators and the evidence-based Comparison trait together, since the leaves'
    // sample_fn closures are invoked while `sample_with`'s thread-local override is
    // installed, regardless of how many operators sit on top of them.
    let a = Uncertain::normal(2.0, 0.1).unwrap();
    let b = Uncertain::normal(3.0, 0.1).unwrap();
    let c = Uncertain::normal(1.0, 0.1).unwrap();
    let expr = (a + b) * c;
    let evidence = expr.gt(4.0);

    let mut rng_a = ChaCha8Rng::seed_from_u64(123);
    let mut rng_b = ChaCha8Rng::seed_from_u64(123);

    let samples_a = evidence.take_samples_with(&mut rng_a, 2000);
    let samples_b = evidence.take_samples_with(&mut rng_b, 2000);

    assert_eq!(samples_a, samples_b);
}

#[test]
fn sample_without_explicit_rng_is_unaffected_by_seeding_infrastructure() {
    // sample()/take_samples() with no explicit RNG must keep drawing from real
    // thread-local randomness -- unseeded, and not accidentally deterministic because
    // some global override leaked from a seeded call elsewhere.
    let normal = Uncertain::normal(0.0, 1.0).unwrap();

    // Use the seeded API once, to make sure it doesn't leave a stuck override behind.
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let _ = normal.take_samples_with(&mut rng, 10);

    let run_a = normal.take_samples(2000);
    let run_b = normal.take_samples(2000);
    assert_ne!(
        run_a, run_b,
        "unseeded take_samples() must not become deterministic"
    );
}

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;

    #[test]
    fn take_samples_with_par_is_reproducible() {
        let normal = Uncertain::normal(0.0, 1.0).unwrap();
        let a = normal.take_samples_with_par(99, 5000);
        let b = normal.take_samples_with_par(99, 5000);
        assert_eq!(a, b);
    }

    #[test]
    fn take_samples_with_par_is_independent_of_thread_count() {
        let normal = Uncertain::normal(0.0, 1.0).unwrap();

        let build_pool = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("failed to build a rayon thread pool")
        };

        let baseline = build_pool(1).install(|| normal.take_samples_with_par(2024, 4000));
        for threads in [2, 8] {
            let result = build_pool(threads).install(|| normal.take_samples_with_par(2024, 4000));
            assert_eq!(
                baseline, result,
                "take_samples_with_par must be identical regardless of thread count (got a mismatch at {threads} threads)"
            );
        }
    }

    #[test]
    fn take_samples_with_par_differs_from_different_seeds() {
        let normal = Uncertain::normal(0.0, 1.0).unwrap();
        let a = normal.take_samples_with_par(1, 2000);
        let b = normal.take_samples_with_par(2, 2000);
        assert_ne!(a, b);
    }
}
