//! Deterministic-sampling support.
//!
//! Every sampling closure in the crate has type `Fn() -> T` — no RNG parameter.
//! Rather than rewrite that signature everywhere (every combinator, every operator
//! overload, the whole computation graph), distribution constructors draw randomness
//! through [`with_rng`] instead of calling `rand::random()`/`rand::rng()` directly.
//! [`Uncertain::sample_with`](crate::Uncertain::sample_with) installs a caller-provided
//! seeded RNG as a thread-local override for the dynamic extent of one call, so those
//! closures become deterministic without changing their shape.
//!
//! The override is a concrete [`ChaCha8Rng`], not a generic `R: Rng`/`Box<dyn Rng>`:
//! storing a boxed trait object in `thread_local!` and installing a caller's `&mut R`
//! into it for a bounded scope would need either a `'static` bound (forcing an owned
//! copy anyway) or unsafe lifetime erasure. Since `ChaCha8Rng` is `Clone`, installing a
//! clone of the caller's state and writing the advanced clone back after the call is
//! both simpler and unsafe-code-free, at the cost of fixing the RNG type.
//!
//! `ChaCha8Rng` (from `rand_chacha`, not `rand::rngs::StdRng`) is used specifically
//! because it's a documented, version-stable, portable generator — `StdRng` is
//! explicitly documented as non-portable and subject to change between `rand`
//! versions, which would break the "same seed produces the same stream across runs"
//! guarantee this module exists to provide.

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;

thread_local! {
    static OVERRIDE: RefCell<Option<ChaCha8Rng>> = const { RefCell::new(None) };
}

/// Installs `rng` as the thread-local override for the duration of `f`, restoring
/// whatever was installed before (including `None`) once `f` returns or panics.
/// Returns `f`'s result along with the installed RNG's state after `f` ran, so the
/// caller can advance their own copy.
pub(crate) fn with_override<T>(rng: ChaCha8Rng, f: impl FnOnce() -> T) -> (T, ChaCha8Rng) {
    struct RestoreGuard(Option<ChaCha8Rng>);
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            OVERRIDE.with(|cell| *cell.borrow_mut() = self.0.take());
        }
    }

    let previous = OVERRIDE.with(|cell| cell.borrow_mut().replace(rng));
    let _guard = RestoreGuard(previous);
    let result = f();
    let advanced = OVERRIDE.with(|cell| {
        cell.borrow_mut()
            .take()
            .expect("OVERRIDE is Some for the duration of with_override")
    });
    (result, advanced)
}

/// Gives `f` access to the thread-local override RNG if one is installed (i.e. this
/// call is happening inside a [`with_override`] scope), otherwise a fresh handle to the
/// real thread-local RNG (`rand::rng()`) — the same source `rand::random()` uses.
pub(crate) fn with_rng<T>(f: impl FnOnce(&mut dyn Rng) -> T) -> T {
    OVERRIDE.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_mut() {
            Some(rng) => f(rng),
            None => {
                drop(guard);
                let mut rng = rand::rng();
                f(&mut rng)
            }
        }
    })
}

/// Convenience wrapper for the common case of drawing a single `f64` in `[0, 1)`.
pub(crate) fn random_f64() -> f64 {
    with_rng(|rng| rng.random::<f64>())
}

/// Deterministically derives a sub-seed for parallel sampling: sample index `i` from a
/// `take_samples_with_par` call with master seed `seed` always gets this same sub-seed,
/// regardless of thread count or scheduling. `splitmix64`-style mixing avoids the
/// correlated streams a bare `seed ^ i` or `seed + i` could produce between adjacent
/// indices.
pub(crate) fn derive_sub_seed(seed: u64, index: u64) -> u64 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(crate) fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_rng_falls_back_to_thread_local_when_no_override_installed() {
        // Just needs to not panic and to produce a value in range; there's no override
        // installed here, so this exercises the fallback path.
        let value = random_f64();
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn with_override_makes_with_rng_deterministic() {
        let rng_a = seeded_rng(42);
        let (values_a, _) =
            with_override(rng_a, || (0..5).map(|_| random_f64()).collect::<Vec<_>>());

        let rng_b = seeded_rng(42);
        let (values_b, _) =
            with_override(rng_b, || (0..5).map(|_| random_f64()).collect::<Vec<_>>());

        assert_eq!(values_a, values_b);
    }

    #[test]
    fn with_override_different_seeds_differ() {
        let rng_a = seeded_rng(1);
        let (values_a, _) =
            with_override(rng_a, || (0..5).map(|_| random_f64()).collect::<Vec<_>>());

        let rng_b = seeded_rng(2);
        let (values_b, _) =
            with_override(rng_b, || (0..5).map(|_| random_f64()).collect::<Vec<_>>());

        assert_ne!(values_a, values_b);
    }

    #[test]
    fn with_override_advances_and_restores_state() {
        let mut rng = seeded_rng(7);
        let (first, advanced_once) = with_override(rng.clone(), random_f64);
        rng = advanced_once;
        let (second, _) = with_override(rng, random_f64);

        // Continuing from the advanced state must not repeat the first draw.
        assert_ne!(first, second);
    }

    #[test]
    fn with_override_restores_previous_state_on_panic() {
        let rng = seeded_rng(99);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_override(rng, || panic!("boom"));
        }));
        assert!(result.is_err());

        // No override should be left installed after the panic unwound.
        let value = random_f64();
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn nested_with_override_restores_outer_scope() {
        let outer = seeded_rng(10);
        let ((), _outer_advanced) = with_override(outer, || {
            let outer_first = random_f64();

            let inner = seeded_rng(20);
            let (inner_value, _) = with_override(inner, random_f64);

            let outer_second = random_f64();
            assert_ne!(
                outer_first, outer_second,
                "outer RNG should still be advancing"
            );
            let _ = inner_value;
        });
    }

    #[test]
    fn derive_sub_seed_is_deterministic_and_index_sensitive() {
        assert_eq!(derive_sub_seed(42, 0), derive_sub_seed(42, 0));
        assert_ne!(derive_sub_seed(42, 0), derive_sub_seed(42, 1));
        assert_ne!(derive_sub_seed(42, 0), derive_sub_seed(43, 0));
    }
}
