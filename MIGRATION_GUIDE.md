# Migration Guide: 0.2.x → 0.3.0

## Distribution constructors now return `Result`

Every distribution constructor validates its parameters and returns
`Result<Uncertain<T>, UncertainError>` instead of constructing an `Uncertain<T>`
infallibly. This affects:

`normal`, `uniform`, `exponential`, `log_normal`, `beta`, `gamma`, `bernoulli`,
`binomial`, `poisson`, `geometric`.

(`mixture`, `empirical`, `categorical`, and `point` were already `Result`-returning or
infallible-by-design in 0.2.x and are unchanged.)

### Why

Before 0.3.0, invalid parameters were silently accepted and produced garbage or `NaN`:

```rust
// 0.2.x — compiles, but silently broken
let dist = Uncertain::normal(0.0, -1.0);   // negative std_dev: undefined behavior
let dist = Uncertain::exponential(0.0);    // rate = 0: every sample is `inf`
```

0.3.0 catches these at construction time instead of at some unpredictable point deep in a
sampling loop.

### How to update

Add `.unwrap()` if you already know the parameters are valid (e.g. compile-time
constants), or handle the `Result` with `?` / a `match` if the parameters come from
user input, config, or any other untrusted source:

```rust
// Before (0.2.x)
let speed = Uncertain::normal(55.2, 5.0);

// After (0.3.0) — known-valid parameters
let speed = Uncertain::normal(55.2, 5.0).unwrap();

// After (0.3.0) — parameters from an untrusted source
fn build_sensor_model(mean: f64, std_dev: f64) -> Result<Uncertain<f64>, UncertainError> {
    let speed = Uncertain::normal(mean, std_dev)?;
    Ok(speed)
}
```

See [`examples/error_handling.rs`](examples/error_handling.rs) for a complete walkthrough
of validation, error matching, and propagation with `?`.

### Validation rules

| Constructor | Constraint | `0`/boundary behavior |
|---|---|---|
| `normal(mean, std_dev)` | both finite; `std_dev >= 0` | `std_dev = 0` → point mass at `mean` |
| `uniform(min, max)` | both finite; `min <= max` | `min == max` → point mass |
| `exponential(rate)` | finite; `rate > 0` | `rate = 0` → **rejected** (would yield `inf`) |
| `log_normal(mu, sigma)` | both finite; `sigma >= 0` | `sigma = 0` → point mass at `exp(mu)` |
| `beta(alpha, beta)` | both finite; `> 0` | `0` → **rejected** (undefined) |
| `gamma(shape, scale)` | both finite; `shape > 0`, `scale >= 0` | `scale = 0` → point mass at `0` |
| `bernoulli(p)` | finite; `p ∈ [0, 1]` | inclusive bounds allowed |
| `binomial(trials, p)` | `p` finite; `p ∈ [0, 1]` | inclusive bounds allowed |
| `poisson(lambda)` | finite; `lambda >= 0` | `lambda = 0` → point mass at `0` |
| `geometric(p)` | finite; `p ∈ (0, 1]` | `p = 0` → **rejected** (would never terminate) |

A rejected parameter returns `UncertainError::NonFiniteParameter { parameter, value }` (NaN
or infinite input) or `UncertainError::InvalidParameter { parameter, value, constraint }`
(finite but out of range) — never a panic.

### Not affected by this release

Statistics methods (`quantile`, `confidence_interval`, `pdf_kde`, and anything taking a
`sample_count`) are unchanged in 0.3.0 — that validation is tracked separately (see
`specs/18-statistics-validation.md` in the repo) and will be called out in its own
CHANGELOG entry when it ships.
