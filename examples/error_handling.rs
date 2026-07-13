use uncertain_rs::{Uncertain, UncertainError};

/// Error Handling with Validated Distribution Constructors
///
/// Every distribution constructor validates its parameters and returns
/// `Result<Uncertain<T>, UncertainError>`. This example walks through the three
/// common patterns for handling that: unwrapping known-good literals, matching
/// on specific error variants, and propagating errors with `?` from a
/// config-driven builder.
fn main() {
    println!("⚠️  Error Handling with Validated Constructors");
    println!("==============================================\n");

    known_valid_parameters();
    matching_on_error_variants();

    println!("\n🏗️  Building a sensor model from untrusted config:");
    match build_sensor_model(72.0, 2.5) {
        Ok(sensor) => {
            let mean = sensor.take_samples(1000).iter().sum::<f64>() / 1000.0;
            println!("   ✅ Built sensor model, sampled mean ≈ {mean:.2}");
        }
        Err(e) => println!("   ❌ Failed to build sensor model: {e}"),
    }

    match build_sensor_model(72.0, -2.5) {
        Ok(_) => println!("   ✅ Built sensor model"),
        Err(e) => println!("   ❌ Failed to build sensor model: {e}"),
    }
}

/// When parameters are known-valid compile-time literals, `.unwrap()` is the
/// simplest choice — the same way you'd unwrap a `Vec::first()` you already
/// know is non-empty.
fn known_valid_parameters() {
    println!("1. Known-valid parameters (`.unwrap()`):");
    let speed = Uncertain::normal(55.2, 5.0).unwrap();
    let sample = speed.sample();
    println!("   GPS speed reading: {sample:.1} mph\n");
}

/// When you want to react differently to different failure modes, match on
/// the `UncertainError` variant.
fn matching_on_error_variants() {
    println!("2. Matching on error variants:");

    let attempts: Vec<(f64, f64)> = vec![(0.0, 1.0), (0.0, -1.0), (f64::NAN, 1.0)];

    for (mean, std_dev) in attempts {
        match Uncertain::normal(mean, std_dev) {
            Ok(_) => println!("   normal({mean}, {std_dev}) -> Ok"),
            Err(UncertainError::NonFiniteParameter { parameter, value }) => {
                println!("   normal({mean}, {std_dev}) -> non-finite '{parameter}': {value}");
            }
            Err(UncertainError::InvalidParameter {
                parameter,
                value,
                constraint,
            }) => {
                println!(
                    "   normal({mean}, {std_dev}) -> invalid '{parameter}' = {value} ({constraint})"
                );
            }
            Err(e) => println!("   normal({mean}, {std_dev}) -> {e}"),
        }
    }
}

/// When parameters come from an untrusted source (user input, config files,
/// deserialized data), propagate the error with `?` and let the caller decide
/// how to handle it.
fn build_sensor_model(mean: f64, std_dev: f64) -> Result<Uncertain<f64>, UncertainError> {
    let sensor = Uncertain::normal(mean, std_dev)?;
    Ok(sensor)
}
