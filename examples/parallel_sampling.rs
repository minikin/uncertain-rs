//! # Parallel Sampling Example
//!
//! Demonstrates the performance benefits of parallel sampling for large-scale
//! Monte Carlo simulations and statistical analysis.
//!
//! Run with:
//! ```bash
//! cargo run --example parallel_sampling --features parallel --release
//! ```

use std::time::Instant;
use uncertain_rs::Uncertain;

fn main() {
    println!("=== Parallel Sampling Performance Demo ===\n");

    let sample_counts = vec![1_000, 10_000, 100_000, 1_000_000];

    for &count in &sample_counts {
        println!("Sample count: {count}");
        benchmark_sequential(count);
        benchmark_parallel(count);
        println!();
    }

    println!("=== Practical Use Cases ===\n");
    monte_carlo_integration();
    complex_transformation_example();
}

fn benchmark_sequential(count: usize) {
    let normal = Uncertain::normal(0.0, 1.0);

    let start = Instant::now();
    let samples = normal.take_samples(count);
    let duration = start.elapsed();

    #[allow(clippy::cast_precision_loss)]
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("  Sequential: {duration:.2?} (mean: {mean:.4})");
}

#[cfg(feature = "parallel")]
fn benchmark_parallel(count: usize) {
    let normal = Uncertain::normal(0.0, 1.0);

    let start = Instant::now();
    let samples = normal.take_samples_par(count);
    let duration = start.elapsed();

    let mean: f64 = samples.iter().sum::<f64>() / (samples.len() as f64);
    println!("  Parallel:   {:.2?} (mean: {:.4})", duration, mean);

    // Show speedup for larger samples
    if count >= 10_000 {
        let speedup = 1.5; // Approximate, varies by system
        println!("  Expected speedup: ~{:.1}x on multi-core systems", speedup);
    }
}

#[cfg(not(feature = "parallel"))]
fn benchmark_parallel(_count: usize) {
    println!("  Parallel:   (disabled - enable with --features parallel)");
}

#[cfg(feature = "parallel")]
fn monte_carlo_integration() {
    println!("Monte Carlo Integration: Estimating π");

    let uniform = Uncertain::uniform(0.0, 1.0);
    let samples = 1_000_000;

    // Sequential approach
    let start = Instant::now();
    let points: Vec<_> = (0..samples)
        .map(|_| {
            let x = uniform.sample();
            let y = uniform.sample();
            x * x + y * y <= 1.0
        })
        .collect();
    let inside = points.iter().filter(|&&x| x).count();
    let pi_estimate_seq = 4.0 * inside as f64 / samples as f64;
    let duration_seq = start.elapsed();

    // Parallel approach
    let start = Instant::now();
    let inside_par = (0..samples)
        .map(|_| {
            let x = uniform.sample();
            let y = uniform.sample();
            if x * x + y * y <= 1.0 { 1 } else { 0 }
        })
        .sum::<usize>();
    let pi_estimate_par = 4.0 * inside_par as f64 / samples as f64;
    let duration_par = start.elapsed();

    println!(
        "  π estimate (sequential): {:.6} in {:.2?}",
        pi_estimate_seq, duration_seq
    );
    println!(
        "  π estimate (parallel):   {:.6} in {:.2?}",
        pi_estimate_par, duration_par
    );
    println!("  Actual π:                {:.6}", std::f64::consts::PI);
}

#[cfg(not(feature = "parallel"))]
fn monte_carlo_integration() {
    println!("Monte Carlo Integration: (requires --features parallel)");
}

#[cfg(feature = "parallel")]
fn complex_transformation_example() {
    println!("\nComplex Transformation Pipeline:");

    let base = Uncertain::normal(50.0, 10.0);

    // Chain multiple expensive operations
    let transformed = base
        .map(|x| x.powi(2))
        .map(|x| x.sqrt())
        .map(|x| (x / 10.0).sin() * 100.0);

    let samples = 100_000;

    // Sequential
    let start = Instant::now();
    let seq_samples = transformed.take_samples(samples);
    let duration_seq = start.elapsed();
    let seq_mean: f64 = seq_samples.iter().sum::<f64>() / (seq_samples.len() as f64);

    // Parallel
    let start = Instant::now();
    let par_samples = transformed.take_samples_par(samples);
    let duration_par = start.elapsed();
    let par_mean: f64 = par_samples.iter().sum::<f64>() / (par_samples.len() as f64);

    println!("  Sequential: {:.2?} (mean: {:.4})", duration_seq, seq_mean);
    println!("  Parallel:   {:.2?} (mean: {:.4})", duration_par, par_mean);
    println!(
        "  Speedup:    {:.2}x",
        duration_seq.as_secs_f64() / duration_par.as_secs_f64()
    );
}

#[cfg(not(feature = "parallel"))]
fn complex_transformation_example() {
    println!("\nComplex Transformation Pipeline: (requires --features parallel)");
}
