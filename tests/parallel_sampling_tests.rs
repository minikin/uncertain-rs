//! Integration tests for parallel sampling functionality
//!
//! Tests the correctness and performance characteristics of parallel sampling.

#![cfg(feature = "parallel")]

use uncertain_rs::Uncertain;

#[test]
fn test_parallel_sampling_produces_valid_results() {
    let normal = Uncertain::normal(0.0, 1.0);
    let samples = normal.take_samples_par(10_000);

    assert_eq!(samples.len(), 10_000);

    for &sample in &samples {
        assert!(sample.abs() < 6.0, "Sample {} is outside 6 sigma", sample);
    }

    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(mean.abs() < 0.1, "Mean {} is too far from 0", mean);

    let variance: f64 =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

    assert!(
        (variance - 1.0).abs() < 0.1,
        "Variance {} is too far from 1",
        variance
    );
}

#[test]
fn test_parallel_vs_sequential_statistical_consistency() {
    let normal = Uncertain::normal(5.0, 2.0);
    let count = 50_000;

    let seq_samples = normal.take_samples(count);
    let par_samples = normal.take_samples_par(count);

    // Both should have the same length
    assert_eq!(seq_samples.len(), count);
    assert_eq!(par_samples.len(), count);

    let seq_mean: f64 = seq_samples.iter().sum::<f64>() / count as f64;
    let par_mean: f64 = par_samples.iter().sum::<f64>() / count as f64;

    // Calculate standard deviations
    let seq_variance: f64 = seq_samples
        .iter()
        .map(|&x| (x - seq_mean).powi(2))
        .sum::<f64>()
        / count as f64;
    let seq_std = seq_variance.sqrt();

    let par_variance: f64 = par_samples
        .iter()
        .map(|&x| (x - par_mean).powi(2))
        .sum::<f64>()
        / count as f64;

    let par_std = par_variance.sqrt();

    assert!(
        (seq_mean - 5.0).abs() < 0.1,
        "Sequential mean {} too far from 5.0",
        seq_mean
    );

    assert!(
        (par_mean - 5.0).abs() < 0.1,
        "Parallel mean {} too far from 5.0",
        par_mean
    );

    assert!(
        (seq_std - 2.0).abs() < 0.1,
        "Sequential std {} too far from 2.0",
        seq_std
    );
    assert!(
        (par_std - 2.0).abs() < 0.1,
        "Parallel std {} too far from 2.0",
        par_std
    );

    assert!(
        (seq_mean - par_mean).abs() < 0.1,
        "Sequential and parallel means differ too much: {} vs {}",
        seq_mean,
        par_mean
    );
}

#[test]
fn test_parallel_sampling_with_transformations() {
    let base = Uncertain::normal(10.0, 2.0);
    let transformed = base.map(|x| x * 2.0 + 5.0);

    let samples = transformed.take_samples_par(10_000);

    // Expected mean: 10 * 2 + 5 = 25
    // Expected std: 2 * 2 = 4
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance: f64 =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let std = variance.sqrt();

    assert!(
        (mean - 25.0).abs() < 0.2,
        "Transformed mean {} too far from 25.0",
        mean
    );

    assert!(
        (std - 4.0).abs() < 0.2,
        "Transformed std {} too far from 4.0",
        std
    );
}

#[test]
fn test_parallel_cached_sampling() {
    let gamma = Uncertain::gamma(2.0, 1.0);
    let count = 10_000;

    let samples1 = gamma.take_samples_cached_par(count);
    assert_eq!(samples1.len(), count);

    let samples2 = gamma.take_samples_cached_par(count);
    assert_eq!(samples2.len(), count);

    assert_eq!(samples1, samples2, "Cached samples should be identical");

    let mean: f64 = samples1.iter().sum::<f64>() / count as f64;

    // Gamma(2, 1) has mean = shape * scale = 2 * 1 = 2
    assert!(
        (mean - 2.0).abs() < 0.1,
        "Gamma mean {} too far from 2.0",
        mean
    );
}

#[test]
fn test_parallel_sampling_different_distributions() {
    let count = 5_000;

    let uniform = Uncertain::uniform(0.0, 10.0);
    let uniform_samples = uniform.take_samples_par(count);
    let uniform_mean: f64 = uniform_samples.iter().sum::<f64>() / count as f64;
    assert!(
        (uniform_mean - 5.0).abs() < 0.2,
        "Uniform mean {} too far from 5.0",
        uniform_mean
    );

    let exponential = Uncertain::exponential(2.0);
    let exp_samples = exponential.take_samples_par(count);
    let exp_mean: f64 = exp_samples.iter().sum::<f64>() / count as f64;

    // Exponential mean = 1/lambda = 1/2 = 0.5
    assert!(
        (exp_mean - 0.5).abs() < 0.1,
        "Exponential mean {} too far from 0.5",
        exp_mean
    );

    // Beta distribution
    let beta = Uncertain::beta(2.0, 5.0);
    let beta_samples = beta.take_samples_par(count);
    let beta_mean: f64 = beta_samples.iter().sum::<f64>() / count as f64;
    // Beta mean = alpha / (alpha + beta) = 2 / 7 ≈ 0.286
    assert!(
        (beta_mean - 0.286).abs() < 0.05,
        "Beta mean {} too far from 0.286",
        beta_mean
    );
}

#[test]
fn test_parallel_sampling_empty_and_edge_cases() {
    let normal = Uncertain::normal(0.0, 1.0);

    let empty = normal.take_samples_par(0);
    assert_eq!(empty.len(), 0);

    let single = normal.take_samples_par(1);
    assert_eq!(single.len(), 1);

    let small = normal.take_samples_par(10);
    assert_eq!(small.len(), 10);
}

#[test]
fn test_parallel_sampling_with_filter() {
    let normal = Uncertain::normal(0.0, 1.0);
    let positive_only = normal.filter(|&x| x > 0.0);

    let samples = positive_only.take_samples_par(1_000);

    // All samples should be positive
    assert!(samples.iter().all(|&x| x > 0.0));
    assert_eq!(samples.len(), 1_000);
}

#[test]
fn test_parallel_sampling_reproducibility_not_required() {
    let normal = Uncertain::normal(0.0, 1.0);

    let samples1 = normal.take_samples_par(1_000);
    let samples2 = normal.take_samples_par(1_000);

    // Samples are unlikely to be identical (non-deterministic ordering)
    // But both should have valid statistical properties
    let mean1: f64 = samples1.iter().sum::<f64>() / 1_000.0;
    let mean2: f64 = samples2.iter().sum::<f64>() / 1_000.0;

    assert!((mean1 - 0.0).abs() < 0.2);
    assert!((mean2 - 0.0).abs() < 0.2);
}
