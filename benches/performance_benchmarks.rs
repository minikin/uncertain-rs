use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use uncertain_rs::{Uncertain, cache};

fn benchmark_statistical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_operations");
    group.measurement_time(Duration::from_secs(10));

    let normal = Uncertain::normal(0.0, 1.0);
    let sample_count = 10000;

    group.bench_function("expected_value_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| black_box(dist.expected_value(sample_count)),
        );
    });

    group.bench_function("expected_value_cached", |b| {
        let _ = normal.expected_value(sample_count);
        b.iter(|| black_box(normal.expected_value(sample_count)));
    });

    group.bench_function("variance_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| black_box(dist.variance(sample_count)),
        );
    });

    group.bench_function("variance_cached", |b| {
        let _ = normal.variance(sample_count);
        b.iter(|| black_box(normal.variance(sample_count)));
    });

    group.bench_function("std_dev_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| black_box(dist.standard_deviation(sample_count)),
        );
    });

    group.bench_function("std_dev_cached", |b| {
        let _ = normal.standard_deviation(sample_count);
        b.iter(|| black_box(normal.standard_deviation(sample_count)));
    });

    group.bench_function("skewness_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| black_box(dist.skewness(sample_count)),
        );
    });

    group.bench_function("skewness_cached", |b| {
        let _ = normal.skewness(sample_count);
        b.iter(|| black_box(normal.skewness(sample_count)));
    });

    group.bench_function("kurtosis_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| black_box(dist.kurtosis(sample_count)),
        );
    });

    group.bench_function("kurtosis_cached", |b| {
        // Prime the cache
        let _ = normal.kurtosis(sample_count);
        b.iter(|| black_box(normal.kurtosis(sample_count)));
    });

    group.finish();
}

fn benchmark_interval_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval_operations");
    group.measurement_time(Duration::from_secs(8));

    let normal = Uncertain::normal(100.0, 15.0);
    let sample_count = 5000;

    group.bench_function("confidence_interval_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(100.0, 15.0)
            },
            |dist| black_box(dist.confidence_interval(0.95, sample_count)),
        );
    });

    group.bench_function("confidence_interval_cached", |b| {
        let _ = normal.confidence_interval(0.95, sample_count);
        b.iter(|| black_box(normal.confidence_interval(0.95, sample_count)));
    });

    group.bench_function("cdf_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(100.0, 15.0)
            },
            |dist| black_box(dist.cdf(100.0, sample_count)),
        );
    });

    group.bench_function("cdf_cached", |b| {
        let _ = normal.cdf(100.0, sample_count);
        b.iter(|| black_box(normal.cdf(100.0, sample_count)));
    });

    group.bench_function("quantile_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(100.0, 15.0)
            },
            |dist| black_box(dist.quantile(0.5, sample_count)),
        );
    });

    group.bench_function("quantile_cached", |b| {
        // Prime the cache
        let _ = normal.quantile(0.5, sample_count);
        b.iter(|| black_box(normal.quantile(0.5, sample_count)));
    });

    group.finish();
}

fn benchmark_pdf_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pdf_operations");
    group.measurement_time(Duration::from_secs(15));

    let normal = Uncertain::normal(0.0, 1.0);
    let sample_count = 2000; // Smaller sample count for expensive operation
    let bandwidth = 0.1;

    group.bench_function("pdf_kde_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| black_box(dist.pdf_kde(0.0, sample_count, bandwidth)),
        );
    });

    group.bench_function("pdf_kde_cached", |b| {
        let _ = normal.pdf_kde(0.0, sample_count, bandwidth);
        b.iter(|| black_box(normal.pdf_kde(0.0, sample_count, bandwidth)));
    });

    group.finish();
}

fn benchmark_distribution_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution_sampling");
    group.measurement_time(Duration::from_secs(10));

    let sample_count = 1000;

    let gamma = Uncertain::gamma(2.0, 1.0);

    group.bench_function("gamma_samples_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::gamma(2.0, 1.0)
            },
            |dist| black_box(dist.take_samples_cached(sample_count)),
        );
    });

    group.bench_function("gamma_samples_cached", |b| {
        let _ = gamma.take_samples_cached(sample_count);
        b.iter(|| black_box(gamma.take_samples_cached(sample_count)));
    });

    let beta = Uncertain::beta(2.0, 5.0);

    group.bench_function("beta_samples_first_run", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::beta(2.0, 5.0)
            },
            |dist| black_box(dist.take_samples_cached(sample_count)),
        );
    });

    group.bench_function("beta_samples_cached", |b| {
        let _ = beta.take_samples_cached(sample_count);
        b.iter(|| black_box(beta.take_samples_cached(sample_count)));
    });

    group.finish();
}

fn benchmark_computation_graphs(c: &mut Criterion) {
    let mut group = c.benchmark_group("computation_graphs");
    group.measurement_time(Duration::from_secs(8));

    let sample_count = 1000;

    // Complex expression: (x + y) * (x - y) where x and y are uncertain
    let x = Uncertain::normal(5.0, 1.0);
    let y = Uncertain::normal(3.0, 1.0);
    let complex_expr = (x.clone() + y.clone()) * (x - y);

    group.bench_function("complex_expression_expected_value", |b| {
        b.iter(|| black_box(complex_expr.expected_value(sample_count)));
    });

    group.bench_function("complex_expression_variance", |b| {
        b.iter(|| black_box(complex_expr.variance(sample_count)));
    });

    group.bench_function("multiple_stats_operations", |b| {
        b.iter(|| {
            let _ = black_box(complex_expr.expected_value(sample_count));
            let _ = black_box(complex_expr.variance(sample_count));
            let _ = black_box(complex_expr.standard_deviation(sample_count));
            black_box(complex_expr.skewness(sample_count))
        });
    });

    group.finish();
}

fn benchmark_cache_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_overhead");

    let normal = Uncertain::normal(0.0, 1.0);

    // Small sample counts where caching might not be beneficial
    group.bench_function("small_samples_no_cache", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| {
                let samples: Vec<f64> = dist.take_samples(100);
                black_box(samples.iter().sum::<f64>() / 100.0)
            },
        );
    });

    group.bench_function("small_samples_with_cache", |b| {
        b.iter(|| {
            let samples = normal.take_samples_cached(100);
            black_box(samples.iter().sum::<f64>() / 100.0)
        });
    });

    group.bench_function("large_samples_no_cache", |b| {
        b.iter_with_setup(
            || {
                cache::clear_global_caches();
                Uncertain::normal(0.0, 1.0)
            },
            |dist| {
                let samples: Vec<f64> = dist.take_samples(10000);
                black_box(samples.iter().sum::<f64>() / 10000.0)
            },
        );
    });

    group.bench_function("large_samples_with_cache", |b| {
        b.iter(|| {
            let samples = normal.take_samples_cached(10000);
            black_box(samples.iter().sum::<f64>() / 10000.0)
        });
    });

    group.finish();
}

#[cfg(feature = "parallel")]
fn benchmark_parallel_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_sampling");
    group.measurement_time(Duration::from_secs(10));

    let sample_counts = vec![1_000, 10_000, 100_000];

    for &count in &sample_counts {
        let normal = Uncertain::normal(0.0, 1.0);

        group.bench_function(format!("normal_sequential_{}", count), |b| {
            b.iter(|| black_box(normal.take_samples(count)));
        });

        group.bench_function(format!("normal_parallel_{}", count), |b| {
            b.iter(|| black_box(normal.take_samples_par(count)));
        });

        let gamma = Uncertain::gamma(2.0, 1.0);

        group.bench_function(format!("gamma_sequential_{}", count), |b| {
            b.iter(|| black_box(gamma.take_samples(count)));
        });

        group.bench_function(format!("gamma_parallel_{}", count), |b| {
            b.iter(|| black_box(gamma.take_samples_par(count)));
        });

        group.bench_function(format!("gamma_cached_sequential_{}", count), |b| {
            b.iter_with_setup(
                || {
                    cache::clear_global_caches();
                    Uncertain::gamma(2.0, 1.0)
                },
                |dist| black_box(dist.take_samples_cached(count)),
            );
        });

        group.bench_function(format!("gamma_cached_parallel_{}", count), |b| {
            b.iter_with_setup(
                || {
                    cache::clear_global_caches();
                    Uncertain::gamma(2.0, 1.0)
                },
                |dist| black_box(dist.take_samples_cached_par(count)),
            );
        });
    }

    let base = Uncertain::normal(50.0, 10.0);
    let transformed = base
        .map(|x| x.powi(2))
        .map(|x| x.sqrt())
        .map(|x| (x / 10.0).sin() * 100.0);

    group.bench_function("complex_transform_sequential_10k", |b| {
        b.iter(|| black_box(transformed.take_samples(10_000)));
    });

    group.bench_function("complex_transform_parallel_10k", |b| {
        b.iter(|| black_box(transformed.take_samples_par(10_000)));
    });

    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    benchmark_statistical_operations,
    benchmark_interval_operations,
    benchmark_pdf_operations,
    benchmark_distribution_sampling,
    benchmark_computation_graphs,
    benchmark_cache_overhead,
    benchmark_parallel_sampling
);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    benchmark_statistical_operations,
    benchmark_interval_operations,
    benchmark_pdf_operations,
    benchmark_distribution_sampling,
    benchmark_computation_graphs,
    benchmark_cache_overhead
);

criterion_main!(benches);
