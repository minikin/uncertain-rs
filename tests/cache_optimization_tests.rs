use uncertain_rs::computation::{AdaptiveSampling, CachingStrategy, SampleContext};
use uncertain_rs::{Uncertain, cache};

#[test]
fn test_improved_cache_hit_rates_with_quantization() {
    cache::clear_global_caches();

    let normal = Uncertain::normal(0.0, 1.0);

    let _result1 = normal.cdf(1.0001, 1000);
    let _result2 = normal.cdf(1.0002, 1000); // Should hit cache due to quantization
    let _result3 = normal.cdf(1.0009, 1000); // Should hit cache due to quantization

    let (stats_stats, _) = cache::global_cache_stats();

    assert!(
        stats_stats.hits > 0,
        "Should have cache hits due to quantization"
    );
    assert!(stats_stats.hit_rate() > 0.0, "Hit rate should be positive");
}

#[test]
fn test_adaptive_sampling_convergence() {
    let normal = Uncertain::normal(10.0, 2.0);

    let config = AdaptiveSampling {
        min_samples: 50,
        max_samples: 5000,
        error_threshold: 0.05,
        growth_factor: 1.5,
    };

    let adaptive_mean = normal.expected_value_adaptive(&config);
    let fixed_mean = normal.expected_value(1000);

    assert!((adaptive_mean - fixed_mean).abs() < 1.0);
    assert!(adaptive_mean > 8.0 && adaptive_mean < 12.0);
}

#[test]
fn test_cache_statistics_tracking() {
    cache::clear_global_caches();

    let normal = Uncertain::normal(0.0, 1.0);

    // First computation - should be cache miss
    let _result1 = normal.expected_value(1000);
    let (stats1, _) = cache::global_cache_stats();
    // The first call may have some hits due to reused calculations
    assert!(stats1.misses > 0);

    // Second computation with same parameters - should be cache hit
    let _result2 = normal.expected_value(1000);
    let (stats2, _) = cache::global_cache_stats();
    assert!(stats2.hits > 0);
    assert!(stats2.hit_rate() > 0.0);
}

#[test]
fn test_caching_strategy_conservative() {
    let context = SampleContext::with_caching_strategy(CachingStrategy::Conservative);

    // Simple leaf node - should not be cached in conservative mode
    let simple_leaf = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    assert!(!context.should_cache_node(&simple_leaf));

    // Complex nested expression - should be cached
    let left = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    let right = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let inner = uncertain_rs::computation::ComputationNode::binary_op(
        left,
        right,
        uncertain_rs::operations::arithmetic::BinaryOperation::Add,
    );
    let outer_left = uncertain_rs::computation::ComputationNode::leaf(|| 3.0);
    let complex = uncertain_rs::computation::ComputationNode::binary_op(
        inner,
        outer_left,
        uncertain_rs::operations::arithmetic::BinaryOperation::Mul,
    );

    assert!(context.should_cache_node(&complex));
}

#[test]
fn test_caching_strategy_adaptive() {
    let context = SampleContext::with_caching_strategy(CachingStrategy::Adaptive);

    // Simple leaf - low complexity, should not be cached
    let simple = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    assert!(!context.should_cache_node(&simple));

    // Build a complex computation graph that should be cached
    let mut complex = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    for _ in 0..3 {
        let right = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
        complex = uncertain_rs::computation::ComputationNode::binary_op(
            complex,
            right,
            uncertain_rs::operations::arithmetic::BinaryOperation::Add,
        );
    }

    assert!(context.should_cache_node(&complex));
}

#[test]
fn test_computation_graph_complexity_metric() {
    let leaf = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    assert_eq!(leaf.compute_complexity(), 1);

    let left = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    let right = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let binary = uncertain_rs::computation::ComputationNode::binary_op(
        left,
        right,
        uncertain_rs::operations::arithmetic::BinaryOperation::Add,
    );
    assert_eq!(binary.compute_complexity(), 4); // 2 + 1 + 1

    let condition = uncertain_rs::computation::ComputationNode::leaf(|| true);
    let if_true = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    let if_false = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let conditional =
        uncertain_rs::computation::ComputationNode::conditional(condition, if_true, if_false);
    assert_eq!(conditional.compute_complexity(), 8); // 5 + 1 + 1 + 1
}

#[test]
fn test_structural_hash_consistency() {
    let left1 = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    let right1 = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let node1 = uncertain_rs::computation::ComputationNode::binary_op(
        left1,
        right1,
        uncertain_rs::operations::arithmetic::BinaryOperation::Add,
    );

    let left2 = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    let right2 = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let node2 = uncertain_rs::computation::ComputationNode::binary_op(
        left2,
        right2,
        uncertain_rs::operations::arithmetic::BinaryOperation::Add,
    );

    // TODO: These will have different hashes due to different UUIDs. We'd need node-level sharing for true CSE.
    let hash1 = node1.structural_hash();
    let hash2 = node2.structural_hash();

    // Hashes should be different due to different leaf IDs
    assert_ne!(hash1, hash2);

    // But the same node should always produce the same hash
    let hash1_again = node1.structural_hash();
    assert_eq!(hash1, hash1_again);
}

#[test]
fn test_complex_computation_graph_performance() {
    cache::clear_global_caches();

    // Create a complex expression: (x + y) * (x - y) * (x / y) + (y ^ 2)
    let x = Uncertain::normal(5.0, 1.0);
    let y = Uncertain::normal(2.0, 0.5);

    let sum = x.clone() + y.clone();
    let diff = x.clone() - y.clone();
    let div = x.clone() / y.clone();
    let y_squared = y.clone() * y.clone();

    let complex_expr = (sum * diff * div) + y_squared;

    // First evaluation - should populate caches
    let result1 = complex_expr.expected_value(1000);
    let (stats_after_first, _) = cache::global_cache_stats();

    // Second evaluation - should benefit from caching
    let result2 = complex_expr.expected_value(1000);
    let (stats_after_second, _) = cache::global_cache_stats();

    // Results should be identical due to caching
    assert_eq!(result1, result2);

    // Should have cache hits on second evaluation
    assert!(stats_after_second.hits > stats_after_first.hits);

    cache::print_cache_report();
}

#[test]
fn test_cache_performance_report() {
    cache::clear_global_caches();

    let normal = Uncertain::normal(0.0, 1.0);

    for i in 0..10 {
        let _ = normal.expected_value(100 * (i + 1));
        let _ = normal.variance(100 * (i + 1));
        let _ = normal.standard_deviation(100 * (i + 1));
    }

    cache::print_cache_report();

    let (stats, _dist_stats) = cache::global_cache_stats();

    assert!(stats.hits + stats.misses > 0);
}

#[test]
fn test_graph_optimizer_subexpression_elimination() {
    let mut optimizer = uncertain_rs::computation::GraphOptimizer::new();

    let left = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);
    let right = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let node = uncertain_rs::computation::ComputationNode::binary_op(
        left,
        right,
        uncertain_rs::operations::arithmetic::BinaryOperation::Add,
    );

    let optimized = optimizer.optimize(node);

    let result = optimized.evaluate_fresh();
    assert_eq!(result, 3.0);
}

#[test]
fn test_different_precision_cache_hits() {
    cache::clear_global_caches();

    let normal = Uncertain::normal(0.0, 1.0);

    // These values should map to the same quantized key
    let confidence_intervals = [
        normal.confidence_interval(0.9501, 1000),
        normal.confidence_interval(0.9502, 1000), // Should hit cache
        normal.confidence_interval(0.9509, 1000), // Should hit cache
    ];

    let (stats, _) = cache::global_cache_stats();

    assert!(stats.hits > 0, "Should have cache hits due to quantization");

    let first = confidence_intervals[0];
    for &interval in &confidence_intervals[1..] {
        assert!((interval.0 - first.0).abs() < 0.5);
        assert!((interval.1 - first.1).abs() < 0.5);
    }
}
