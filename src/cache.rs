#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Global cache managers
static STATS_CACHE: std::sync::LazyLock<StatisticsCache> =
    std::sync::LazyLock::new(StatisticsCache::new);
static DIST_CACHE: std::sync::LazyLock<DistributionCache> =
    std::sync::LazyLock::new(DistributionCache::new);

/// Thread-safe cache with TTL (time-to-live) support for expensive computations
pub struct TtlCache<K, V> {
    data: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    ttl: Duration,
}

struct CacheEntry<V> {
    value: V,
    created_at: Instant,
}

impl<K, V> TtlCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new TTL cache with specified time-to-live duration
    #[must_use]
    pub fn new(ttl: Duration) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            ttl,
        }
    }

    /// Get a value from cache if it exists and hasn't expired
    pub fn get(&self, key: &K) -> Option<V> {
        let cache = self.data.read().ok()?;
        let entry = cache.get(key)?;

        if entry.created_at.elapsed() < self.ttl {
            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut cache) = self.data.write() {
            cache.insert(
                key,
                CacheEntry {
                    value,
                    created_at: Instant::now(),
                },
            );
        }
    }

    /// Get or compute a value, caching the result
    pub fn get_or_compute<F>(&self, key: K, compute_fn: F) -> V
    where
        F: FnOnce() -> V,
    {
        if let Some(cached) = self.get(&key) {
            return cached;
        }

        let value = compute_fn();
        self.insert(key, value.clone());
        value
    }

    /// Clear expired entries from the cache
    pub fn cleanup_expired(&self) {
        if let Ok(mut cache) = self.data.write() {
            cache.retain(|_, entry| entry.created_at.elapsed() < self.ttl);
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.data.write() {
            cache.clear();
        }
    }

    /// Get the number of entries in the cache
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.read().map_or(0, |cache| cache.len())
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V> Default for TtlCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new(Duration::from_secs(300)) // 5 minutes default TTL
    }
}

/// Cache for statistical computations
pub struct StatisticsCache {
    expected_value: TtlCache<(uuid::Uuid, usize), f64>,
    variance: TtlCache<(uuid::Uuid, usize), f64>,
    std_dev: TtlCache<(uuid::Uuid, usize), f64>,
    skewness: TtlCache<(uuid::Uuid, usize), f64>,
    kurtosis: TtlCache<(uuid::Uuid, usize), f64>,
    confidence_intervals: TtlCache<(uuid::Uuid, usize, u64), (f64, f64)>, // confidence as f64 * 1000 for key
    cdf: TtlCache<(uuid::Uuid, usize, u64), f64>, // value as f64 * 1000 for key
    quantiles: TtlCache<(uuid::Uuid, usize, u64), f64>, // q as f64 * 1000 for key
}

impl StatisticsCache {
    /// Create a new statistics cache with default TTL
    #[must_use]
    pub fn new() -> Self {
        let ttl = Duration::from_secs(300); // 5 minutes
        Self {
            expected_value: TtlCache::new(ttl),
            variance: TtlCache::new(ttl),
            std_dev: TtlCache::new(ttl),
            skewness: TtlCache::new(ttl),
            kurtosis: TtlCache::new(ttl),
            confidence_intervals: TtlCache::new(ttl),
            cdf: TtlCache::new(ttl),
            quantiles: TtlCache::new(ttl),
        }
    }

    /// Cache expected value computation
    pub fn get_or_compute_expected_value<F>(
        &self,
        id: uuid::Uuid,
        sample_count: usize,
        compute: F,
    ) -> f64
    where
        F: FnOnce() -> f64,
    {
        self.expected_value
            .get_or_compute((id, sample_count), compute)
    }

    /// Cache variance computation
    pub fn get_or_compute_variance<F>(&self, id: uuid::Uuid, sample_count: usize, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        self.variance.get_or_compute((id, sample_count), compute)
    }

    /// Cache standard deviation computation
    pub fn get_or_compute_std_dev<F>(&self, id: uuid::Uuid, sample_count: usize, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        self.std_dev.get_or_compute((id, sample_count), compute)
    }

    /// Cache skewness computation
    pub fn get_or_compute_skewness<F>(&self, id: uuid::Uuid, sample_count: usize, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        self.skewness.get_or_compute((id, sample_count), compute)
    }

    /// Cache kurtosis computation
    pub fn get_or_compute_kurtosis<F>(&self, id: uuid::Uuid, sample_count: usize, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        self.kurtosis.get_or_compute((id, sample_count), compute)
    }

    /// Cache confidence interval computation
    pub fn get_or_compute_confidence_interval<F>(
        &self,
        id: uuid::Uuid,
        sample_count: usize,
        confidence: f64,
        compute: F,
    ) -> (f64, f64)
    where
        F: FnOnce() -> (f64, f64),
    {
        let confidence_key = (confidence * 1000.0) as u64;
        self.confidence_intervals
            .get_or_compute((id, sample_count, confidence_key), compute)
    }

    /// Cache CDF computation
    pub fn get_or_compute_cdf<F>(
        &self,
        id: uuid::Uuid,
        sample_count: usize,
        value: f64,
        compute: F,
    ) -> f64
    where
        F: FnOnce() -> f64,
    {
        let value_key = (value * 1000.0) as u64;
        self.cdf
            .get_or_compute((id, sample_count, value_key), compute)
    }

    /// Cache quantile computation
    pub fn get_or_compute_quantile<F>(
        &self,
        id: uuid::Uuid,
        sample_count: usize,
        q: f64,
        compute: F,
    ) -> f64
    where
        F: FnOnce() -> f64,
    {
        let q_key = (q * 1000.0) as u64;
        self.quantiles
            .get_or_compute((id, sample_count, q_key), compute)
    }

    /// Clear all statistical caches
    pub fn clear_all(&self) {
        self.expected_value.clear();
        self.variance.clear();
        self.std_dev.clear();
        self.skewness.clear();
        self.kurtosis.clear();
        self.confidence_intervals.clear();
        self.cdf.clear();
        self.quantiles.clear();
    }

    /// Clean up expired entries in all caches
    pub fn cleanup_all_expired(&self) {
        self.expected_value.cleanup_expired();
        self.variance.cleanup_expired();
        self.std_dev.cleanup_expired();
        self.skewness.cleanup_expired();
        self.kurtosis.cleanup_expired();
        self.confidence_intervals.cleanup_expired();
        self.cdf.cleanup_expired();
        self.quantiles.cleanup_expired();
    }
}

impl Default for StatisticsCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for distribution sampling operations
pub struct DistributionCache {
    samples: TtlCache<(uuid::Uuid, usize), Vec<f64>>,
    pdf_kde: TtlCache<(uuid::Uuid, usize, u64, u64), f64>, // x and bandwidth as keys
}

impl DistributionCache {
    /// Create a new distribution cache
    #[must_use]
    pub fn new() -> Self {
        let ttl = Duration::from_secs(300); // 5 minutes
        Self {
            samples: TtlCache::new(ttl),
            pdf_kde: TtlCache::new(ttl),
        }
    }

    /// Cache samples for reuse
    pub fn get_or_compute_samples<F>(
        &self,
        id: uuid::Uuid,
        sample_count: usize,
        compute: F,
    ) -> Vec<f64>
    where
        F: FnOnce() -> Vec<f64>,
    {
        self.samples.get_or_compute((id, sample_count), compute)
    }

    /// Cache PDF KDE computation
    pub fn get_or_compute_pdf_kde<F>(
        &self,
        id: uuid::Uuid,
        sample_count: usize,
        x: f64,
        bandwidth: f64,
        compute: F,
    ) -> f64
    where
        F: FnOnce() -> f64,
    {
        let x_key = (x * 1000.0) as u64;
        let bandwidth_key = (bandwidth * 1000.0) as u64;
        self.pdf_kde
            .get_or_compute((id, sample_count, x_key, bandwidth_key), compute)
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.samples.clear();
        self.pdf_kde.clear();
    }

    /// Clean up expired entries
    pub fn cleanup_all_expired(&self) {
        self.samples.cleanup_expired();
        self.pdf_kde.cleanup_expired();
    }
}

impl Default for DistributionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global statistics cache
#[must_use]
pub fn stats_cache() -> &'static StatisticsCache {
    &STATS_CACHE
}

/// Get the global distribution cache
#[must_use]
pub fn dist_cache() -> &'static DistributionCache {
    &DIST_CACHE
}

/// Cleanup expired entries in all global caches
pub fn cleanup_global_caches() {
    STATS_CACHE.cleanup_all_expired();
    DIST_CACHE.cleanup_all_expired();
}

/// Clear all global caches
pub fn clear_global_caches() {
    STATS_CACHE.clear_all();
    DIST_CACHE.clear_all();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_ttl_cache_basic() {
        let cache = TtlCache::new(Duration::from_millis(100));

        cache.insert("key1", "value1");
        assert_eq!(cache.get(&"key1"), Some("value1"));

        thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(&"key1"), None);
    }

    #[test]
    fn test_ttl_cache_get_or_compute() {
        let cache = TtlCache::new(Duration::from_secs(1));
        let mut call_count = 0;

        let result = cache.get_or_compute("test", || {
            call_count += 1;
            42
        });
        assert_eq!(result, 42);
        assert_eq!(call_count, 1);

        let result2 = cache.get_or_compute("test", || {
            call_count += 1;
            99
        });
        assert_eq!(result2, 42);
        assert_eq!(call_count, 1);
    }

    #[test]
    fn test_statistics_cache() {
        let cache = StatisticsCache::new();
        let test_id = uuid::Uuid::new_v4();

        let result = cache.get_or_compute_expected_value(test_id, 1000, || 42.0);
        assert_eq!(result, 42.0);

        let result2 = cache.get_or_compute_expected_value(test_id, 1000, || 99.0);
        assert_eq!(result2, 42.0);
    }

    #[test]
    fn test_distribution_cache() {
        let cache = DistributionCache::new();
        let test_id = uuid::Uuid::new_v4();

        let result = cache.get_or_compute_samples(test_id, 100, || vec![1.0, 2.0, 3.0]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);

        let result2 = cache.get_or_compute_samples(test_id, 100, || vec![4.0, 5.0, 6.0]);
        assert_eq!(result2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cache_cleanup() {
        let cache = TtlCache::new(Duration::from_millis(50));

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        assert_eq!(cache.len(), 2);

        thread::sleep(Duration::from_millis(100));
        cache.cleanup_expired();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_clear() {
        let cache = TtlCache::new(Duration::from_secs(1));

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_ttl_cache_default() {
        let cache = TtlCache::<String, i32>::default();

        cache.insert("test".to_string(), 42);
        assert_eq!(cache.get(&"test".to_string()), Some(42));
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_ttl_cache_is_empty() {
        let cache = TtlCache::new(Duration::from_secs(1));

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert("key", "value");
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_ttl_cache_concurrent_access() {
        use std::sync::Arc;

        let cache = Arc::new(TtlCache::new(Duration::from_secs(1)));
        let mut handles = vec![];

        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                cache_clone.insert(format!("key{i}"), i);
                cache_clone.get(&format!("key{i}"))
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_some());
        }

        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_statistics_cache_all_methods() {
        let cache = StatisticsCache::new();
        let test_id = uuid::Uuid::new_v4();

        let variance = cache.get_or_compute_variance(test_id, 1000, || 25.0);
        assert_eq!(variance, 25.0);
        let variance2 = cache.get_or_compute_variance(test_id, 1000, || 50.0);
        assert_eq!(variance2, 25.0); // Should use cached value

        let std_dev = cache.get_or_compute_std_dev(test_id, 1000, || 5.0);
        assert_eq!(std_dev, 5.0);

        let skewness = cache.get_or_compute_skewness(test_id, 1000, || 0.5);
        assert_eq!(skewness, 0.5);

        let kurtosis = cache.get_or_compute_kurtosis(test_id, 1000, || 3.0);
        assert_eq!(kurtosis, 3.0);

        let ci = cache.get_or_compute_confidence_interval(test_id, 1000, 0.95, || (1.0, 2.0));
        assert_eq!(ci, (1.0, 2.0));
        let ci2 = cache.get_or_compute_confidence_interval(test_id, 1000, 0.95, || (3.0, 4.0));
        assert_eq!(ci2, (1.0, 2.0)); // Should use cached value

        let cdf = cache.get_or_compute_cdf(test_id, 1000, 1.5, || 0.75);
        assert_eq!(cdf, 0.75);
        let cdf2 = cache.get_or_compute_cdf(test_id, 1000, 1.5, || 0.85);
        assert_eq!(cdf2, 0.75); // Should use cached value

        let quantile = cache.get_or_compute_quantile(test_id, 1000, 0.5, || 1.0);
        assert_eq!(quantile, 1.0);
        let quantile2 = cache.get_or_compute_quantile(test_id, 1000, 0.5, || 2.0);
        assert_eq!(quantile2, 1.0); // Should use cached value
    }

    #[test]
    fn test_statistics_cache_clear_and_cleanup() {
        let cache = StatisticsCache::new();
        let test_id = uuid::Uuid::new_v4();

        cache.get_or_compute_expected_value(test_id, 1000, || 42.0);
        cache.get_or_compute_variance(test_id, 1000, || 25.0);
        cache.get_or_compute_confidence_interval(test_id, 1000, 0.95, || (1.0, 2.0));

        cache.clear_all();

        let result = cache.get_or_compute_expected_value(test_id, 1000, || 99.0);
        assert_eq!(result, 99.0);

        cache.cleanup_all_expired();
        let result2 = cache.get_or_compute_expected_value(test_id, 1000, || 88.0);
        assert_eq!(result2, 99.0); // Should still be cached
    }

    #[test]
    fn test_statistics_cache_default() {
        let cache = StatisticsCache::default();
        let test_id = uuid::Uuid::new_v4();

        let result = cache.get_or_compute_expected_value(test_id, 1000, || 42.0);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_distribution_cache_all_methods() {
        let cache = DistributionCache::new();
        let test_id = uuid::Uuid::new_v4();

        let samples = cache.get_or_compute_samples(test_id, 100, || vec![1.0, 2.0, 3.0]);
        assert_eq!(samples, vec![1.0, 2.0, 3.0]);

        let pdf = cache.get_or_compute_pdf_kde(test_id, 100, 1.5, 0.1, || 0.25);
        assert_eq!(pdf, 0.25);
        let pdf2 = cache.get_or_compute_pdf_kde(test_id, 100, 1.5, 0.1, || 0.50);
        assert_eq!(pdf2, 0.25);

        let pdf3 = cache.get_or_compute_pdf_kde(test_id, 100, 1.6, 0.1, || 0.30);
        assert_eq!(pdf3, 0.30);

        let pdf4 = cache.get_or_compute_pdf_kde(test_id, 100, 1.5, 0.2, || 0.35);
        assert_eq!(pdf4, 0.35);
    }

    #[test]
    fn test_distribution_cache_clear_and_cleanup() {
        let cache = DistributionCache::new();
        let test_id = uuid::Uuid::new_v4();

        cache.get_or_compute_samples(test_id, 100, || vec![1.0, 2.0]);
        cache.get_or_compute_pdf_kde(test_id, 100, 1.5, 0.1, || 0.25);

        cache.clear_all();

        let samples = cache.get_or_compute_samples(test_id, 100, || vec![3.0, 4.0]);
        assert_eq!(samples, vec![3.0, 4.0]);

        let pdf = cache.get_or_compute_pdf_kde(test_id, 100, 1.5, 0.1, || 0.50);
        assert_eq!(pdf, 0.50);

        cache.cleanup_all_expired();
        let samples2 = cache.get_or_compute_samples(test_id, 100, || vec![5.0, 6.0]);
        assert_eq!(samples2, vec![3.0, 4.0]);
    }

    #[test]
    fn test_distribution_cache_default() {
        let cache = DistributionCache::default();
        let test_id = uuid::Uuid::new_v4();

        let samples = cache.get_or_compute_samples(test_id, 100, || vec![1.0, 2.0]);
        assert_eq!(samples, vec![1.0, 2.0]);
    }

    #[test]
    fn test_global_cache_functions() {
        let stats = stats_cache();
        let dist = dist_cache();

        let test_id = uuid::Uuid::new_v4();

        let result = stats.get_or_compute_expected_value(test_id, 1000, || 42.0);
        assert_eq!(result, 42.0);

        let samples = dist.get_or_compute_samples(test_id, 100, || vec![1.0, 2.0]);
        assert_eq!(samples, vec![1.0, 2.0]);

        cleanup_global_caches();

        let result2 = stats.get_or_compute_expected_value(test_id, 1000, || 99.0);
        assert_eq!(result2, 42.0);

        clear_global_caches();

        let result3 = stats.get_or_compute_expected_value(test_id, 1000, || 99.0);
        assert_eq!(result3, 99.0);

        let samples2 = dist.get_or_compute_samples(test_id, 100, || vec![3.0, 4.0]);
        assert_eq!(samples2, vec![3.0, 4.0]);
    }

    #[test]
    fn test_cache_key_precision_handling() {
        let cache = StatisticsCache::new();
        let test_id = uuid::Uuid::new_v4();

        let ci1 = cache.get_or_compute_confidence_interval(test_id, 1000, 0.95, || (1.0, 2.0));
        let ci2 = cache.get_or_compute_confidence_interval(test_id, 1000, 0.96, || (3.0, 4.0));

        assert_eq!(ci1, (1.0, 2.0));
        assert_eq!(ci2, (3.0, 4.0));

        let ci3 = cache.get_or_compute_confidence_interval(test_id, 1000, 0.95, || (5.0, 6.0));
        assert_eq!(ci3, (1.0, 2.0));
    }
}
