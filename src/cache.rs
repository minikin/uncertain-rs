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

        // Second call should use cached value
        let result2 = cache.get_or_compute("test", || {
            call_count += 1;
            99
        });
        assert_eq!(result2, 42);
        assert_eq!(call_count, 1); // Shouldn't increment
    }

    #[test]
    fn test_statistics_cache() {
        let cache = StatisticsCache::new();
        let test_id = uuid::Uuid::new_v4();

        let result = cache.get_or_compute_expected_value(test_id, 1000, || 42.0);
        assert_eq!(result, 42.0);

        // Should use cached value
        let result2 = cache.get_or_compute_expected_value(test_id, 1000, || 99.0);
        assert_eq!(result2, 42.0);
    }

    #[test]
    fn test_distribution_cache() {
        let cache = DistributionCache::new();
        let test_id = uuid::Uuid::new_v4();

        let result = cache.get_or_compute_samples(test_id, 100, || vec![1.0, 2.0, 3.0]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);

        // Should use cached value
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
}
