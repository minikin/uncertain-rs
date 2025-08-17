#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::Uncertain;
use crate::cache;
use crate::computation::AdaptiveSampling;
use crate::traits::Shareable;
use std::collections::HashMap;
use std::hash::Hash;

/// Statistical analysis methods for uncertain values
impl<T> Uncertain<T>
where
    T: Shareable,
{
    /// Estimates the mode (most frequent value) of the distribution
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    /// use std::collections::HashMap;
    ///
    /// let mut probs = HashMap::new();
    /// probs.insert("red", 0.5);
    /// probs.insert("blue", 0.3);
    /// probs.insert("green", 0.2);
    ///
    /// let categorical = Uncertain::categorical(&probs).unwrap();
    /// let mode = categorical.mode(1000);
    /// // Should likely be "red"
    /// ```
    #[must_use]
    pub fn mode(&self, sample_count: usize) -> Option<T>
    where
        T: Hash + Eq,
    {
        let samples = self.take_samples(sample_count);
        if samples.is_empty() {
            return None;
        }

        let mut counts = HashMap::new();
        for sample in samples {
            *counts.entry(sample).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
    }

    /// Creates a histogram of the distribution
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let bernoulli = Uncertain::bernoulli(0.7);
    /// let histogram = bernoulli.histogram(1000);
    /// // Should show roughly 700 true, 300 false
    /// ```
    #[must_use]
    pub fn histogram(&self, sample_count: usize) -> HashMap<T, usize>
    where
        T: Hash + Eq,
    {
        let samples = self.take_samples(sample_count);
        let mut histogram = HashMap::new();

        for sample in samples {
            *histogram.entry(sample).or_insert(0) += 1;
        }

        histogram
    }

    /// Calculates the empirical entropy of the distribution in bits
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let uniform_coin = Uncertain::bernoulli(0.5);
    /// let entropy = uniform_coin.entropy(1000);
    /// // Should be close to 1.0 bit for fair coin
    /// ```
    #[must_use]
    pub fn entropy(&self, sample_count: usize) -> f64
    where
        T: Hash + Eq,
    {
        let histogram = self.histogram(sample_count);
        let total = sample_count as f64;

        histogram
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }
}

/// Statistical methods for numeric types
impl<T> Uncertain<T>
where
    T: Clone + Send + Sync + Into<f64> + 'static,
{
    /// Calculates the expected value (mean) of the distribution
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(10.0, 2.0);
    /// let mean = normal.expected_value(1000);
    /// // Should be approximately 10.0
    /// ```
    #[must_use]
    pub fn expected_value(&self, sample_count: usize) -> f64 {
        cache::stats_cache().get_or_compute_expected_value(self.id, sample_count, || {
            let samples = self.take_samples(sample_count);
            let sum: f64 = samples.into_iter().map(Into::into).sum();
            sum / sample_count as f64
        })
    }

    /// Calculates the expected value using adaptive sampling for better efficiency
    ///
    /// This method automatically determines the optimal sample count based on
    /// convergence criteria, potentially improving cache hit rates for similar computations.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    /// use uncertain_rs::computation::AdaptiveSampling;
    ///
    /// let normal = Uncertain::normal(10.0, 2.0);
    /// let config = AdaptiveSampling::default();
    /// let mean = normal.expected_value_adaptive(&config);
    /// // Should be approximately 10.0 with optimal sample count
    /// ```
    #[must_use]
    pub fn expected_value_adaptive(&self, config: &AdaptiveSampling) -> f64 {
        let mut sample_count = config.min_samples;
        let mut prev_mean = 0.0;

        loop {
            let mean = self.expected_value(sample_count);

            if sample_count > config.min_samples {
                let relative_error = ((mean - prev_mean) / mean).abs();
                if relative_error < config.error_threshold || sample_count >= config.max_samples {
                    return mean;
                }
            }

            prev_mean = mean;
            sample_count =
                ((sample_count as f64 * config.growth_factor) as usize).min(config.max_samples);
        }
    }

    /// Calculates the variance of the distribution
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 2.0);
    /// let variance = normal.variance(1000);
    /// // Should be approximately 4.0 (std_dev^2)
    /// ```
    #[must_use]
    pub fn variance(&self, sample_count: usize) -> f64 {
        cache::stats_cache().get_or_compute_variance(self.id, sample_count, || {
            let samples: Vec<f64> = self
                .take_samples(sample_count)
                .into_iter()
                .map(Into::into)
                .collect();

            let mean = samples.iter().sum::<f64>() / samples.len() as f64;

            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64
        })
    }

    /// Calculates the standard deviation of the distribution
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 2.0);
    /// let std_dev = normal.standard_deviation(1000);
    /// // Should be approximately 2.0
    /// ```
    #[must_use]
    pub fn standard_deviation(&self, sample_count: usize) -> f64 {
        cache::stats_cache()
            .get_or_compute_std_dev(self.id, sample_count, || self.variance(sample_count).sqrt())
    }

    /// Calculates the skewness of the distribution
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let skewness = normal.skewness(1000);
    /// // Should be approximately 0 for normal distribution
    /// ```
    #[must_use]
    pub fn skewness(&self, sample_count: usize) -> f64 {
        cache::stats_cache().get_or_compute_skewness(self.id, sample_count, || {
            let samples: Vec<f64> = self
                .take_samples(sample_count)
                .into_iter()
                .map(Into::into)
                .collect();

            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let std_dev = self.standard_deviation(sample_count);

            if std_dev == 0.0 {
                return 0.0;
            }

            let n = samples.len() as f64;
            samples
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n
        })
    }

    /// Calculates the excess kurtosis of the distribution
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let kurtosis = normal.kurtosis(1000);
    /// // Should be approximately 0 for normal distribution (excess kurtosis)
    /// ```
    #[must_use]
    pub fn kurtosis(&self, sample_count: usize) -> f64 {
        cache::stats_cache().get_or_compute_kurtosis(self.id, sample_count, || {
            let samples: Vec<f64> = self
                .take_samples(sample_count)
                .into_iter()
                .map(Into::into)
                .collect();

            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let std_dev = self.standard_deviation(sample_count);

            if std_dev == 0.0 {
                return 0.0;
            }

            let n = samples.len() as f64;
            let kurt = samples
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n;

            kurt - 3.0 // Excess kurtosis (subtract 3 for normal distribution baseline)
        })
    }
}

/// Statistical methods for ordered types
impl<T> Uncertain<T>
where
    T: Clone + Send + Sync + PartialOrd + Into<f64> + 'static,
{
    /// Calculates confidence interval bounds
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Panics
    ///
    /// Panics if the samples contain values that cannot be compared (e.g., NaN values).
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(100.0, 15.0);
    /// let (lower, upper) = normal.confidence_interval(0.95, 1000);
    /// // 95% of values should fall between lower and upper
    /// ```
    #[must_use]
    pub fn confidence_interval(&self, confidence: f64, sample_count: usize) -> (f64, f64) {
        cache::stats_cache().get_or_compute_confidence_interval(
            self.id,
            sample_count,
            confidence,
            || {
                let mut samples: Vec<f64> = self
                    .take_samples(sample_count)
                    .into_iter()
                    .map(Into::into)
                    .collect();

                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let alpha = 1.0 - confidence;
                let samples_len = samples.len();
                let lower_idx = ((alpha / 2.0) * samples_len as f64).floor() as usize;
                let upper_idx =
                    (((1.0 - alpha / 2.0) * samples_len as f64).floor() as usize).saturating_sub(1);

                let lower_idx = lower_idx.min(samples.len() - 1);
                let upper_idx = upper_idx.min(samples.len() - 1);

                (samples[lower_idx], samples[upper_idx])
            },
        )
    }

    /// Estimates the cumulative distribution function (CDF) at a given value
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let prob = normal.cdf(0.0, 1000);
    /// // Should be approximately 0.5 for standard normal at 0
    /// ```
    #[must_use]
    pub fn cdf(&self, value: f64, sample_count: usize) -> f64 {
        cache::stats_cache().get_or_compute_cdf(self.id, sample_count, value, || {
            let samples: Vec<f64> = self
                .take_samples(sample_count)
                .into_iter()
                .map(Into::into)
                .collect();

            let count = samples.iter().filter(|&&x| x <= value).count();
            count as f64 / samples.len() as f64
        })
    }

    /// Estimates quantiles of the distribution
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Panics
    ///
    /// Panics if the samples contain values that cannot be compared (e.g., NaN values).
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let median = normal.quantile(0.5, 1000);
    /// // Should be approximately 0.0 for standard normal
    /// ```
    #[must_use]
    pub fn quantile(&self, q: f64, sample_count: usize) -> f64 {
        cache::stats_cache().get_or_compute_quantile(self.id, sample_count, q, || {
            let mut samples: Vec<f64> = self
                .take_samples(sample_count)
                .into_iter()
                .map(Into::into)
                .collect();

            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let index = (q * samples.len().saturating_sub(1) as f64).floor() as usize;
            let index = index.min(samples.len() - 1);

            samples[index]
        })
    }

    /// Calculates the interquartile range (IQR)
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let iqr = normal.interquartile_range(1000);
    /// // IQR for standard normal is approximately 1.35
    /// ```
    #[must_use]
    pub fn interquartile_range(&self, sample_count: usize) -> f64 {
        let q75 = self.quantile(0.75, sample_count);
        let q25 = self.quantile(0.25, sample_count);
        q75 - q25
    }

    /// Estimates the median absolute deviation (MAD)
    ///
    /// # Panics
    ///
    /// Panics if the samples contain values that cannot be compared (e.g., NaN values).
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let mad = normal.median_absolute_deviation(1000);
    /// ```
    #[must_use]
    pub fn median_absolute_deviation(&self, sample_count: usize) -> f64 {
        let samples: Vec<f64> = self
            .take_samples(sample_count)
            .into_iter()
            .map(std::convert::Into::into)
            .collect();

        let median = self.quantile(0.5, sample_count);
        let mut deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();

        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad_index = (deviations.len() - 1) / 2;
        deviations[mad_index]
    }
}

/// Advanced statistical methods
impl Uncertain<f64> {
    /// Estimates the probability density function (PDF) using kernel density estimation
    ///
    /// This method uses caching to avoid recomputing the same result.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let density = normal.pdf_kde(0.0, 1000, 0.1);
    /// ```
    #[must_use]
    pub fn pdf_kde(&self, x: f64, sample_count: usize, bandwidth: f64) -> f64 {
        cache::dist_cache().get_or_compute_pdf_kde(self.id, sample_count, x, bandwidth, || {
            let samples = self.take_samples(sample_count);

            let kernel_sum: f64 = samples
                .iter()
                .map(|&xi| {
                    let z = (x - xi) / bandwidth;
                    (-0.5 * z * z).exp()
                })
                .sum();

            kernel_sum / (sample_count as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt())
        })
    }

    /// Estimates the log-likelihood of a value using kernel density estimation
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let log_likelihood = normal.log_likelihood(0.0, 1000, 0.1);
    /// ```
    #[must_use]
    pub fn log_likelihood(&self, x: f64, sample_count: usize, bandwidth: f64) -> f64 {
        let pdf = self.pdf_kde(x, sample_count, bandwidth);
        if pdf > 0.0 {
            pdf.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Estimates correlation with another uncertain value
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let x = Uncertain::normal(0.0, 1.0);
    /// let y = x.map(|v| v * 2.0 + Uncertain::normal(0.0, 0.5).sample());
    /// let correlation = x.correlation(&y, 1000);
    /// // Should be positive correlation
    /// ```
    #[must_use]
    pub fn correlation(&self, other: &Uncertain<f64>, sample_count: usize) -> f64 {
        let samples_x = self.take_samples(sample_count);
        let samples_y = other.take_samples(sample_count);

        let mean_x = samples_x.iter().sum::<f64>() / sample_count as f64;
        let mean_y = samples_y.iter().sum::<f64>() / sample_count as f64;

        let numerator: f64 = samples_x
            .iter()
            .zip(samples_y.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();

        let var_x: f64 = samples_x.iter().map(|x| (x - mean_x).powi(2)).sum();

        let var_y: f64 = samples_y.iter().map(|y| (y - mean_y).powi(2)).sum();

        if var_x * var_y > 0.0 {
            numerator / (var_x * var_y).sqrt()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_expected_value() {
        let normal = Uncertain::normal(10.0, 1.0);
        let mean = normal.expected_value(1000);
        assert!((mean - 10.0).abs() < 0.2);
    }

    #[test]
    fn test_standard_deviation() {
        let normal = Uncertain::normal(0.0, 2.0);
        let std_dev = normal.standard_deviation(1000);
        assert!((std_dev - 2.0).abs() < 0.3);
    }

    #[test]
    fn test_confidence_interval() {
        let normal = Uncertain::normal(0.0, 1.0);
        let (lower, upper) = normal.confidence_interval(0.95, 1000);

        // For standard normal, 95% CI should be approximately [-1.96, 1.96]
        assert!(lower < -1.5 && lower > -2.5);
        assert!(upper > 1.5 && upper < 2.5);
    }

    #[test]
    fn test_cdf() {
        let normal = Uncertain::normal(0.0, 1.0);
        let prob = normal.cdf(0.0, 1000);

        // For standard normal, P(X <= 0) should be approximately 0.5
        assert!((prob - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_quantile() {
        let normal = Uncertain::normal(0.0, 1.0);
        let median = normal.quantile(0.5, 1000);

        // Median of standard normal should be approximately 0
        assert!(median.abs() < 0.2);
    }

    #[test]
    fn test_mode_categorical() {
        let mut probs = HashMap::new();
        probs.insert("red", 0.6);
        probs.insert("blue", 0.4);

        let categorical = Uncertain::categorical(&probs).unwrap();
        let mode = categorical.mode(1000);

        // Mode should likely be "red" with 60% probability
        assert_eq!(mode, Some("red"));
    }

    #[test]
    fn test_entropy() {
        let fair_coin = Uncertain::bernoulli(0.5);
        let entropy = fair_coin.entropy(1000);

        // Fair coin should have entropy close to 1 bit
        assert!((entropy - 1.0).abs() < 0.1);

        let biased_coin = Uncertain::bernoulli(0.9);
        let biased_entropy = biased_coin.entropy(1000);

        // Biased coin should have lower entropy
        assert!(biased_entropy < entropy);
    }

    #[test]
    fn test_skewness_normal() {
        let normal = Uncertain::normal(0.0, 1.0);
        let skewness = normal.skewness(1000);

        // Normal distribution should have skewness close to 0
        assert!(skewness.abs() < 0.3);
    }

    #[test]
    fn test_kurtosis_normal() {
        let normal = Uncertain::normal(0.0, 1.0);
        let kurtosis = normal.kurtosis(1000);

        // Normal distribution should have excess kurtosis close to 0
        // Allow for some statistical variance in the test
        assert!(kurtosis.abs() < 2.0);
    }

    #[test]
    fn test_mode_empty_samples() {
        let uncertain = Uncertain::new(|| None::<i32>);
        let mode = uncertain.mode(0);
        assert_eq!(mode, None);
    }

    #[test]
    fn test_mode_integers() {
        let uniform = Uncertain::new(|| if rand::random::<f64>() < 0.7 { 1 } else { 2 });
        let mode = uniform.mode(1000);
        assert_eq!(mode, Some(1));
    }

    #[test]
    fn test_histogram_empty() {
        let uncertain = Uncertain::new(|| 42);
        let histogram = uncertain.histogram(0);
        assert!(histogram.is_empty());
    }

    #[test]
    fn test_histogram_bernoulli() {
        let bernoulli = Uncertain::bernoulli(0.3);
        let histogram = bernoulli.histogram(1000);

        let true_count = histogram.get(&true).copied().unwrap_or(0);
        let false_count = histogram.get(&false).copied().unwrap_or(0);

        assert!(true_count > 200);
        assert!(true_count < 400);
        assert!(false_count > 600);
        assert!(false_count < 800);
        assert_eq!(true_count + false_count, 1000);
    }

    #[test]
    fn test_variance_standalone() {
        let normal = Uncertain::normal(5.0, 3.0);
        let variance = normal.variance(1000);
        assert!((variance - 9.0).abs() < 1.5);
    }

    #[test]
    fn test_variance_zero() {
        let constant = Uncertain::new(|| 42.0);
        let variance = constant.variance(1000);
        assert!(variance < 0.001);
    }

    #[test]
    fn test_skewness_zero_std_dev() {
        let constant = Uncertain::new(|| 5.0);
        let skewness = constant.skewness(1000);
        assert!(skewness.abs() < f64::EPSILON);
    }

    #[test]
    fn test_kurtosis_zero_std_dev() {
        let constant = Uncertain::new(|| 5.0);
        let kurtosis = constant.kurtosis(1000);
        assert!(kurtosis.abs() < f64::EPSILON);
    }

    #[test]
    fn test_pdf_kde() {
        let normal = Uncertain::normal(0.0, 1.0);
        let density_at_mean = normal.pdf_kde(0.0, 1000, 0.1);
        let density_at_tail = normal.pdf_kde(3.0, 1000, 0.1);

        assert!(density_at_mean > density_at_tail);
        assert!(density_at_mean > 0.0);
        assert!(density_at_tail > 0.0);
    }

    #[test]
    fn test_log_likelihood() {
        let normal = Uncertain::normal(0.0, 1.0);
        let ll_at_mean = normal.log_likelihood(0.0, 1000, 0.1);
        let ll_at_tail = normal.log_likelihood(3.0, 1000, 0.1);

        assert!(ll_at_mean > ll_at_tail);
        assert!(ll_at_mean.is_finite());
    }

    #[test]
    fn test_log_likelihood_zero_pdf() {
        let point = Uncertain::new(|| 0.0);
        let ll = point.log_likelihood(10.0, 100, 0.01);
        assert!(ll.is_infinite() && ll.is_sign_negative());
    }

    #[test]
    fn test_correlation_positive() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let x_counter = Arc::new(AtomicUsize::new(0));
        let x = Uncertain::new({
            let values = values.clone();
            let counter = x_counter.clone();
            move || {
                let idx = counter.fetch_add(1, Ordering::SeqCst);
                values[idx % values.len()]
            }
        });

        let y_counter = Arc::new(AtomicUsize::new(0));
        let y = Uncertain::new({
            let values = y_values.clone();
            let counter = y_counter.clone();
            move || {
                let idx = counter.fetch_add(1, Ordering::SeqCst);
                values[idx % values.len()]
            }
        });

        let correlation = x.correlation(&y, 5);
        assert!((correlation - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_correlation_negative() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let x_counter = Arc::new(AtomicUsize::new(0));
        let x = Uncertain::new({
            let values = values.clone();
            let counter = x_counter.clone();
            move || {
                let idx = counter.fetch_add(1, Ordering::SeqCst);
                values[idx % values.len()]
            }
        });

        let y_counter = Arc::new(AtomicUsize::new(0));
        let y = Uncertain::new({
            let values = y_values.clone();
            let counter = y_counter.clone();
            move || {
                let idx = counter.fetch_add(1, Ordering::SeqCst);
                values[idx % values.len()]
            }
        });

        let correlation = x.correlation(&y, 5);
        assert!((correlation + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_correlation_independent() {
        let x = Uncertain::normal(0.0, 1.0);
        let y = Uncertain::normal(10.0, 1.0);
        let correlation = x.correlation(&y, 1000);

        assert!(correlation.abs() < 0.2);
    }

    #[test]
    fn test_correlation_zero_variance() {
        let x = Uncertain::new(|| 5.0);
        let y = Uncertain::normal(0.0, 1.0);
        let correlation = x.correlation(&y, 1000);

        assert!(correlation.abs() < f64::EPSILON);
    }

    #[test]
    fn test_interquartile_range() {
        let normal = Uncertain::normal(0.0, 1.0);
        let iqr = normal.interquartile_range(1000);

        assert!(iqr > 1.0);
        assert!(iqr < 2.0);
    }

    #[test]
    fn test_median_absolute_deviation() {
        let normal = Uncertain::normal(0.0, 1.0);
        let mad = normal.median_absolute_deviation(1000);

        assert!(mad > 0.5);
        assert!(mad < 1.0);
    }

    #[test]
    fn test_quantile_extremes() {
        let normal = Uncertain::normal(0.0, 1.0);
        let min_quantile = normal.quantile(0.0, 1000);
        let max_quantile = normal.quantile(1.0, 1000);

        assert!(min_quantile < max_quantile);
        assert!(min_quantile < -2.0);
        assert!(max_quantile > 2.0);
    }

    #[test]
    fn test_cdf_extremes() {
        let normal = Uncertain::normal(0.0, 1.0);
        let prob_low = normal.cdf(-5.0, 1000);
        let prob_high = normal.cdf(5.0, 1000);

        assert!(prob_low < 0.1);
        assert!(prob_high > 0.9);
        assert!(prob_low < prob_high);
    }

    #[test]
    fn test_confidence_interval_different_levels() {
        let normal = Uncertain::normal(0.0, 1.0);
        let (lower_90, upper_90) = normal.confidence_interval(0.90, 1000);
        let (lower_99, upper_99) = normal.confidence_interval(0.99, 1000);

        assert!(lower_99 < lower_90);
        assert!(upper_99 > upper_90);
        assert!(upper_90 - lower_90 < upper_99 - lower_99);
    }

    #[test]
    fn test_entropy_deterministic() {
        let constant = Uncertain::new(|| "always");
        let entropy = constant.entropy(1000);
        assert!(entropy < 0.01);
    }

    #[test]
    fn test_entropy_maximum() {
        let mut probs = HashMap::new();
        probs.insert("a", 0.25);
        probs.insert("b", 0.25);
        probs.insert("c", 0.25);
        probs.insert("d", 0.25);

        let uniform = Uncertain::categorical(&probs).unwrap();
        let entropy = uniform.entropy(2000);

        assert!((entropy - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_mode_tie_handling() {
        let balanced = Uncertain::new(|| if rand::random::<bool>() { 1 } else { 2 });
        let mode = balanced.mode(1000);
        assert!(mode == Some(1) || mode == Some(2));
    }

    #[test]
    fn test_statistical_consistency() {
        let normal = Uncertain::normal(10.0, 2.0);

        let mean = normal.expected_value(2000);
        let median = normal.quantile(0.5, 2000);
        let mode_samples: Vec<f64> = (0..1000).map(|_| normal.sample()).collect();
        let empirical_mean = mode_samples.iter().sum::<f64>() / mode_samples.len() as f64;

        assert!((mean - 10.0).abs() < 0.3);
        assert!((median - 10.0).abs() < 0.3);
        assert!((empirical_mean - 10.0).abs() < 0.3);
        assert!((mean - median).abs() < 0.3);
    }

    #[test]
    fn test_caching_behavior() {
        let normal = Uncertain::normal(5.0, 1.0);

        let mean1 = normal.expected_value(1000);
        let mean2 = normal.expected_value(1000);
        assert!((mean1 - mean2).abs() < f64::EPSILON);

        let var1 = normal.variance(1000);
        let var2 = normal.variance(1000);
        assert!((var1 - var2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_large_sample_counts() {
        let normal = Uncertain::normal(0.0, 1.0);
        let mean = normal.expected_value(10000);
        let std_dev = normal.standard_deviation(10000);

        assert!((mean - 0.0).abs() < 0.1);
        assert!((std_dev - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_histogram_with_different_types() {
        let chars = Uncertain::new(|| match rand::random::<u8>() % 3 {
            0 => 'A',
            1 => 'B',
            _ => 'C',
        });

        let histogram = chars.histogram(300);
        assert_eq!(histogram.len(), 3);
        assert!(histogram.contains_key(&'A'));
        assert!(histogram.contains_key(&'B'));
        assert!(histogram.contains_key(&'C'));

        let total: usize = histogram.values().sum();
        assert_eq!(total, 300);
    }
}
