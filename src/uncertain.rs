use crate::computation::{ComputationNode, SampleContext};
use crate::operations::Arithmetic;
use crate::traits::Shareable;
use std::sync::Arc;

/// A type that represents uncertain data as a probability distribution
/// using sampling-based computation with conditional semantics.
///
/// `Uncertain` provides a way to work with probabilistic values
/// by representing them as sampling functions with a computation graph
/// for lazy evaluation and proper uncertainty-aware conditionals.
#[derive(Clone)]
pub struct Uncertain<T> {
    /// The sampling function that generates values from this distribution
    pub sample_fn: Arc<dyn Fn() -> T + Send + Sync>,
    /// The computation graph node for lazy evaluation
    pub(crate) node: ComputationNode<T>,
}

impl<T> Uncertain<T>
where
    T: Shareable,
{
    /// Creates an uncertain value with the given sampling function.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let custom = Uncertain::new(|| {
    ///     // Your custom sampling logic
    ///     rand::random::<f64>() * 10.0
    /// });
    /// ```
    pub fn new<F>(sampler: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let sampler = Arc::new(sampler);
        let node = ComputationNode::Leaf {
            id: uuid::Uuid::new_v4(),
            sample: sampler.clone(),
        };

        Self {
            sample_fn: sampler,
            node,
        }
    }

    /// Internal constructor with computation node for building computation graphs
    pub(crate) fn with_node(node: ComputationNode<T>) -> Self
    where
        T: Arithmetic,
    {
        let node_clone = node.clone();
        let sample_fn = Arc::new(move || {
            let mut context = SampleContext::new();
            node_clone.evaluate_conditional_with_arithmetic(&mut context)
        });

        Self { sample_fn, node }
    }

    /// Generate a sample from this distribution
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let sample = normal.sample();
    /// println!("Sample: {}", sample);
    /// ```
    #[must_use]
    pub fn sample(&self) -> T {
        (self.sample_fn)()
    }

    /// Transforms an uncertain value by applying a function to each sample.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let celsius = Uncertain::normal(20.0, 2.0);
    /// let fahrenheit = celsius.map(|c| c * 9.0/5.0 + 32.0);
    /// ```
    #[must_use]
    pub fn map<U, F>(&self, transform: F) -> Uncertain<U>
    where
        U: Shareable,
        F: Fn(T) -> U + Send + Sync + 'static,
    {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || transform(sample_fn()))
    }

    /// Transforms an uncertain value by applying a function that returns another uncertain value.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let base = Uncertain::normal(5.0, 1.0);
    /// let dependent = base.flat_map(|b| Uncertain::normal(b, 0.5));
    /// ```
    #[must_use]
    pub fn flat_map<U, F>(&self, transform: F) -> Uncertain<U>
    where
        U: Shareable,
        F: Fn(T) -> Uncertain<U> + Send + Sync + 'static,
    {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || transform(sample_fn()).sample())
    }

    /// Filters samples using rejection sampling.
    ///
    /// Only samples that satisfy the predicate are accepted.
    /// This method will keep sampling until a valid sample is found,
    /// so ensure the predicate has a reasonable acceptance rate.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let positive_only = normal.filter(|&x| x > 0.0);
    /// ```
    #[must_use]
    pub fn filter<F>(&self, predicate: F) -> Uncertain<T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || {
            loop {
                let value = sample_fn();
                if predicate(&value) {
                    return value;
                }
            }
        })
    }

    /// Generate an iterator of samples
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let first_10: Vec<f64> = normal.samples().take(10).collect();
    /// ```
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn samples(&self) -> impl Iterator<Item = T> + '_ {
        std::iter::repeat_with(|| self.sample())
    }

    /// Take a specific number of samples
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let uniform = Uncertain::uniform(0.0, 1.0);
    /// let samples = uniform.take_samples(1000);
    /// ```
    #[must_use]
    pub fn take_samples(&self, count: usize) -> Vec<T> {
        self.samples().take(count).collect()
    }
}

impl<T> Uncertain<T>
where
    T: Shareable + PartialOrd,
{
    /// Compare this uncertain value with another, returning an uncertain boolean
    #[must_use]
    pub fn less_than(&self, other: &Self) -> Uncertain<bool> {
        let self_fn = self.sample_fn.clone();
        let other_fn = other.sample_fn.clone();

        Uncertain::new(move || {
            let a = self_fn();
            let b = other_fn();
            a < b
        })
    }

    /// Compare this uncertain value with another, returning an uncertain boolean
    #[must_use]
    pub fn greater_than(&self, other: &Self) -> Uncertain<bool> {
        let self_fn = self.sample_fn.clone();
        let other_fn = other.sample_fn.clone();

        Uncertain::new(move || {
            let a = self_fn();
            let b = other_fn();
            a > b
        })
    }
}

impl<T> Uncertain<T>
where
    T: Shareable + PartialOrd + PartialEq + Copy,
{
    /// Returns uncertain boolean evidence that this value is greater than threshold
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let speed = Uncertain::normal(55.2, 5.0);
    /// let speeding_evidence = speed.gt(60.0);
    ///
    /// if speeding_evidence.probability_exceeds(0.95) {
    ///     println!("Issue speeding ticket");
    /// }
    /// ```
    #[must_use]
    pub fn gt(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() > threshold)
    }

    /// Returns uncertain boolean evidence that this value is less than threshold
    #[must_use]
    pub fn lt(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() < threshold)
    }

    /// Returns uncertain boolean evidence that this value is greater than or equal to threshold
    #[must_use]
    pub fn ge(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() >= threshold)
    }

    /// Returns uncertain boolean evidence that this value is less than or equal to threshold
    #[must_use]
    pub fn le(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() <= threshold)
    }

    /// Returns uncertain boolean evidence that this value equals threshold
    ///
    /// Note: For floating point types, exact equality is rarely meaningful.
    /// Consider using range-based comparisons instead.
    #[must_use]
    pub fn eq_value(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() == threshold)
    }

    /// Returns uncertain boolean evidence that this value does not equal threshold
    #[must_use]
    pub fn ne_value(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() != threshold)
    }
}

impl<T> std::cmp::PartialEq for Uncertain<T>
where
    T: Shareable + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // This is a fallback for direct equality testing
        let sample_a = self.sample();
        let sample_b = other.sample();
        sample_a == sample_b
    }
}

impl<T> std::cmp::PartialOrd for Uncertain<T>
where
    T: Shareable + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let sample_a = self.sample();
        let sample_b = other.sample();
        sample_a.partial_cmp(&sample_b)
    }

    fn lt(&self, other: &Self) -> bool {
        let sample_a = self.sample();
        let sample_b = other.sample();
        sample_a < sample_b
    }

    fn gt(&self, other: &Self) -> bool {
        let sample_a = self.sample();
        let sample_b = other.sample();
        sample_a > sample_b
    }
}

impl<T> std::fmt::Debug for Uncertain<T>
where
    T: Shareable + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Uncertain")
            .field("sample", &self.sample())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_uncertain() {
        let uncertain = Uncertain::new(|| 42.0_f64);
        assert!((uncertain.sample() - 42.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sample() {
        let uncertain = Uncertain::new(|| std::f64::consts::PI);
        assert!((uncertain.sample() - std::f64::consts::PI).abs() < f64::EPSILON);
        assert!((uncertain.sample() - std::f64::consts::PI).abs() < f64::EPSILON); // Should be consistent for deterministic sampler
    }

    #[test]
    fn test_map() {
        let uncertain = Uncertain::new(|| 5.0_f64);
        let mapped = uncertain.map(|x| x * 2.0);
        assert!((mapped.sample() - 10.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn test_map_type_conversion() {
        let uncertain = Uncertain::new(|| 5.0_f64);
        let mapped = uncertain.map(|x| x as i32);
        assert_eq!(mapped.sample(), 5);
    }

    #[test]
    fn test_flat_map() {
        let base = Uncertain::new(|| 3.0_f64);
        let dependent = base.flat_map(|x| Uncertain::new(move || x + 1.0));
        assert!((dependent.sample() - 4.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_flat_map_chain() {
        let base = Uncertain::new(|| 2.0_f64);
        let chained = base
            .flat_map(|x| Uncertain::new(move || x * 2.0))
            .flat_map(|x| Uncertain::new(move || x + 1.0));
        assert!((chained.sample() - 5.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filter() {
        let uncertain = Uncertain::new(|| 10.0);
        let filtered = uncertain.filter(|&x| x > 5.0);
        assert!(filtered.sample() > 5.0);
    }

    #[test]
    fn test_filter_rejection_sampling() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicI32, Ordering};
        let counter = Arc::new(AtomicI32::new(0));
        let counter_clone = counter.clone();
        let uncertain = Uncertain::new(move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count < 3 { 1.0 } else { 10.0 }
        });
        let filtered = uncertain.filter(|&x| x > 5.0);
        assert!(filtered.sample() > 5.0);
    }

    #[test]
    fn test_samples_iterator() {
        let uncertain = Uncertain::new(|| 42.0);
        let samples: Vec<f64> = uncertain.samples().take(5).collect();
        assert_eq!(samples, vec![42.0, 42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_take_samples() {
        let uncertain = Uncertain::new(|| 7.0);
        let samples = uncertain.take_samples(3);
        assert_eq!(samples, vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_take_samples_empty() {
        let uncertain = Uncertain::new(|| 1.0);
        let samples = uncertain.take_samples(0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_less_than() {
        let smaller = Uncertain::new(|| 1.0);
        let larger = Uncertain::new(|| 2.0);
        let comparison = smaller.less_than(&larger);
        assert!(comparison.sample());
    }

    #[test]
    fn test_less_than_false() {
        let larger = Uncertain::new(|| 2.0);
        let smaller = Uncertain::new(|| 1.0);
        let comparison = larger.less_than(&smaller);
        assert!(!comparison.sample());
    }

    #[test]
    fn test_greater_than() {
        let larger = Uncertain::new(|| 2.0);
        let smaller = Uncertain::new(|| 1.0);
        let comparison = larger.greater_than(&smaller);
        assert!(comparison.sample());
    }

    #[test]
    fn test_greater_than_false() {
        let smaller = Uncertain::new(|| 1.0);
        let larger = Uncertain::new(|| 2.0);
        let comparison = smaller.greater_than(&larger);
        assert!(!comparison.sample());
    }

    #[test]
    fn test_partial_eq() {
        let a = Uncertain::new(|| 5.0);
        let b = Uncertain::new(|| 5.0);
        let c = Uncertain::new(|| 10.0);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_partial_ord() {
        let smaller = Uncertain::new(|| 1.0);
        let larger = Uncertain::new(|| 2.0);

        assert!(smaller < larger);
        assert!(larger > smaller);
        assert!(smaller.partial_cmp(&larger).is_some());
    }

    #[test]
    fn test_partial_ord_equal() {
        let a = Uncertain::new(|| 5.0);
        let b = Uncertain::new(|| 5.0);

        assert!(a.partial_cmp(&b).is_some());
        assert!(b.partial_cmp(&a).is_some());
    }

    #[test]
    fn test_debug_formatting() {
        let uncertain = Uncertain::new(|| 42);
        let debug_str = format!("{uncertain:?}");
        assert!(debug_str.contains("Uncertain"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_clone() {
        let original = Uncertain::new(|| 123.0_f64);
        let cloned = original.clone();

        assert!((original.sample() - cloned.sample()).abs() < f64::EPSILON);
        assert!((original.sample() - 123.0_f64).abs() < f64::EPSILON);
        assert!((cloned.sample() - 123.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_random_sampler() {
        use rand::random;
        let uncertain = Uncertain::new(random::<f64>);

        // Should generate different values (with very high probability)
        let sample1 = uncertain.sample();
        let sample2 = uncertain.sample();
        // Very unlikely they'll be exactly equal for random f64
        assert!((0.0..=1.0).contains(&sample1));
        assert!((0.0..=1.0).contains(&sample2));
    }

    #[test]
    fn test_map_preserves_uncertainty() {
        use rand::random;
        let base = Uncertain::new(random::<f64>);
        let transformed = base.map(|x| x * 100.0);

        let sample = transformed.sample();
        assert!((0.0..=100.0).contains(&sample));
    }

    #[test]
    fn test_gt_method_api() {
        let speed = Uncertain::new(|| 65.0);
        let speeding_evidence = speed.gt(60.0);
        assert!(speeding_evidence.sample()); // 65 > 60
    }

    #[test]
    fn test_lt_method_api() {
        let temperature = Uncertain::new(|| -5.0);
        let freezing_evidence = temperature.lt(0.0);
        assert!(freezing_evidence.sample()); // -5 < 0
    }

    #[test]
    fn test_ge_method_api() {
        let value = Uncertain::new(|| 10.0);
        let evidence = value.ge(10.0);
        assert!(evidence.sample()); // 10 >= 10
    }

    #[test]
    fn test_le_method_api() {
        let value = Uncertain::new(|| 5.0);
        let evidence = value.le(10.0);
        assert!(evidence.sample()); // 5 <= 10
    }

    #[test]
    fn test_eq_value_method_api() {
        let value = Uncertain::new(|| 42);
        let evidence = value.eq_value(42);
        assert!(evidence.sample()); // 42 == 42
    }

    #[test]
    fn test_ne_value_method_api() {
        let value = Uncertain::new(|| 42);
        let evidence = value.ne_value(0);
        assert!(evidence.sample()); // 42 != 0
    }

    #[test]
    fn test_readme_example_api() {
        // Test the exact API shown in the README
        let speed = Uncertain::normal(55.2, 5.0);
        let speeding_evidence = speed.gt(60.0);

        // This should compile and work (the exact API from README)
        let _result = speeding_evidence.probability_exceeds(0.95);

        // Test with a value that's definitely over the threshold
        let high_speed = Uncertain::point(70.0);
        let high_speed_evidence = high_speed.gt(60.0);
        assert!(high_speed_evidence.probability_exceeds(0.95));
    }
}
