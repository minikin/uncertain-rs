#![allow(clippy::cast_precision_loss)]

use crate::Uncertain;
use crate::traits::Shareable;

/// Trait for comparison operations that return uncertain boolean evidence
///
/// The key insight from the paper: comparisons return `Uncertain<bool>` (evidence),
/// not `bool` (boolean facts). This prevents uncertainty bugs.
///
/// **Note**: For most use cases, prefer the method-based API (e.g., `value.gt(threshold)`)
/// over the trait-based API (e.g., `Comparison::gt(&value, threshold)`). The trait is
/// primarily useful for generic programming and advanced scenarios.
pub trait Comparison<T> {
    /// Returns uncertain boolean evidence that this value is greater than threshold
    #[must_use]
    fn gt(&self, threshold: T) -> Uncertain<bool>;

    /// Returns uncertain boolean evidence that this value is less than threshold
    #[must_use]
    fn lt(&self, threshold: T) -> Uncertain<bool>;

    /// Returns uncertain boolean evidence that this value is greater than or equal to threshold
    #[must_use]
    fn ge(&self, threshold: T) -> Uncertain<bool>;

    /// Returns uncertain boolean evidence that this value is less than or equal to threshold
    #[must_use]
    fn le(&self, threshold: T) -> Uncertain<bool>;

    /// Returns uncertain boolean evidence that this value equals threshold
    #[must_use]
    fn eq(&self, threshold: T) -> Uncertain<bool>;

    /// Returns uncertain boolean evidence that this value does not equal threshold
    #[must_use]
    fn ne(&self, threshold: T) -> Uncertain<bool>;
}

impl<T> Comparison<T> for Uncertain<T>
where
    T: PartialOrd + PartialEq + Shareable,
{
    /// Greater than comparison
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let speed = Uncertain::normal(55.0, 5.0).unwrap();
    /// let speeding_evidence = speed.gt(60.0);
    ///
    /// if speeding_evidence.probability_exceeds(0.95) {
    ///     println!("95% confident speeding");
    /// }
    /// ```
    fn gt(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() > threshold)
    }

    /// Less than comparison
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let temperature = Uncertain::normal(1.0, 2.0).unwrap();
    /// let freezing_evidence = temperature.lt(0.0);
    ///
    /// if freezing_evidence.probability_exceeds(0.8) {
    ///     println!("Likely freezing");
    /// }
    /// ```
    fn lt(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() < threshold)
    }

    /// Greater than or equal comparison
    fn ge(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() >= threshold)
    }

    /// Less than or equal comparison
    fn le(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() <= threshold)
    }

    /// Equality comparison
    ///
    /// Note: For floating point types, exact equality is rarely meaningful.
    /// Consider using range-based comparisons instead.
    fn eq(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() == threshold)
    }

    /// Inequality comparison
    fn ne(&self, threshold: T) -> Uncertain<bool> {
        let sample_fn = self.sample_fn.clone();
        Uncertain::new(move || sample_fn() != threshold)
    }
}

// Comparisons between two uncertain values
impl<T> Uncertain<T>
where
    T: PartialOrd + PartialEq + Shareable,
{
    /// Compare two uncertain values for greater than
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let sensor1 = Uncertain::normal(10.0, 1.0).unwrap();
    /// let sensor2 = Uncertain::normal(12.0, 1.0).unwrap();
    /// let evidence = sensor2.gt_uncertain(&sensor1);
    /// ```
    #[must_use]
    pub fn gt_uncertain(&self, other: &Self) -> Uncertain<bool> {
        let sample_fn1 = self.sample_fn.clone();
        let sample_fn2 = other.sample_fn.clone();
        Uncertain::new(move || sample_fn1() > sample_fn2())
    }

    /// Compare two uncertain values for less than
    #[must_use]
    pub fn lt_uncertain(&self, other: &Self) -> Uncertain<bool> {
        let sample_fn1 = self.sample_fn.clone();
        let sample_fn2 = other.sample_fn.clone();
        Uncertain::new(move || sample_fn1() < sample_fn2())
    }

    /// Compare two uncertain values for equality
    #[must_use]
    pub fn eq_uncertain(&self, other: &Self) -> Uncertain<bool> {
        let sample_fn1 = self.sample_fn.clone();
        let sample_fn2 = other.sample_fn.clone();
        Uncertain::new(move || sample_fn1() == sample_fn2())
    }
}

// Floating point specific comparisons
impl Uncertain<f64> {
    /// Check if value is approximately equal within tolerance
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let measurement = Uncertain::normal(10.0, 0.1).unwrap();
    /// let target = 10.0;
    /// let tolerance = 0.5;
    ///
    /// let close_evidence = measurement.approx_eq(target, tolerance);
    /// ```
    #[must_use]
    pub fn approx_eq(&self, target: f64, tolerance: f64) -> Uncertain<bool> {
        self.map(move |x| (x - target).abs() <= tolerance)
    }

    /// Check if value is within a range
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let measurement = Uncertain::normal(10.0, 2.0).unwrap();
    /// let in_range = measurement.within_range(8.0, 12.0);
    /// ```
    #[must_use]
    pub fn within_range(&self, min: f64, max: f64) -> Uncertain<bool> {
        self.map(move |x| x >= min && x <= max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_returns_uncertain_bool() {
        let value = Uncertain::point(5.0);
        let evidence = Comparison::gt(&value, 3.0);

        // Should return Uncertain<bool>, not bool
        assert!(evidence.sample()); // 5 > 3 is always true
    }

    #[test]
    fn test_comparison_with_uncertainty() {
        let value = Uncertain::normal(5.0, 1.0).unwrap();
        let evidence = Comparison::gt(&value, 4.0);

        // With normal(5, 1), most samples should be > 4
        let samples: Vec<bool> = evidence.take_samples(1000);
        let true_ratio = samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64;
        assert!(true_ratio > 0.8); // Should be high probability
    }

    #[test]
    fn test_approximate_equality() {
        let measurement = Uncertain::normal(10.0, 0.1).unwrap();
        let close = measurement.approx_eq(10.0, 0.5);

        // With small std dev, should almost always be close to 10
        let samples: Vec<bool> = close.take_samples(100);
        let true_ratio = samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64;
        assert!(true_ratio > 0.95);
    }

    #[test]
    fn test_within_range() {
        let value = Uncertain::uniform(0.0, 10.0).unwrap();
        let in_range = value.within_range(2.0, 8.0);

        // About 60% of uniform[0,10] should be in [2,8]
        let samples: Vec<bool> = in_range.take_samples(1000);
        let true_ratio = samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64;
        assert!((true_ratio - 0.6).abs() < 0.1);
    }

    #[test]
    fn test_uncertain_vs_uncertain_comparison() {
        let x = Uncertain::normal(5.0, 1.0).unwrap();
        let y = Uncertain::normal(3.0, 1.0).unwrap();
        let evidence = x.gt_uncertain(&y);

        // x should usually be greater than y
        let samples: Vec<bool> = evidence.take_samples(1000);
        let true_ratio = samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64;
        assert!(true_ratio > 0.8);
    }

    #[test]
    fn test_comparison_trait_gt_boundary() {
        let value = Uncertain::point(5.0);
        assert!(!Comparison::gt(&value, 5.0).sample());
        assert!(Comparison::gt(&value, 4.0).sample());
        assert!(!Comparison::gt(&value, 6.0).sample());
    }

    #[test]
    fn test_gt_uncertain_boundary() {
        let equal_a = Uncertain::point(5.0);
        let equal_b = Uncertain::point(5.0);
        assert!(!equal_a.gt_uncertain(&equal_b).sample());

        let larger = Uncertain::point(5.0);
        let smaller = Uncertain::point(3.0);
        assert!(larger.gt_uncertain(&smaller).sample());
        assert!(!smaller.gt_uncertain(&larger).sample());
    }

    #[test]
    fn test_comparison_trait_lt_boundary() {
        let value = Uncertain::point(5.0);
        assert!(Comparison::lt(&value, 6.0).sample());
        assert!(!Comparison::lt(&value, 5.0).sample());
        assert!(!Comparison::lt(&value, 4.0).sample());
    }

    #[test]
    fn test_comparison_trait_ge_boundary() {
        let value = Uncertain::point(5.0);
        assert!(Comparison::ge(&value, 5.0).sample());
        assert!(Comparison::ge(&value, 4.0).sample());
        assert!(!Comparison::ge(&value, 6.0).sample());
    }

    #[test]
    fn test_comparison_trait_le_boundary() {
        let value = Uncertain::point(5.0);
        assert!(Comparison::le(&value, 5.0).sample());
        assert!(Comparison::le(&value, 6.0).sample());
        assert!(!Comparison::le(&value, 4.0).sample());
    }

    #[test]
    fn test_comparison_trait_eq_and_ne() {
        let value = Uncertain::point(5.0);
        assert!(Comparison::eq(&value, 5.0).sample());
        assert!(!Comparison::eq(&value, 4.0).sample());
        assert!(Comparison::ne(&value, 4.0).sample());
        assert!(!Comparison::ne(&value, 5.0).sample());
    }

    #[test]
    fn test_lt_uncertain_boundary() {
        let equal_a = Uncertain::point(5.0);
        let equal_b = Uncertain::point(5.0);
        assert!(!equal_a.lt_uncertain(&equal_b).sample());

        let smaller = Uncertain::point(3.0);
        let larger = Uncertain::point(5.0);
        assert!(smaller.lt_uncertain(&larger).sample());
        assert!(!larger.lt_uncertain(&smaller).sample());
    }

    #[test]
    fn test_eq_uncertain_boundary() {
        let a = Uncertain::point(5.0);
        let b = Uncertain::point(5.0);
        let c = Uncertain::point(6.0);
        assert!(a.eq_uncertain(&b).sample());
        assert!(!a.eq_uncertain(&c).sample());
    }
}
