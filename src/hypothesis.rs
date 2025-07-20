use crate::Uncertain;

/// Result of hypothesis testing
#[derive(Debug, Clone)]
pub struct HypothesisResult {
    /// Whether the hypothesis was accepted (true) or rejected (false)
    pub decision: bool,
    /// Observed probability from samples
    pub probability: f64,
    /// Confidence level used in the test
    pub confidence_level: f64,
    /// Number of samples used to make the decision
    pub samples_used: usize,
}

/// Evidence-based conditional methods for uncertain boolean values
impl Uncertain<bool> {
    /// Evidence-based conditional using hypothesis testing
    ///
    /// This is the core method that implements the paper's key insight:
    /// ask about evidence, not boolean facts.
    ///
    /// # Arguments
    /// * `threshold` - Probability threshold to exceed
    /// * `confidence_level` - Confidence level for the test (default: 0.95)
    /// * `max_samples` - Maximum number of samples to use (default: 10000)
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::{Uncertain, operations::Comparison};
    ///
    /// let speed = Uncertain::normal(55.0, 5.0);
    /// let speeding_evidence = speed.gt(60.0);
    ///
    /// // Only issue ticket if 95% confident speeding
    /// if speeding_evidence.probability_exceeds(0.95) {
    ///     println!("Issue speeding ticket");
    /// }
    /// ```
    pub fn probability_exceeds(&self, threshold: f64) -> bool {
        self.probability_exceeds_with_params(threshold, 0.95, 10000)
    }

    /// Evidence-based conditional with configurable parameters
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let condition = Uncertain::bernoulli(0.7);
    /// let confident = condition.probability_exceeds_with_params(0.6, 0.99, 5000);
    /// ```
    pub fn probability_exceeds_with_params(
        &self,
        threshold: f64,
        confidence_level: f64,
        max_samples: usize,
    ) -> bool {
        let result = self.evaluate_hypothesis(
            threshold,
            confidence_level,
            max_samples,
            None, // epsilon
            None, // alpha
            None, // beta
            10,   // batch_size
        );
        result.decision
    }

    /// Implicit conditional (equivalent to probability_exceeds(0.5))
    ///
    /// This provides a convenient way to use uncertain booleans in if statements
    /// while still respecting the uncertainty.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::{Uncertain, operations::Comparison};
    ///
    /// let measurement = Uncertain::normal(10.0, 2.0);
    /// let above_threshold = measurement.gt(8.0);
    ///
    /// if above_threshold.implicit_conditional() {
    ///     println!("More likely than not above threshold");
    /// }
    /// ```
    pub fn implicit_conditional(&self) -> bool {
        self.probability_exceeds(0.5)
    }

    /// Performs Sequential Probability Ratio Test (SPRT) for efficient hypothesis testing
    ///
    /// This implements the SPRT algorithm described in the paper for efficient
    /// evidence evaluation with automatic sample size determination.
    ///
    /// # Arguments
    /// * `threshold` - Probability threshold for H1: P(true) > threshold
    /// * `confidence_level` - Overall confidence level
    /// * `max_samples` - Maximum samples before fallback decision
    /// * `epsilon` - Indifference region size (default: 0.05)
    /// * `alpha` - Type I error rate (false positive, default: 1 - confidence_level)
    /// * `beta` - Type II error rate (false negative, default: alpha)
    /// * `batch_size` - Samples to process in each batch (default: 10)
    ///
    /// # Returns
    /// `HypothesisResult` containing the decision and test statistics
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let biased_coin = Uncertain::bernoulli(0.7);
    /// let result = biased_coin.evaluate_hypothesis(
    ///     0.6,      // threshold
    ///     0.95,     // confidence_level
    ///     5000,     // max_samples
    ///     Some(0.05), // epsilon
    ///     Some(0.01), // alpha (Type I error)
    ///     Some(0.01), // beta (Type II error)
    ///     20,       // batch_size
    /// );
    ///
    /// println!("Decision: {}", result.decision);
    /// println!("Probability: {:.3}", result.probability);
    /// println!("Samples used: {}", result.samples_used);
    /// ```
    pub fn evaluate_hypothesis(
        &self,
        threshold: f64,
        confidence_level: f64,
        max_samples: usize,
        epsilon: Option<f64>,
        alpha: Option<f64>,
        beta: Option<f64>,
        batch_size: usize,
    ) -> HypothesisResult {
        let epsilon = epsilon.unwrap_or(0.05);
        let alpha_error = alpha.unwrap_or(1.0 - confidence_level);
        let beta_error = beta.unwrap_or(alpha_error);

        // Calculate SPRT boundaries
        let a = (beta_error / (1.0 - alpha_error)).ln();
        let b = ((1.0 - beta_error) / alpha_error).ln();

        // Set indifference region (avoiding edge cases)
        let p0 = (threshold - epsilon).max(0.001).min(0.999);
        let p1 = (threshold + epsilon).max(0.001).min(0.999);

        let mut successes = 0;
        let mut samples = 0;

        while samples < max_samples {
            // Batch sampling for efficiency
            let current_batch_size = (max_samples - samples).min(batch_size);
            let mut batch_successes = 0;

            for _ in 0..current_batch_size {
                if self.sample() {
                    batch_successes += 1;
                }
            }

            successes += batch_successes;
            samples += current_batch_size;

            // Compute log-likelihood ratio (LLR)
            let n = samples as f64;
            let x = successes as f64;

            // Avoid log(0) by clamping probabilities
            let p0_clamped = p0.max(1e-10).min(1.0 - 1e-10);
            let p1_clamped = p1.max(1e-10).min(1.0 - 1e-10);

            let llr = x * (p1_clamped / p0_clamped).ln()
                + (n - x) * ((1.0 - p1_clamped) / (1.0 - p0_clamped)).ln();

            if llr <= a {
                // Accept H0: P(true) <= threshold
                return HypothesisResult {
                    decision: false,
                    probability: successes as f64 / samples as f64,
                    confidence_level,
                    samples_used: samples,
                };
            } else if llr >= b {
                // Accept H1: P(true) > threshold
                return HypothesisResult {
                    decision: true,
                    probability: successes as f64 / samples as f64,
                    confidence_level,
                    samples_used: samples,
                };
            }
        }

        // Fallback decision based on observed probability
        let final_p = successes as f64 / samples as f64;
        HypothesisResult {
            decision: final_p > threshold,
            probability: final_p,
            confidence_level,
            samples_used: samples,
        }
    }

    /// Estimates the probability that this condition is true
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let condition = Uncertain::bernoulli(0.7);
    /// let prob = condition.estimate_probability(1000);
    /// // Should be approximately 0.7
    /// ```
    pub fn estimate_probability(&self, sample_count: usize) -> f64 {
        let samples: Vec<bool> = self.take_samples(sample_count);
        samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64
    }

    /// Bayesian evidence update using Bayes' theorem
    ///
    /// Updates the probability of this condition given observed evidence.
    ///
    /// # Arguments
    /// * `prior_prob` - Prior probability of the condition
    /// * `evidence` - Observed evidence (another uncertain boolean)
    /// * `likelihood_given_true` - P(evidence | condition is true)
    /// * `likelihood_given_false` - P(evidence | condition is false)
    /// * `sample_count` - Number of samples for estimation
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let disease = Uncertain::bernoulli(0.01); // 1% base rate
    /// let test_positive = Uncertain::bernoulli(0.95); // Test result
    ///
    /// let posterior = disease.bayesian_update(
    ///     0.01, // prior probability
    ///     &test_positive,
    ///     0.95, // sensitivity (true positive rate)
    ///     0.05, // false positive rate
    ///     1000
    /// );
    /// ```
    pub fn bayesian_update(
        &self,
        prior_prob: f64,
        evidence: &Uncertain<bool>,
        likelihood_given_true: f64,
        likelihood_given_false: f64,
        sample_count: usize,
    ) -> f64 {
        let evidence_prob = evidence.estimate_probability(sample_count);

        // Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
        // P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        let evidence_total =
            likelihood_given_true * prior_prob + likelihood_given_false * (1.0 - prior_prob);

        if evidence_total == 0.0 {
            return prior_prob;
        }

        if evidence_prob > 0.5 {
            // Evidence is present
            (likelihood_given_true * prior_prob) / evidence_total
        } else {
            // Evidence is absent
            ((1.0 - likelihood_given_true) * prior_prob) / (1.0 - evidence_total)
        }
    }
}

/// Sequential testing for multiple hypotheses
pub struct MultipleHypothesisTester {
    hypotheses: Vec<Uncertain<bool>>,
    names: Vec<String>,
}

impl MultipleHypothesisTester {
    /// Create a new multiple hypothesis tester
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::{Uncertain, hypothesis::MultipleHypothesisTester};
    ///
    /// let temp = Uncertain::normal(22.0, 2.0);
    /// let hypotheses = vec![
    ///     temp.gt(20.0),
    ///     temp.gt(25.0),
    ///     temp.lt(18.0),
    /// ];
    /// let names = vec!["warm", "hot", "cold"];
    ///
    /// let tester = MultipleHypothesisTester::new(hypotheses, names);
    /// ```
    pub fn new(hypotheses: Vec<Uncertain<bool>>, names: Vec<&str>) -> Self {
        Self {
            hypotheses,
            names: names.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Test all hypotheses and return results
    ///
    /// Uses Bonferroni correction to control family-wise error rate.
    pub fn test_all(
        &self,
        overall_alpha: f64,
        max_samples: usize,
    ) -> Vec<(String, HypothesisResult)> {
        let corrected_alpha = overall_alpha / self.hypotheses.len() as f64;
        let confidence_level = 1.0 - corrected_alpha;

        self.hypotheses
            .iter()
            .zip(self.names.iter())
            .map(|(hypothesis, name)| {
                let result = hypothesis.evaluate_hypothesis(
                    0.5,
                    confidence_level,
                    max_samples,
                    None,
                    Some(corrected_alpha),
                    None,
                    10,
                );
                (name.clone(), result)
            })
            .collect()
    }

    /// Find the hypothesis with the highest probability
    pub fn find_most_likely(&self, sample_count: usize) -> Option<(String, f64)> {
        let mut best_name = None;
        let mut best_prob = 0.0;

        for (hypothesis, name) in self.hypotheses.iter().zip(self.names.iter()) {
            let prob = hypothesis.estimate_probability(sample_count);
            if prob > best_prob {
                best_prob = prob;
                best_name = Some(name.clone());
            }
        }

        best_name.map(|name| (name, best_prob))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::Comparison;

    #[test]
    fn test_probability_exceeds() {
        let always_true = Uncertain::point(true);
        let always_false = Uncertain::point(false);

        assert!(always_true.probability_exceeds(0.5));
        assert!(always_true.probability_exceeds(0.95));
        assert!(!always_false.probability_exceeds(0.05));
        assert!(!always_false.probability_exceeds(0.95));
    }

    #[test]
    fn test_implicit_conditional() {
        let likely_true = Uncertain::bernoulli(0.8);
        let likely_false = Uncertain::bernoulli(0.2);

        // These are probabilistic, so we test multiple times
        let mut true_count = 0;
        let mut false_count = 0;

        for _ in 0..10 {
            if likely_true.implicit_conditional() {
                true_count += 1;
            }
            if likely_false.implicit_conditional() {
                false_count += 1;
            }
        }

        // Should be mostly true and mostly false respectively
        assert!(true_count > false_count);
    }

    #[test]
    fn test_hypothesis_testing_with_known_probability() {
        let biased_coin = Uncertain::bernoulli(0.7);

        // Test exceeds 0.6 (should be true)
        let result1 = biased_coin.evaluate_hypothesis(0.6, 0.95, 5000, None, None, None, 20);
        assert!(result1.decision);
        assert!((result1.probability - 0.7).abs() < 0.15);

        // Test exceeds 0.8 (should be false)
        let result2 = biased_coin.evaluate_hypothesis(0.8, 0.95, 5000, None, None, None, 20);
        assert!(!result2.decision);
    }

    #[test]
    fn test_sprt_efficiency() {
        // Clear cases should be decided quickly
        let certain_true = Uncertain::point(true);
        let result = certain_true.evaluate_hypothesis(0.5, 0.95, 10000, None, None, None, 10);

        assert!(result.decision);
        assert!(result.samples_used < 100); // Should decide quickly
    }

    #[test]
    fn test_evidence_based_conditionals() {
        let speed = Uncertain::normal(55.0, 5.0);
        let speeding_evidence = speed.gt(60.0);

        // With mean=55, std=5, P(X > 60) should be relatively low
        let high_confidence = speeding_evidence.probability_exceeds(0.95);

        // These tests are probabilistic but should generally hold
        assert!(!high_confidence); // Very unlikely to be 95% confident
    }

    #[test]
    fn test_bayesian_update() {
        let disease = Uncertain::bernoulli(0.01);
        let test_positive = Uncertain::bernoulli(0.95);

        let posterior = disease.bayesian_update(
            0.01, // prior: 1% disease rate
            &test_positive,
            0.95, // sensitivity
            0.05, // false positive rate
            1000,
        );

        // Posterior should be higher than prior given positive test
        assert!(posterior > 0.01);
    }

    #[test]
    fn test_multiple_hypothesis_testing() {
        let temp = Uncertain::normal(22.0, 2.0);
        let hypotheses = vec![
            temp.gt(20.0), // Should be likely true
            temp.gt(30.0), // Should be false
            temp.lt(15.0), // Should be false
        ];
        let names = vec!["warm", "hot", "cold"];

        let tester = MultipleHypothesisTester::new(hypotheses, names);
        let results = tester.test_all(0.05, 1000);

        assert_eq!(results.len(), 3);

        // First hypothesis (warm) should likely be true
        assert_eq!(results[0].0, "warm");
        // Can't guarantee the decision due to randomness, but we can check structure
        assert!(results[0].1.samples_used > 0);
    }

    #[test]
    fn test_find_most_likely() {
        let temp = Uncertain::normal(22.0, 1.0);
        let hypotheses = vec![
            temp.gt(25.0), // Unlikely
            temp.gt(20.0), // Very likely
            temp.lt(18.0), // Unlikely
        ];
        let names = vec!["hot", "warm", "cold"];

        let tester = MultipleHypothesisTester::new(hypotheses, names);
        let most_likely = tester.find_most_likely(1000);

        // Should find "warm" as most likely
        assert!(most_likely.is_some());
        let (name, prob) = most_likely.unwrap();
        assert_eq!(name, "warm");
        assert!(prob > 0.8); // Should have high probability
    }
}
