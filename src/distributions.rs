#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::Uncertain;
use crate::error::{Result, UncertainError};
use crate::rng::{random_f64, with_rng};
use crate::traits::Shareable;
use rand::prelude::*;
use rand_distr::{
    Bernoulli, Beta, Binomial, Exp, Gamma, Geometric, LogNormal, Normal, Poisson, Uniform,
};
use std::collections::HashMap;

fn validate_finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(UncertainError::non_finite(name, value))
    }
}

fn validate_positive(name: &'static str, value: f64) -> Result<()> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(UncertainError::invalid_parameter(
            name,
            value,
            "must be positive",
        ))
    }
}

fn validate_non_negative(name: &'static str, value: f64) -> Result<()> {
    validate_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(UncertainError::invalid_parameter(
            name,
            value,
            "must be non-negative",
        ))
    }
}

fn validate_unit_interval(name: &'static str, value: f64) -> Result<()> {
    validate_finite(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(UncertainError::invalid_parameter(
            name,
            value,
            "must be in [0, 1]",
        ))
    }
}

fn validate_left_open_unit_interval(name: &'static str, value: f64) -> Result<()> {
    validate_finite(name, value)?;
    if value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(UncertainError::invalid_parameter(
            name,
            value,
            "must be in (0, 1]",
        ))
    }
}

impl<T> Uncertain<T>
where
    T: Shareable,
{
    /// Creates a point-mass distribution (certain value)
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let certain_value = Uncertain::point(42.0);
    /// assert_eq!(certain_value.sample(), 42.0);
    /// ```
    #[must_use]
    pub fn point(value: T) -> Self {
        Uncertain::new(move || value.clone())
    }

    /// Creates a mixture of distributions with optional weights
    ///
    /// # Arguments
    /// * `components` - Vector of distributions to mix
    /// * `weights` - Optional weights for each component (uniform if None)
    ///
    /// # Errors
    /// Returns an error if the components vector is empty or if the weights count
    /// doesn't match the components count.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal1 = Uncertain::normal(0.0, 1.0).unwrap();
    /// let normal2 = Uncertain::normal(5.0, 1.0).unwrap();
    /// let mixture = Uncertain::mixture(
    ///     vec![normal1, normal2],
    ///     Some(vec![0.7, 0.3])
    /// ).unwrap();
    /// ```
    pub fn mixture(components: Vec<Uncertain<T>>, weights: Option<Vec<f64>>) -> Result<Self> {
        if components.is_empty() {
            return Err(UncertainError::EmptyComponents);
        }

        if components.len() == 1 {
            return Ok(components.into_iter().next().unwrap());
        }

        let weights = match weights {
            Some(w) => {
                if w.len() != components.len() {
                    return Err(UncertainError::weight_mismatch(components.len(), w.len()));
                }
                w
            }
            None => vec![1.0; components.len()],
        };

        let total: f64 = weights.iter().sum();
        let normalized: Vec<f64> = weights.iter().map(|&w| w / total).collect();
        let cumulative: Vec<f64> = normalized
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        Ok(Uncertain::new(move || {
            let r: f64 = random_f64();
            let idx = cumulative
                .iter()
                .position(|&x| r <= x)
                .unwrap_or_else(|| components.len().saturating_sub(1));
            components.get(idx).map_or_else(
                || components[0].sample(),
                super::uncertain::Uncertain::sample,
            )
        }))
    }

    /// Creates an empirical distribution from observed data
    ///
    /// # Arguments
    /// * `data` - Vector of observed data points
    ///
    /// # Errors
    /// Returns an error if the data vector is empty.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let empirical = Uncertain::empirical(data).unwrap();
    /// ```
    pub fn empirical(data: Vec<T>) -> Result<Self> {
        if data.is_empty() {
            return Err(UncertainError::EmptyData);
        }

        Ok(Uncertain::new(move || {
            with_rng(|rng| data.choose(rng).cloned().unwrap_or_else(|| data[0].clone()))
        }))
    }
}

// Categorical distribution for hashable types
impl<T> Uncertain<T>
where
    T: Clone + Send + Sync + std::hash::Hash + Eq + 'static,
{
    /// Creates a categorical distribution from value-probability pairs
    ///
    /// # Arguments
    /// * `probabilities` - A map of values to their probabilities
    ///
    /// # Errors
    /// Returns an error if the probabilities map is empty.
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
    /// let color = Uncertain::categorical(&probs).unwrap();
    /// ```
    pub fn categorical(probabilities: &HashMap<T, f64>) -> Result<Self> {
        if probabilities.is_empty() {
            return Err(UncertainError::EmptyProbabilities);
        }

        let total: f64 = probabilities.values().sum();
        let mut cumulative = Vec::new();
        let mut sum = 0.0;

        for (value, &prob) in probabilities {
            sum += prob / total;
            cumulative.push((value.clone(), sum));
        }

        Ok(Uncertain::new(move || {
            let r: f64 = random_f64();
            cumulative.iter().find(|(_, cum)| r <= *cum).map_or_else(
                || cumulative[cumulative.len() - 1].0.clone(),
                |(val, _)| val.clone(),
            )
        }))
    }
}

// Floating point distributions
impl Uncertain<f64> {
    /// Creates a normal (Gaussian) distribution
    ///
    /// # Arguments
    /// * `mean` - The mean of the distribution
    /// * `std_dev` - The standard deviation (must be non-negative; `0.0` degenerates to a
    ///   point mass at `mean`)
    ///
    /// # Errors
    /// Returns an error if `mean` or `std_dev` is not finite, or if `std_dev` is negative.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0).unwrap(); // Standard normal
    /// let measurement = Uncertain::normal(100.0, 5.0).unwrap(); // Measurement with error
    /// ```
    pub fn normal(mean: f64, std_dev: f64) -> Result<Self> {
        validate_finite("mean", mean)?;
        validate_non_negative("std_dev", std_dev)?;
        Ok(Self::normal_unchecked(mean, std_dev))
    }

    fn normal_unchecked(mean: f64, std_dev: f64) -> Self {
        let dist = Normal::new(mean, std_dev)
            .expect("mean finite and std_dev >= 0 already validated by normal()/log_normal()");
        Uncertain::new(move || with_rng(|rng| dist.sample(rng)))
    }

    /// Creates a uniform distribution
    ///
    /// # Arguments
    /// * `min` - The lower bound (inclusive)
    /// * `max` - The upper bound (inclusive); must be `>= min`. `min == max` degenerates
    ///   to a point mass.
    ///
    /// # Errors
    /// Returns an error if `min` or `max` is not finite, or if `min > max`.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let uniform = Uncertain::uniform(0.0, 10.0).unwrap();
    /// ```
    pub fn uniform(min: f64, max: f64) -> Result<Self> {
        validate_finite("min", min)?;
        validate_finite("max", max)?;
        if min > max {
            return Err(UncertainError::invalid_parameter(
                "min",
                min,
                "must be less than or equal to max",
            ));
        }
        let dist = Uniform::new_inclusive(min, max).expect("min <= max already validated above");
        Ok(Uncertain::new(move || with_rng(|rng| dist.sample(rng))))
    }

    /// Creates an exponential distribution
    ///
    /// # Arguments
    /// * `rate` - The rate parameter (lambda), must be positive
    ///
    /// # Errors
    /// Returns an error if `rate` is not finite or not positive.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let exponential = Uncertain::exponential(1.0).unwrap();
    /// ```
    pub fn exponential(rate: f64) -> Result<Self> {
        validate_positive("rate", rate)?;
        let dist = Exp::new(rate).expect("rate > 0 already validated above");
        Ok(Uncertain::new(move || with_rng(|rng| dist.sample(rng))))
    }

    /// Creates a log-normal distribution
    ///
    /// # Arguments
    /// * `mu` - Mean of the underlying normal distribution
    /// * `sigma` - Standard deviation of the underlying normal distribution (must be
    ///   non-negative; `0.0` degenerates to a point mass at `exp(mu)`)
    ///
    /// # Errors
    /// Returns an error if `mu` or `sigma` is not finite, or if `sigma` is negative.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let lognormal = Uncertain::log_normal(0.0, 1.0).unwrap();
    /// ```
    pub fn log_normal(mu: f64, sigma: f64) -> Result<Self> {
        validate_finite("mu", mu)?;
        validate_non_negative("sigma", sigma)?;
        let dist =
            LogNormal::new(mu, sigma).expect("mu finite and sigma >= 0 already validated above");
        Ok(Uncertain::new(move || with_rng(|rng| dist.sample(rng))))
    }

    /// Creates a beta distribution
    ///
    /// # Arguments
    /// * `alpha` - First shape parameter, must be positive
    /// * `beta` - Second shape parameter, must be positive
    ///
    /// # Errors
    /// Returns an error if `alpha` or `beta` is not finite or not positive.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let beta = Uncertain::beta(2.0, 5.0).unwrap();
    /// ```
    pub fn beta(alpha: f64, beta: f64) -> Result<Self> {
        validate_positive("alpha", alpha)?;
        validate_positive("beta", beta)?;
        let dist = Beta::new(alpha, beta).expect("alpha > 0 and beta > 0 already validated above");
        Ok(Uncertain::new(move || with_rng(|rng| dist.sample(rng))))
    }

    /// Creates a gamma distribution
    ///
    /// # Arguments
    /// * `shape` - Shape parameter (alpha), must be positive
    /// * `scale` - Scale parameter (beta), must be non-negative (`0.0` degenerates to a
    ///   point mass at `0.0`)
    ///
    /// # Errors
    /// Returns an error if `shape` is not finite or not positive, or if `scale` is not
    /// finite or negative.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let gamma = Uncertain::gamma(2.0, 1.0).unwrap();
    /// ```
    pub fn gamma(shape: f64, scale: f64) -> Result<Self> {
        validate_positive("shape", shape)?;
        validate_non_negative("scale", scale)?;
        Ok(Self::gamma_unchecked(shape, scale))
    }

    fn gamma_unchecked(shape: f64, scale: f64) -> Self {
        // rand_distr::Gamma requires scale > 0 strictly; scale == 0 is a legitimate
        // degenerate point mass at 0 this crate has supported since spec 02, so it's
        // handled directly rather than passed to rand_distr.
        if scale == 0.0 {
            return Uncertain::new(|| 0.0);
        }
        let dist =
            Gamma::new(shape, scale).expect("shape > 0 and scale > 0 already validated above");
        Uncertain::new(move || with_rng(|rng| dist.sample(rng)))
    }
}

// Boolean distributions
impl Uncertain<bool> {
    /// Creates a Bernoulli distribution
    ///
    /// # Arguments
    /// * `probability` - Probability of success (true), must be in `[0, 1]`
    ///
    /// # Errors
    /// Returns an error if `probability` is not finite or outside `[0, 1]`.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let biased_coin = Uncertain::bernoulli(0.7).unwrap(); // 70% chance of true
    /// ```
    pub fn bernoulli(probability: f64) -> Result<Self> {
        validate_unit_interval("probability", probability)?;
        let dist =
            Bernoulli::new(probability).expect("probability in [0, 1] already validated above");
        Ok(Uncertain::new(move || with_rng(|rng| dist.sample(rng))))
    }
}

// Integer distributions
impl<T> Uncertain<T>
where
    T: Clone
        + Send
        + Sync
        + From<u32>
        + std::ops::AddAssign
        + std::ops::Sub<Output = T>
        + Default
        + 'static,
{
    /// Creates a binomial distribution
    ///
    /// # Arguments
    /// * `trials` - Number of trials
    /// * `probability` - Probability of success on each trial, must be in `[0, 1]`
    ///
    /// # Errors
    /// Returns an error if `probability` is not finite or outside `[0, 1]`.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let binomial: Uncertain<u32> = Uncertain::binomial(100, 0.3).unwrap();
    /// ```
    pub fn binomial(trials: u32, probability: f64) -> Result<Self> {
        validate_unit_interval("probability", probability)?;
        let dist = Binomial::new(u64::from(trials), probability)
            .expect("probability in [0, 1] already validated above");
        Ok(Uncertain::new(move || {
            let count: u64 = with_rng(|rng| dist.sample(rng));
            // count is a number of successes out of `trials` (a u32), so it always
            // fits back into a u32 without truncation.
            T::from(count as u32)
        }))
    }

    /// Creates a Poisson distribution
    ///
    /// # Arguments
    /// * `lambda` - Rate parameter, must be non-negative (`0.0` degenerates to a point
    ///   mass at `0`) and at most [`Poisson::MAX_LAMBDA`] (~1.844e19)
    ///
    /// # Errors
    /// Returns an error if `lambda` is not finite, negative, or exceeds
    /// [`Poisson::MAX_LAMBDA`].
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let poisson: Uncertain<u32> = Uncertain::poisson(3.5).unwrap();
    /// ```
    pub fn poisson(lambda: f64) -> Result<Self> {
        validate_non_negative("lambda", lambda)?;
        if lambda > Poisson::<f64>::MAX_LAMBDA {
            return Err(UncertainError::invalid_parameter(
                "lambda",
                lambda,
                "must not exceed Poisson::MAX_LAMBDA (~1.844e19)",
            ));
        }
        // rand_distr::Poisson requires lambda > 0 strictly; lambda == 0 is a legitimate
        // degenerate point mass at 0 this crate has supported since spec 02.
        if lambda == 0.0 {
            return Ok(Uncertain::new(|| T::from(0)));
        }
        let dist = Poisson::new(lambda)
            .expect("lambda in (0, Poisson::MAX_LAMBDA] already validated above");
        Ok(Uncertain::new(move || {
            let count: f64 = with_rng(|rng| dist.sample(rng));
            T::from(count.round() as u32)
        }))
    }

    /// Creates a geometric distribution
    ///
    /// # Arguments
    /// * `probability` - Probability of success on each trial, must be in `(0, 1]`
    ///   (`0.0` would never terminate)
    ///
    /// # Errors
    /// Returns an error if `probability` is not finite or outside `(0, 1]`.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let geometric: Uncertain<u32> = Uncertain::geometric(0.1).unwrap();
    /// ```
    pub fn geometric(probability: f64) -> Result<Self> {
        validate_left_open_unit_interval("probability", probability)?;
        let dist =
            Geometric::new(probability).expect("probability in (0, 1] already validated above");
        Ok(Uncertain::new(move || {
            let failures: u64 = with_rng(|rng| dist.sample(rng));
            // rand_distr::Geometric counts failures before the first success
            // (0-indexed). This crate documents "number of trials until first
            // success" (1-indexed, >= 1), matching its pre-rand_distr behavior.
            T::from((failures as u32) + 1)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_distribution() {
        let normal = Uncertain::normal(0.0, 1.0).unwrap();
        let samples: Vec<f64> = normal.take_samples(1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_uniform_distribution() {
        let uniform = Uncertain::uniform(0.0, 10.0).unwrap();
        let samples: Vec<f64> = uniform.take_samples(1000);
        assert!(samples.iter().all(|&x| (0.0..=10.0).contains(&x)));
    }

    #[test]
    fn test_bernoulli_distribution() {
        let bernoulli = Uncertain::bernoulli(0.7).unwrap();
        let samples: Vec<bool> = bernoulli.take_samples(1000);
        let true_ratio = samples.iter().filter(|&&x| x).count() as f64 / samples.len() as f64;
        assert!((true_ratio - 0.7).abs() < 0.1);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_point_mass() {
        let point = Uncertain::point(42.0);
        for _ in 0..10 {
            assert_eq!(point.sample(), 42.0);
        }
    }

    #[test]
    fn test_categorical() {
        let mut probs = HashMap::new();
        probs.insert("a", 0.5);
        probs.insert("b", 0.3);
        probs.insert("c", 0.2);

        let categorical = Uncertain::categorical(&probs).unwrap();
        let samples: Vec<&str> = categorical.take_samples(1000);

        assert!(samples.iter().all(|&x| ["a", "b", "c"].contains(&x)));
    }

    #[test]
    fn test_exponential_distribution() {
        let exponential = Uncertain::exponential(2.0).unwrap();
        let samples: Vec<f64> = exponential.take_samples(1000);

        assert!(samples.iter().all(|&x| x >= 0.0));

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_log_normal_distribution() {
        let log_normal = Uncertain::log_normal(0.0, 1.0).unwrap();
        let samples: Vec<f64> = log_normal.take_samples(1000);

        assert!(samples.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_beta_distribution() {
        let beta = Uncertain::beta(2.0, 5.0).unwrap();
        let samples: Vec<f64> = beta.take_samples(1000);

        assert!(samples.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_gamma_distribution() {
        let gamma = Uncertain::gamma(2.0, 1.0).unwrap();
        let samples: Vec<f64> = gamma.take_samples(1000);

        assert!(samples.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_binomial_distribution() {
        let binomial: Uncertain<u32> = Uncertain::binomial(10, 0.5).unwrap();
        let samples: Vec<u32> = binomial.take_samples(1000);

        assert!(samples.iter().all(|&x| x <= 10));

        let mean = f64::from(samples.iter().sum::<u32>()) / samples.len() as f64;
        assert!((mean - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_poisson_distribution() {
        let poisson: Uncertain<u32> = Uncertain::poisson(3.0).unwrap();
        let samples: Vec<u32> = poisson.take_samples(1000);

        let mean = f64::from(samples.iter().sum::<u32>()) / samples.len() as f64;
        assert!((mean - 3.0).abs() < 1.0);
    }

    #[test]
    fn test_geometric_distribution() {
        let geometric: Uncertain<u32> = Uncertain::geometric(0.2).unwrap();
        let samples: Vec<u32> = geometric.take_samples(1000);

        assert!(samples.iter().all(|&x| x >= 1));

        let mean = f64::from(samples.iter().sum::<u32>()) / samples.len() as f64;
        assert!((mean - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_mixture_distribution() {
        let normal1 = Uncertain::normal(0.0, 1.0).unwrap();
        let normal2 = Uncertain::normal(10.0, 1.0).unwrap();
        let mixture = Uncertain::mixture(vec![normal1, normal2], Some(vec![0.5, 0.5])).unwrap();

        let samples: Vec<f64> = mixture.take_samples(1000);

        let low_count = samples.iter().filter(|&&x| x < 5.0).count();
        let high_count = samples.iter().filter(|&&x| x > 5.0).count();

        assert!(low_count > 100);
        assert!(high_count > 100);
    }

    #[test]
    fn test_mixture_single_component() {
        let normal = Uncertain::normal(0.0, 1.0).unwrap();
        let mixture = Uncertain::mixture(vec![normal], None).unwrap();
        let samples: Vec<f64> = mixture.take_samples(100);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.0).abs() < 0.5);
    }

    #[test]
    fn test_mixture_uniform_weights() {
        let normal1 = Uncertain::normal(0.0, 1.0).unwrap();
        let normal2 = Uncertain::normal(5.0, 1.0).unwrap();
        let mixture = Uncertain::mixture(vec![normal1, normal2], None).unwrap();
        let _samples: Vec<f64> = mixture.take_samples(100);
    }

    #[test]
    fn test_empirical_distribution() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let empirical = Uncertain::empirical(data.clone()).unwrap();
        let samples: Vec<f64> = empirical.take_samples(1000);

        assert!(samples.iter().all(|&x| data.contains(&x)));
    }

    #[test]
    fn test_mixture_empty_components() {
        let result = Uncertain::<f64>::mixture(vec![], None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), UncertainError::EmptyComponents);
    }

    #[test]
    fn test_mixture_mismatched_weights() {
        let normal1 = Uncertain::normal(0.0, 1.0).unwrap();
        let normal2 = Uncertain::normal(1.0, 1.0).unwrap();
        let result = Uncertain::mixture(vec![normal1, normal2], Some(vec![0.5]));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), UncertainError::weight_mismatch(2, 1));
    }

    #[test]
    fn test_empirical_empty_data() {
        let result = Uncertain::<f64>::empirical(vec![]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), UncertainError::EmptyData);
    }

    #[test]
    fn test_categorical_empty_probabilities() {
        let probs: HashMap<&str, f64> = HashMap::new();
        let result = Uncertain::categorical(&probs);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), UncertainError::EmptyProbabilities);
    }

    #[test]
    fn test_categorical_with_unnormalized_probabilities() {
        let mut probs = HashMap::new();
        probs.insert("a", 2.0);
        probs.insert("b", 3.0);
        probs.insert("c", 5.0);

        let categorical = Uncertain::categorical(&probs).unwrap();
        let samples: Vec<&str> = categorical.take_samples(1000);

        assert!(samples.iter().all(|&x| ["a", "b", "c"].contains(&x)));
    }

    #[test]
    fn test_normal_distribution_edge_cases() {
        let normal_zero_std = Uncertain::normal(5.0, 0.0).unwrap();
        let samples: Vec<f64> = normal_zero_std.take_samples(10);
        assert!(samples.iter().all(|&x| (x - 5.0).abs() < 0.01));

        let normal_negative = Uncertain::normal(-10.0, 2.0).unwrap();
        let samples: Vec<f64> = normal_negative.take_samples(1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - (-10.0)).abs() < 0.5);
    }

    #[test]
    fn test_uniform_edge_cases() {
        let uniform_point = Uncertain::uniform(5.0, 5.0).unwrap();
        let samples: Vec<f64> = uniform_point.take_samples(10);
        assert!(samples.iter().all(|&x| (x - 5.0).abs() < f64::EPSILON));

        let uniform_negative = Uncertain::uniform(-10.0, -5.0).unwrap();
        let samples: Vec<f64> = uniform_negative.take_samples(100);
        assert!(samples.iter().all(|&x| (-10.0..=-5.0).contains(&x)));
    }

    #[test]
    fn test_bernoulli_edge_cases() {
        let bernoulli_false = Uncertain::bernoulli(0.0).unwrap();
        let samples: Vec<bool> = bernoulli_false.take_samples(100);
        assert!(samples.iter().all(|&x| !x));

        let bernoulli_true = Uncertain::bernoulli(1.0).unwrap();
        let samples: Vec<bool> = bernoulli_true.take_samples(100);
        assert!(samples.iter().all(|&x| x));
    }

    #[test]
    fn test_exponential_edge_cases() {
        let exponential_high_rate = Uncertain::exponential(100.0).unwrap();
        let samples: Vec<f64> = exponential_high_rate.take_samples(100);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean < 0.1);
    }

    #[test]
    fn test_binomial_edge_cases() {
        let binomial_zero: Uncertain<u32> = Uncertain::binomial(0, 0.5).unwrap();
        let samples: Vec<u32> = binomial_zero.take_samples(10);
        assert!(samples.iter().all(|&x| x == 0));

        let binomial_p_zero: Uncertain<u32> = Uncertain::binomial(10, 0.0).unwrap();
        let samples: Vec<u32> = binomial_p_zero.take_samples(10);
        assert!(samples.iter().all(|&x| x == 0));

        let binomial_p_one: Uncertain<u32> = Uncertain::binomial(10, 1.0).unwrap();
        let samples: Vec<u32> = binomial_p_one.take_samples(10);
        assert!(samples.iter().all(|&x| x == 10));
    }

    #[test]
    fn test_normal_rejects_non_finite() {
        match Uncertain::normal(f64::NAN, 1.0).unwrap_err() {
            UncertainError::NonFiniteParameter { parameter, value } => {
                assert_eq!(parameter, "mean");
                assert!(value.is_nan());
            }
            other => panic!("expected NonFiniteParameter, got {other:?}"),
        }
        assert!(Uncertain::normal(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn test_normal_rejects_negative_std_dev() {
        let err = Uncertain::normal(0.0, -1.0).unwrap_err();
        assert_eq!(
            err,
            UncertainError::invalid_parameter("std_dev", -1.0, "must be non-negative")
        );
    }

    #[test]
    fn test_uniform_rejects_inverted_bounds() {
        assert!(Uncertain::uniform(10.0, 0.0).is_err());
    }

    #[test]
    fn test_exponential_rejects_non_positive_rate() {
        assert!(Uncertain::exponential(0.0).is_err());
        assert!(Uncertain::exponential(-1.0).is_err());
    }

    #[test]
    fn test_log_normal_rejects_negative_sigma() {
        assert!(Uncertain::log_normal(0.0, -1.0).is_err());
    }

    #[test]
    fn test_beta_rejects_non_positive_shape_parameters() {
        assert!(Uncertain::beta(0.0, 1.0).is_err());
        assert!(Uncertain::beta(1.0, 0.0).is_err());
        assert!(Uncertain::beta(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_gamma_rejects_non_positive_shape() {
        assert!(Uncertain::gamma(0.0, 1.0).is_err());
        assert!(Uncertain::gamma(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_gamma_rejects_negative_scale() {
        assert!(Uncertain::gamma(1.0, -1.0).is_err());
    }

    #[test]
    fn test_gamma_allows_zero_scale_degenerate() {
        let gamma = Uncertain::gamma(2.0, 0.0).unwrap();
        let samples: Vec<f64> = gamma.take_samples(10);
        assert!(samples.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_bernoulli_rejects_out_of_range_probability() {
        assert!(Uncertain::bernoulli(-0.1).is_err());
        assert!(Uncertain::bernoulli(1.1).is_err());
    }

    #[test]
    fn test_binomial_rejects_out_of_range_probability() {
        let result: Result<Uncertain<u32>> = Uncertain::binomial(10, 1.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_rejects_negative_lambda() {
        let result: Result<Uncertain<u32>> = Uncertain::poisson(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_rejects_lambda_above_max() {
        let result: Result<Uncertain<u32>> = Uncertain::poisson(Poisson::<f64>::MAX_LAMBDA * 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_allows_zero_lambda_degenerate() {
        let poisson: Uncertain<u32> = Uncertain::poisson(0.0).unwrap();
        let samples: Vec<u32> = poisson.take_samples(10);
        assert!(samples.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_geometric_rejects_zero_probability() {
        let result: Result<Uncertain<u32>> = Uncertain::geometric(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_geometric_rejects_out_of_range_probability() {
        let result: Result<Uncertain<u32>> = Uncertain::geometric(1.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_geometric_allows_probability_one() {
        let geometric: Uncertain<u32> = Uncertain::geometric(1.0).unwrap();
        let samples: Vec<u32> = geometric.take_samples(100);
        assert!(samples.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_mixture_weighted_selection_uses_correct_proportions() {
        // Weights [1.0, 3.0] over total 4.0 => component 0 should be picked ~25% of the
        // time. A `/` -> `%` or `/` -> `*` mutation in the weight-normalization leaves
        // unnormalized weights (both >= total's modulus/product), which collapses
        // selection onto whichever component is checked first in the cumulative array,
        // producing a ratio far from 0.25. This also exercises the `<=` boundary in the
        // cumulative lookup (a `<=` -> `>` mutation there swaps which component wins).
        let low = Uncertain::normal(0.0, 0.001).unwrap();
        let high = Uncertain::normal(100.0, 0.001).unwrap();
        let mixture = Uncertain::mixture(vec![low, high], Some(vec![1.0, 3.0])).unwrap();
        let samples: Vec<f64> = mixture.take_samples(10_000);
        let low_ratio = samples.iter().filter(|&&x| x < 50.0).count() as f64 / samples.len() as f64;
        assert!(
            (low_ratio - 0.25).abs() < 0.05,
            "expected ~25% low-component selections, got {low_ratio}"
        );
    }

    #[test]
    fn test_categorical_selection_uses_correct_proportions() {
        // Same rationale as the mixture test above, for the categorical distribution's
        // independent weight-normalization code path. Proportions are checked
        // order-independently (HashMap iteration order is unspecified): "common"'s share
        // is prob/total regardless of which key the loop visits first.
        let mut probs = HashMap::new();
        probs.insert("rare", 1.0);
        probs.insert("common", 3.0);
        let categorical = Uncertain::categorical(&probs).unwrap();
        let samples: Vec<&str> = categorical.take_samples(10_000);
        let common_ratio =
            samples.iter().filter(|&&x| x == "common").count() as f64 / samples.len() as f64;
        assert!(
            (common_ratio - 0.75).abs() < 0.05,
            "expected ~75% 'common' selections, got {common_ratio}"
        );
    }

    #[test]
    fn test_normal_distribution_moments() {
        let normal = Uncertain::normal(10.0, 3.0).unwrap();
        let samples: Vec<f64> = normal.take_samples(20_000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std = variance.sqrt();
        assert!((mean - 10.0).abs() < 0.15, "mean {mean}, expected 10.0");
        assert!((std - 3.0).abs() < 0.25, "std {std}, expected 3.0");
    }

    #[test]
    fn test_normal_tails_are_not_truncated() {
        // The pre-rand_distr Box-Muller implementation clamped its uniforms to
        // [0.001, 0.999], making |z| > ~3.09 unreachable. rand_distr's Normal (Ziggurat
        // method) has no such clamp. Seeded for a deterministic, reproducible check
        // instead of "usually produces a tail value."
        use rand::SeedableRng;
        let normal = Uncertain::normal(0.0, 1.0).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let samples = normal.take_samples_with(&mut rng, 10_000_000);
        assert!(
            samples.iter().any(|&x| x.abs() > 4.0),
            "expected at least one |z| > 4 in 10,000,000 samples of a standard normal"
        );
    }

    #[test]
    fn test_beta_distribution_moments() {
        // Closed-form: mean = a/(a+b), variance = ab / ((a+b)^2 (a+b+1)).
        let alpha = 2.0;
        let beta_param = 8.0;
        let beta = Uncertain::beta(alpha, beta_param).unwrap();
        let samples: Vec<f64> = beta.take_samples(20_000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let expected_mean = alpha / (alpha + beta_param);
        assert!(
            (mean - expected_mean).abs() < 0.02,
            "mean {mean}, expected {expected_mean}"
        );

        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let expected_variance =
            (alpha * beta_param) / ((alpha + beta_param).powi(2) * (alpha + beta_param + 1.0));
        assert!(
            (variance - expected_variance).abs() < 0.01,
            "variance {variance}, expected {expected_variance}"
        );
    }

    #[test]
    fn test_gamma_distribution_moments_shape_at_least_one() {
        // Closed-form: mean = shape*scale, variance = shape*scale^2.
        let shape = 3.0;
        let scale = 2.0;
        let gamma = Uncertain::gamma(shape, scale).unwrap();
        let samples: Vec<f64> = gamma.take_samples(30_000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let expected_mean = shape * scale;
        assert!(
            (mean - expected_mean).abs() < expected_mean * 0.1,
            "mean {mean}, expected {expected_mean}"
        );

        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let expected_variance = shape * scale * scale;
        assert!(
            (variance - expected_variance).abs() < expected_variance * 0.2,
            "variance {variance}, expected {expected_variance}"
        );
    }

    #[test]
    fn test_gamma_distribution_moments_shape_below_one() {
        let shape = 0.5;
        let scale = 3.0;
        let gamma = Uncertain::gamma(shape, scale).unwrap();
        let samples: Vec<f64> = gamma.take_samples(30_000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let expected_mean = shape * scale;
        assert!(
            (mean - expected_mean).abs() < expected_mean * 0.2,
            "mean {mean}, expected {expected_mean}"
        );

        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let expected_variance = shape * scale * scale;
        assert!(
            (variance - expected_variance).abs() < expected_variance * 0.3,
            "variance {variance}, expected {expected_variance}"
        );
    }
}
