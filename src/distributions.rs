#![allow(clippy::cast_precision_loss)]

use crate::Uncertain;
use crate::error::{Result, UncertainError};
use crate::traits::Shareable;
use rand::prelude::*;
use rand::random;
use rand::rng;
use std::collections::HashMap;
use std::f64::consts::PI;

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
            let r: f64 = random();
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
            data.choose(&mut rng())
                .cloned()
                .unwrap_or_else(|| data[0].clone())
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
            let r: f64 = random();
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
        Uncertain::new(move || {
            // Box-Muller transform for normal distribution
            let u1: f64 = random::<f64>().clamp(0.001, 0.999);
            let u2: f64 = random::<f64>().clamp(0.001, 0.999);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            mean + std_dev * z0
        })
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
        Ok(Uncertain::new(move || min + (max - min) * random::<f64>()))
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
        Ok(Uncertain::new(move || -random::<f64>().ln() / rate))
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
        Ok(Self::normal_unchecked(mu, sigma).map(f64::exp))
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
        Ok(Uncertain::new(move || {
            // Using rejection sampling method
            loop {
                let u1: f64 = random();
                let u2: f64 = random();

                let x = u1.powf(1.0 / alpha);
                let y = u2.powf(1.0 / beta);

                if x + y <= 1.0 {
                    return x / (x + y);
                }
            }
        }))
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
        Uncertain::new(move || {
            // Marsaglia and Tsang method for shape >= 1
            if shape >= 1.0 {
                let d = shape - 1.0 / 3.0;
                let c = 1.0 / (9.0 * d).sqrt();

                loop {
                    let normal_sample = Self::normal_unchecked(0.0, 1.0).sample();
                    let v = (1.0 + c * normal_sample).powi(3);

                    if v > 0.0 {
                        let u: f64 = random();
                        if u < 1.0 - 0.0331 * normal_sample.powi(4) {
                            return d * v * scale;
                        }
                        if u.ln() < 0.5 * normal_sample.powi(2) + d * (1.0 - v + v.ln()) {
                            return d * v * scale;
                        }
                    }
                }
            } else {
                // For shape < 1, use transformation
                let gamma_1_plus_shape = Self::gamma_unchecked(shape + 1.0, scale).sample();
                let u: f64 = random();
                gamma_1_plus_shape * u.powf(1.0 / shape)
            }
        })
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
        Ok(Uncertain::new(move || random::<f64>() < probability))
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
        Ok(Uncertain::new(move || {
            let mut count = T::default();
            for _ in 0..trials {
                if random::<f64>() < probability {
                    count += T::from(1);
                }
            }
            count
        }))
    }

    /// Creates a Poisson distribution
    ///
    /// # Arguments
    /// * `lambda` - Rate parameter, must be non-negative (`0.0` degenerates to a point
    ///   mass at `0`)
    ///
    /// # Errors
    /// Returns an error if `lambda` is not finite or negative.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let poisson: Uncertain<u32> = Uncertain::poisson(3.5).unwrap();
    /// ```
    pub fn poisson(lambda: f64) -> Result<Self> {
        validate_non_negative("lambda", lambda)?;
        Ok(Uncertain::new(move || {
            let l = (-lambda).exp();
            let mut k = T::from(0);
            let mut p = 1.0;

            loop {
                k += T::from(1);
                p *= random::<f64>();
                if p <= l {
                    break;
                }
            }

            // Return k - 1 per Knuth's algorithm
            k - T::from(1)
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
        Ok(Uncertain::new(move || {
            let mut trials = T::from(1);
            while random::<f64>() >= probability {
                trials += T::from(1);
            }
            trials
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
}
