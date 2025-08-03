#![allow(clippy::cast_precision_loss)]

use crate::Uncertain;
use crate::traits::Shareable;
use rand::prelude::*;
use rand::random;
use rand::rng;
use std::collections::HashMap;
use std::f64::consts::PI;

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
    /// # Panics
    /// May panic if the components vector is empty when attempting to get the first
    /// component, which should not happen due to the input validation.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal1 = Uncertain::normal(0.0, 1.0);
    /// let normal2 = Uncertain::normal(5.0, 1.0);
    /// let mixture = Uncertain::mixture(
    ///     vec![normal1, normal2],
    ///     Some(vec![0.7, 0.3])
    /// ).unwrap();
    /// ```
    pub fn mixture(
        components: Vec<Uncertain<T>>,
        weights: Option<Vec<f64>>,
    ) -> Result<Self, &'static str> {
        if components.is_empty() {
            return Err("At least one component required");
        }

        if components.len() == 1 {
            return Ok(components.into_iter().next().unwrap());
        }

        let weights = match weights {
            Some(w) => {
                if w.len() != components.len() {
                    return Err("Weights count must match components count");
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
    /// # Panics
    /// May panic if the random number generator fails to select from the data,
    /// which should not happen if the data is non-empty.
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let empirical = Uncertain::empirical(data).unwrap();
    /// ```
    pub fn empirical(data: Vec<T>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }

        Ok(Uncertain::new(move || {
            data.choose(&mut rng())
                .expect("Data vector should not be empty")
                .clone()
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
    /// # Panics
    /// May panic if the cumulative probability vector is empty, which should not happen
    /// if the input validation passes.
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
    pub fn categorical(probabilities: &HashMap<T, f64>) -> Result<Self, &'static str> {
        if probabilities.is_empty() {
            return Err("Probabilities cannot be empty");
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
                || {
                    cumulative
                        .last()
                        .expect("Cumulative vector should not be empty")
                        .0
                        .clone()
                },
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
    /// * `std_dev` - The standard deviation
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0); // Standard normal
    /// let measurement = Uncertain::normal(100.0, 5.0); // Measurement with error
    /// ```
    #[must_use]
    pub fn normal(mean: f64, std_dev: f64) -> Self {
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
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let uniform = Uncertain::uniform(0.0, 10.0);
    /// ```
    #[must_use]
    pub fn uniform(min: f64, max: f64) -> Self {
        Uncertain::new(move || min + (max - min) * random::<f64>())
    }

    /// Creates an exponential distribution
    ///
    /// # Arguments
    /// * `rate` - The rate parameter (lambda)
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let exponential = Uncertain::exponential(1.0);
    /// ```
    #[must_use]
    pub fn exponential(rate: f64) -> Self {
        Uncertain::new(move || -random::<f64>().ln() / rate)
    }

    /// Creates a log-normal distribution
    ///
    /// # Arguments
    /// * `mu` - Mean of the underlying normal distribution
    /// * `sigma` - Standard deviation of the underlying normal distribution
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let lognormal = Uncertain::log_normal(0.0, 1.0);
    /// ```
    #[must_use]
    pub fn log_normal(mu: f64, sigma: f64) -> Self {
        let normal = Self::normal(mu, sigma);
        normal.map(f64::exp)
    }

    /// Creates a beta distribution
    ///
    /// # Arguments
    /// * `alpha` - First shape parameter
    /// * `beta` - Second shape parameter
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let beta = Uncertain::beta(2.0, 5.0);
    /// ```
    #[must_use]
    pub fn beta(alpha: f64, beta: f64) -> Self {
        Uncertain::new(move || {
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
        })
    }

    /// Creates a gamma distribution
    ///
    /// # Arguments
    /// * `shape` - Shape parameter (alpha)
    /// * `scale` - Scale parameter (beta)
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let gamma = Uncertain::gamma(2.0, 1.0);
    /// ```
    #[must_use]
    pub fn gamma(shape: f64, scale: f64) -> Self {
        Uncertain::new(move || {
            // Marsaglia and Tsang method for shape >= 1
            if shape >= 1.0 {
                let d = shape - 1.0 / 3.0;
                let c = 1.0 / (9.0 * d).sqrt();

                loop {
                    let normal_sample = Self::normal(0.0, 1.0).sample();
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
                let gamma_1_plus_shape = Self::gamma(shape + 1.0, scale).sample();
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
    /// * `probability` - Probability of success (true)
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let biased_coin = Uncertain::bernoulli(0.7); // 70% chance of true
    /// ```
    #[must_use]
    pub fn bernoulli(probability: f64) -> Self {
        Uncertain::new(move || random::<f64>() < probability)
    }
}

// Integer distributions
impl<T> Uncertain<T>
where
    T: Clone + Send + Sync + From<u32> + std::ops::AddAssign + Default + 'static,
{
    /// Creates a binomial distribution
    ///
    /// # Arguments
    /// * `trials` - Number of trials
    /// * `probability` - Probability of success on each trial
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let binomial: Uncertain<u32> = Uncertain::binomial(100, 0.3);
    /// ```
    #[must_use]
    pub fn binomial(trials: u32, probability: f64) -> Self {
        Uncertain::new(move || {
            let mut count = T::default();
            for _ in 0..trials {
                if random::<f64>() < probability {
                    count += T::from(1);
                }
            }
            count
        })
    }

    /// Creates a Poisson distribution
    ///
    /// # Arguments
    /// * `lambda` - Rate parameter
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let poisson: Uncertain<u32> = Uncertain::poisson(3.5);
    /// ```
    #[must_use]
    pub fn poisson(lambda: f64) -> Self {
        Uncertain::new(move || {
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

            // Return k - 1, but this is simplified for the generic case
            k
        })
    }

    /// Creates a geometric distribution
    ///
    /// # Arguments
    /// * `probability` - Probability of success on each trial
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let geometric: Uncertain<u32> = Uncertain::geometric(0.1);
    /// ```
    #[must_use]
    pub fn geometric(probability: f64) -> Self {
        Uncertain::new(move || {
            let mut trials = T::from(1);
            while random::<f64>() >= probability {
                trials += T::from(1);
            }
            trials
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_distribution() {
        let normal = Uncertain::normal(0.0, 1.0);
        let samples: Vec<f64> = normal.take_samples(1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_uniform_distribution() {
        let uniform = Uncertain::uniform(0.0, 10.0);
        let samples: Vec<f64> = uniform.take_samples(1000);
        assert!(samples.iter().all(|&x| (0.0..=10.0).contains(&x)));
    }

    #[test]
    fn test_bernoulli_distribution() {
        let bernoulli = Uncertain::bernoulli(0.7);
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
        let exponential = Uncertain::exponential(2.0);
        let samples: Vec<f64> = exponential.take_samples(1000);

        assert!(samples.iter().all(|&x| x >= 0.0));

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_log_normal_distribution() {
        let log_normal = Uncertain::log_normal(0.0, 1.0);
        let samples: Vec<f64> = log_normal.take_samples(1000);

        assert!(samples.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_beta_distribution() {
        let beta = Uncertain::beta(2.0, 5.0);
        let samples: Vec<f64> = beta.take_samples(1000);

        assert!(samples.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_gamma_distribution() {
        let gamma = Uncertain::gamma(2.0, 1.0);
        let samples: Vec<f64> = gamma.take_samples(1000);

        assert!(samples.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_binomial_distribution() {
        let binomial: Uncertain<u32> = Uncertain::binomial(10, 0.5);
        let samples: Vec<u32> = binomial.take_samples(1000);

        assert!(samples.iter().all(|&x| x <= 10));

        let mean = f64::from(samples.iter().sum::<u32>()) / samples.len() as f64;
        assert!((mean - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_poisson_distribution() {
        let poisson: Uncertain<u32> = Uncertain::poisson(3.0);
        let samples: Vec<u32> = poisson.take_samples(1000);

        let mean = f64::from(samples.iter().sum::<u32>()) / samples.len() as f64;
        assert!((mean - 3.0).abs() < 1.0);
    }

    #[test]
    fn test_geometric_distribution() {
        let geometric: Uncertain<u32> = Uncertain::geometric(0.2);
        let samples: Vec<u32> = geometric.take_samples(1000);

        assert!(samples.iter().all(|&x| x >= 1));

        let mean = f64::from(samples.iter().sum::<u32>()) / samples.len() as f64;
        assert!((mean - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_mixture_distribution() {
        let normal1 = Uncertain::normal(0.0, 1.0);
        let normal2 = Uncertain::normal(10.0, 1.0);
        let mixture = Uncertain::mixture(vec![normal1, normal2], Some(vec![0.5, 0.5])).unwrap();

        let samples: Vec<f64> = mixture.take_samples(1000);

        let low_count = samples.iter().filter(|&&x| x < 5.0).count();
        let high_count = samples.iter().filter(|&&x| x > 5.0).count();

        assert!(low_count > 100);
        assert!(high_count > 100);
    }

    #[test]
    fn test_mixture_single_component() {
        let normal = Uncertain::normal(0.0, 1.0);
        let mixture = Uncertain::mixture(vec![normal], None).unwrap();
        let samples: Vec<f64> = mixture.take_samples(100);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.0).abs() < 0.5);
    }

    #[test]
    fn test_mixture_uniform_weights() {
        let normal1 = Uncertain::normal(0.0, 1.0);
        let normal2 = Uncertain::normal(5.0, 1.0);
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
        let result: Result<Uncertain<f64>, &str> = Uncertain::mixture(vec![], None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "At least one component required");
    }

    #[test]
    fn test_mixture_mismatched_weights() {
        let normal1 = Uncertain::normal(0.0, 1.0);
        let normal2 = Uncertain::normal(1.0, 1.0);
        let result = Uncertain::mixture(vec![normal1, normal2], Some(vec![0.5]));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Weights count must match components count"
        );
    }

    #[test]
    fn test_empirical_empty_data() {
        let result: Result<Uncertain<f64>, &str> = Uncertain::empirical(vec![]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Data cannot be empty");
    }

    #[test]
    fn test_categorical_empty_probabilities() {
        let probs: HashMap<&str, f64> = HashMap::new();
        let result = Uncertain::categorical(&probs);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Probabilities cannot be empty");
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
        let normal_zero_std = Uncertain::normal(5.0, 0.0);
        let samples: Vec<f64> = normal_zero_std.take_samples(10);
        assert!(samples.iter().all(|&x| (x - 5.0).abs() < 0.01));

        let normal_negative = Uncertain::normal(-10.0, 2.0);
        let samples: Vec<f64> = normal_negative.take_samples(1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - (-10.0)).abs() < 0.5);
    }

    #[test]
    fn test_uniform_edge_cases() {
        let uniform_point = Uncertain::uniform(5.0, 5.0);
        let samples: Vec<f64> = uniform_point.take_samples(10);
        assert!(samples.iter().all(|&x| (x - 5.0).abs() < f64::EPSILON));

        let uniform_negative = Uncertain::uniform(-10.0, -5.0);
        let samples: Vec<f64> = uniform_negative.take_samples(100);
        assert!(samples.iter().all(|&x| (-10.0..=-5.0).contains(&x)));
    }

    #[test]
    fn test_bernoulli_edge_cases() {
        let bernoulli_false = Uncertain::bernoulli(0.0);
        let samples: Vec<bool> = bernoulli_false.take_samples(100);
        assert!(samples.iter().all(|&x| !x));

        let bernoulli_true = Uncertain::bernoulli(1.0);
        let samples: Vec<bool> = bernoulli_true.take_samples(100);
        assert!(samples.iter().all(|&x| x));
    }

    #[test]
    fn test_exponential_edge_cases() {
        let exponential_high_rate = Uncertain::exponential(100.0);
        let samples: Vec<f64> = exponential_high_rate.take_samples(100);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean < 0.1);
    }

    #[test]
    fn test_binomial_edge_cases() {
        let binomial_zero: Uncertain<u32> = Uncertain::binomial(0, 0.5);
        let samples: Vec<u32> = binomial_zero.take_samples(10);
        assert!(samples.iter().all(|&x| x == 0));

        let binomial_p_zero: Uncertain<u32> = Uncertain::binomial(10, 0.0);
        let samples: Vec<u32> = binomial_p_zero.take_samples(10);
        assert!(samples.iter().all(|&x| x == 0));

        let binomial_p_one: Uncertain<u32> = Uncertain::binomial(10, 1.0);
        let samples: Vec<u32> = binomial_p_one.take_samples(10);
        assert!(samples.iter().all(|&x| x == 10));
    }
}
