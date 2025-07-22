use crate::computation::{ComputationNode, SampleContext};
use crate::operations::Arithmetic;
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
    T: Clone + Send + Sync + 'static,
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
    pub fn map<U, F>(&self, transform: F) -> Uncertain<U>
    where
        U: Clone + Send + Sync + 'static,
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
    pub fn flat_map<U, F>(&self, transform: F) -> Uncertain<U>
    where
        U: Clone + Send + Sync + 'static,
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
    pub fn take_samples(&self, count: usize) -> Vec<T> {
        self.samples().take(count).collect()
    }
}

impl<T> Uncertain<T>
where
    T: Clone + Send + Sync + PartialOrd + 'static,
{
    /// Compare this uncertain value with another, returning an uncertain boolean
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

// Implement comparison operators

impl<T> std::cmp::PartialEq for Uncertain<T>
where
    T: Clone + Send + Sync + PartialEq + 'static,
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
    T: Clone + Send + Sync + PartialOrd + 'static,
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
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Uncertain")
            .field("sample", &self.sample())
            .finish()
    }
}
