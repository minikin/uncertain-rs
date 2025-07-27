use crate::traits::Shareable;
use crate::{Uncertain, computation::ComputationNode};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Trait alias for types that support arithmetic operations
pub trait Arithmetic:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Clone
    + Send
    + Sync
    + 'static
{
}

/// Blanket implementation for any type that satisfies the arithmetic requirements
impl<T> Arithmetic for T where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Clone
        + Send
        + Sync
        + 'static
{
}

/// Binary operation types for computation graph
#[derive(Clone)]
pub enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOperation {
    #[must_use]
    pub fn apply<T>(&self, left: T, right: T) -> T
    where
        T: Arithmetic,
    {
        match self {
            BinaryOperation::Add => left + right,
            BinaryOperation::Sub => left - right,
            BinaryOperation::Mul => left * right,
            BinaryOperation::Div => left / right,
        }
    }
}

// Addition operations
impl<T> Add for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = ComputationNode::BinaryOp {
            left: Box::new(self.node),
            right: Box::new(rhs.node),
            operation: BinaryOperation::Add,
        };
        Uncertain::with_node(node)
    }
}

impl<T> Add<T> for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn add(self, rhs: T) -> Self::Output {
        self + Uncertain::point(rhs)
    }
}

impl Add<Uncertain<f64>> for f64 {
    type Output = Uncertain<f64>;

    fn add(self, rhs: Uncertain<f64>) -> Self::Output {
        Uncertain::point(self) + rhs
    }
}

// Subtraction operations
impl<T> Sub for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let node = ComputationNode::BinaryOp {
            left: Box::new(self.node),
            right: Box::new(rhs.node),
            operation: BinaryOperation::Sub,
        };
        Uncertain::with_node(node)
    }
}

impl<T> Sub<T> for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self - Uncertain::point(rhs)
    }
}

impl Sub<Uncertain<f64>> for f64 {
    type Output = Uncertain<f64>;

    fn sub(self, rhs: Uncertain<f64>) -> Self::Output {
        Uncertain::point(self) - rhs
    }
}

// Multiplication operations
impl<T> Mul for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = ComputationNode::BinaryOp {
            left: Box::new(self.node),
            right: Box::new(rhs.node),
            operation: BinaryOperation::Mul,
        };
        Uncertain::with_node(node)
    }
}

impl<T> Mul<T> for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self * Uncertain::point(rhs)
    }
}

impl Mul<Uncertain<f64>> for f64 {
    type Output = Uncertain<f64>;

    fn mul(self, rhs: Uncertain<f64>) -> Self::Output {
        Uncertain::point(self) * rhs
    }
}

// Division operations
impl<T> Div for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let node = ComputationNode::BinaryOp {
            left: Box::new(self.node),
            right: Box::new(rhs.node),
            operation: BinaryOperation::Div,
        };
        Uncertain::with_node(node)
    }
}

impl<T> Div<T> for Uncertain<T>
where
    T: Arithmetic,
{
    type Output = Uncertain<T>;

    fn div(self, rhs: T) -> Self::Output {
        self / Uncertain::point(rhs)
    }
}

impl Div<Uncertain<f64>> for f64 {
    type Output = Uncertain<f64>;

    fn div(self, rhs: Uncertain<f64>) -> Self::Output {
        Uncertain::point(self) / rhs
    }
}

// Negation
impl<T> Neg for Uncertain<T>
where
    T: Neg<Output = T> + Shareable,
{
    type Output = Uncertain<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

// Additional mathematical operations for floating point types
impl Uncertain<f64> {
    /// Raises the uncertain value to a power
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let base = Uncertain::normal(2.0, 0.1);
    /// let squared = base.pow(2.0);
    /// ```
    #[must_use]
    pub fn pow(&self, exponent: f64) -> Uncertain<f64> {
        self.map(move |x| x.powf(exponent))
    }

    /// Takes the square root of the uncertain value
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let positive = Uncertain::uniform(1.0, 100.0);
    /// let sqrt_val = positive.sqrt();
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Uncertain<f64> {
        self.map(f64::sqrt)
    }

    /// Takes the natural logarithm of the uncertain value
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let positive = Uncertain::uniform(0.1, 10.0);
    /// let ln_val = positive.ln();
    /// ```
    #[must_use]
    pub fn ln(&self) -> Uncertain<f64> {
        self.map(f64::ln)
    }

    /// Takes the exponential of the uncertain value
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 1.0);
    /// let exp_val = normal.exp();
    /// ```
    #[must_use]
    pub fn exp(&self) -> Uncertain<f64> {
        self.map(f64::exp)
    }

    /// Takes the absolute value of the uncertain value
    ///
    /// # Example
    /// ```rust
    /// use uncertain_rs::Uncertain;
    ///
    /// let normal = Uncertain::normal(0.0, 2.0);
    /// let abs_val = normal.abs();
    /// ```
    #[must_use]
    pub fn abs(&self) -> Uncertain<f64> {
        self.map(f64::abs)
    }

    /// Applies sine function to the uncertain value
    #[must_use]
    pub fn sin(&self) -> Uncertain<f64> {
        self.map(f64::sin)
    }

    /// Applies cosine function to the uncertain value
    #[must_use]
    pub fn cos(&self) -> Uncertain<f64> {
        self.map(f64::cos)
    }

    /// Applies tangent function to the uncertain value
    #[must_use]
    pub fn tan(&self) -> Uncertain<f64> {
        self.map(f64::tan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let x = Uncertain::point(5.0);
        let y = Uncertain::point(3.0);
        let sum = x + y;
        assert!((sum.sample() - 8.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalar_addition() {
        let x = Uncertain::point(5.0);
        let sum = x + 3.0;
        assert!((sum.sample() - 8.0_f64).abs() < f64::EPSILON);

        let sum2 = 3.0 + Uncertain::point(5.0);
        assert!((sum2.sample() - 8.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiplication() {
        let x = Uncertain::point(4.0);
        let y = Uncertain::point(3.0);
        let product = x * y;
        assert!((product.sample() - 12.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_complex_expression() {
        let x = Uncertain::point(2.0);
        let y = Uncertain::point(3.0);
        let result = (x + y) * 2.0 - 1.0;
        assert!((result.sample() - 9.0_f64).abs() < f64::EPSILON); // (2 + 3) * 2 - 1 = 9
    }

    #[test]
    fn test_mathematical_functions() {
        let x = Uncertain::point(4.0);
        assert!((x.sqrt().sample() - 2.0_f64).abs() < f64::EPSILON);

        let y = Uncertain::point(2.0);
        assert!((y.pow(3.0).sample() - 8.0_f64).abs() < f64::EPSILON);
    }
}
