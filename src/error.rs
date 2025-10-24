//! Error types for the uncertain-rs library.
//!
//! This module defines all error types that can occur when working with uncertain values.

use thiserror::Error;

/// The main error type for the uncertain-rs library.
///
/// This enum represents all possible errors that can occur when creating
/// or manipulating uncertain values.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum UncertainError {
    /// Error when an empty collection is provided where at least one element is required.
    #[error("Empty components: at least one component is required")]
    EmptyComponents,

    /// Error when the number of weights doesn't match the number of components.
    #[error("Weight count mismatch: expected {expected} components, got {actual} weights")]
    WeightCountMismatch {
        /// The expected number of weights
        expected: usize,
        /// The actual number of weights provided
        actual: usize,
    },

    /// Error when an empty data vector is provided.
    #[error("Empty data: data vector cannot be empty")]
    EmptyData,

    /// Error when an empty probability map is provided.
    #[error("Empty probabilities: probability map cannot be empty")]
    EmptyProbabilities,

    /// Error when an invalid parameter value is provided.
    #[error("Invalid parameter '{parameter}': value {value} {constraint}")]
    InvalidParameter {
        /// The name of the parameter
        parameter: &'static str,
        /// The invalid value
        value: f64,
        /// A description of the constraint that was violated
        constraint: &'static str,
    },

    /// Error when a parameter is not finite (NaN or infinite).
    #[error("Non-finite parameter '{parameter}': {value}")]
    NonFiniteParameter {
        /// The name of the parameter
        parameter: &'static str,
        /// The non-finite value
        value: f64,
    },

    /// Error when weights are invalid (negative or all zero).
    #[error("Invalid weights: {reason}")]
    InvalidWeights {
        /// The reason the weights are invalid
        reason: String,
    },

    /// Error when sample count is invalid (zero or too large).
    #[error("Invalid sample count: {count} ({reason})")]
    InvalidSampleCount {
        /// The invalid sample count
        count: usize,
        /// The reason the count is invalid
        reason: &'static str,
    },

    /// Error when a quantile value is out of range [0, 1].
    #[error("Invalid quantile: {value} (must be in range [0, 1])")]
    InvalidQuantile {
        /// The invalid quantile value
        value: f64,
    },

    /// Error when a confidence level is out of range (0, 1).
    #[error("Invalid confidence level: {value} (must be in range (0, 1))")]
    InvalidConfidence {
        /// The invalid confidence level
        value: f64,
    },

    /// Error when a bandwidth parameter is invalid (non-positive).
    #[error("Invalid bandwidth: {value} (must be positive)")]
    InvalidBandwidth {
        /// The invalid bandwidth value
        value: f64,
    },
}

/// A specialized `Result` type for uncertain operations.
///
/// This is a convenience type alias for `Result<T, UncertainError>`.
pub type Result<T> = std::result::Result<T, UncertainError>;

impl UncertainError {
    /// Create an error for invalid parameter with constraint.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::invalid_parameter("std_dev", -1.0, "must be positive");
    /// assert!(error.to_string().contains("std_dev"));
    /// ```
    pub fn invalid_parameter(
        parameter: &'static str,
        value: f64,
        constraint: &'static str,
    ) -> Self {
        Self::InvalidParameter {
            parameter,
            value,
            constraint,
        }
    }

    /// Create an error for non-finite parameter.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::non_finite("mean", f64::NAN);
    /// assert!(error.to_string().contains("mean"));
    /// ```
    pub fn non_finite(parameter: &'static str, value: f64) -> Self {
        Self::NonFiniteParameter { parameter, value }
    }

    /// Create an error for weight count mismatch.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::weight_mismatch(3, 2);
    /// assert!(error.to_string().contains("expected 3"));
    /// ```
    pub fn weight_mismatch(expected: usize, actual: usize) -> Self {
        Self::WeightCountMismatch { expected, actual }
    }

    /// Create an error for invalid weights.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::invalid_weights("all weights are zero");
    /// assert!(error.to_string().contains("all weights are zero"));
    /// ```
    pub fn invalid_weights(reason: impl Into<String>) -> Self {
        Self::InvalidWeights {
            reason: reason.into(),
        }
    }

    /// Create an error for invalid sample count.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::invalid_sample_count(0, "must be greater than zero");
    /// assert!(error.to_string().contains("must be greater than zero"));
    /// ```
    pub fn invalid_sample_count(count: usize, reason: &'static str) -> Self {
        Self::InvalidSampleCount { count, reason }
    }

    /// Create an error for invalid quantile.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::invalid_quantile(1.5);
    /// assert!(error.to_string().contains("1.5"));
    /// ```
    pub fn invalid_quantile(value: f64) -> Self {
        Self::InvalidQuantile { value }
    }

    /// Create an error for invalid confidence level.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::invalid_confidence(1.5);
    /// assert!(error.to_string().contains("1.5"));
    /// ```
    pub fn invalid_confidence(value: f64) -> Self {
        Self::InvalidConfidence { value }
    }

    /// Create an error for invalid bandwidth.
    ///
    /// # Example
    /// ```
    /// use uncertain_rs::error::UncertainError;
    ///
    /// let error = UncertainError::invalid_bandwidth(-0.1);
    /// assert!(error.to_string().contains("-0.1"));
    /// ```
    pub fn invalid_bandwidth(value: f64) -> Self {
        Self::InvalidBandwidth { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_components_error() {
        let error = UncertainError::EmptyComponents;
        assert_eq!(
            error.to_string(),
            "Empty components: at least one component is required"
        );
    }

    #[test]
    fn test_weight_mismatch_error() {
        let error = UncertainError::weight_mismatch(3, 2);
        assert_eq!(
            error.to_string(),
            "Weight count mismatch: expected 3 components, got 2 weights"
        );
    }

    #[test]
    fn test_empty_data_error() {
        let error = UncertainError::EmptyData;
        assert_eq!(error.to_string(), "Empty data: data vector cannot be empty");
    }

    #[test]
    fn test_empty_probabilities_error() {
        let error = UncertainError::EmptyProbabilities;
        assert_eq!(
            error.to_string(),
            "Empty probabilities: probability map cannot be empty"
        );
    }

    #[test]
    fn test_invalid_parameter_error() {
        let error = UncertainError::invalid_parameter("std_dev", -1.0, "must be positive");
        assert!(error.to_string().contains("std_dev"));
        assert!(error.to_string().contains("-1"));
        assert!(error.to_string().contains("must be positive"));
    }

    #[test]
    fn test_non_finite_error() {
        let error = UncertainError::non_finite("mean", f64::NAN);
        assert!(error.to_string().contains("mean"));
        assert!(error.to_string().contains("NaN"));
    }

    #[test]
    fn test_invalid_weights_error() {
        let error = UncertainError::invalid_weights("all weights are zero");
        assert_eq!(error.to_string(), "Invalid weights: all weights are zero");
    }

    #[test]
    fn test_invalid_sample_count_error() {
        let error = UncertainError::invalid_sample_count(0, "must be greater than zero");
        assert!(error.to_string().contains("0"));
        assert!(error.to_string().contains("must be greater than zero"));
    }

    #[test]
    fn test_invalid_quantile_error() {
        let error = UncertainError::invalid_quantile(1.5);
        assert!(error.to_string().contains("1.5"));
        assert!(error.to_string().contains("[0, 1]"));
    }

    #[test]
    fn test_invalid_confidence_error() {
        let error = UncertainError::invalid_confidence(1.5);
        assert!(error.to_string().contains("1.5"));
        assert!(error.to_string().contains("(0, 1)"));
    }

    #[test]
    fn test_invalid_bandwidth_error() {
        let error = UncertainError::invalid_bandwidth(-0.1);
        assert!(error.to_string().contains("-0.1"));
        assert!(error.to_string().contains("positive"));
    }

    #[test]
    fn test_error_clone() {
        let error = UncertainError::EmptyComponents;
        let cloned = error.clone();
        assert_eq!(error, cloned);
    }

    #[test]
    fn test_error_debug() {
        let error = UncertainError::EmptyComponents;
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("EmptyComponents"));
    }

    #[test]
    fn test_error_partial_eq() {
        let error1 = UncertainError::weight_mismatch(3, 2);
        let error2 = UncertainError::weight_mismatch(3, 2);
        let error3 = UncertainError::weight_mismatch(4, 2);

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }
}
