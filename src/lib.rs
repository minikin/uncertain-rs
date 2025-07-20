//! # uncertain-rs
//!
//! A Rust library for uncertainty-aware programming, implementing the approach from
//! "Uncertain<T>: A First-Order Type for Uncertain Data" by Bornholt, Mytkowicz, and McKinley.
//!
//! ## Core Concept: Evidence-Based Conditionals
//!
//! Instead of treating uncertain data as exact values (which leads to bugs), this library
//! uses evidence-based conditionals that account for uncertainty:
//!
//! ```rust
//! use uncertain_rs::Uncertain;
//!
//! // Create uncertain values from probability distributions
//! let speed = Uncertain::normal(55.2, 5.0); // GPS reading with ±5 mph error
//!
//! // Evidence-based conditional (returns Uncertain<bool>)
//! let speeding_evidence = speed.gt(60.0);
//!
//! // Convert evidence to decision with confidence level
//! if speeding_evidence.probability_exceeds(0.95) {
//!     // Only act if 95% confident
//!     println!("Issue speeding ticket");
//! }
//! ```
//!
//! ## Features
//!
//! - **Evidence-based conditionals**: Comparisons return evidence, not boolean facts
//! - **Uncertainty propagation**: Arithmetic operations preserve uncertainty
//! - **Lazy evaluation**: Computation graphs built lazily for efficiency
//! - **SPRT hypothesis testing**: Sequential Probability Ratio Test for optimal sampling
//! - **Rich distributions**: Normal, uniform, exponential, binomial, categorical, etc.
//! - **Statistical analysis**: Mean, std dev, confidence intervals, CDF, etc.

pub mod computation;
pub mod distributions;
pub mod hypothesis;
pub mod operations;
pub mod statistics;
pub mod uncertain;

pub use hypothesis::HypothesisResult;
pub use uncertain::Uncertain;

pub use operations::{Arithmetic, Comparison, LogicalOps};

// #[cfg(test)]
// mod tests;
