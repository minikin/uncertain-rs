use crate::error::UncertainError;
use crate::operations::{Arithmetic, arithmetic::BinaryOperation};
use crate::traits::Shareable;
use std::collections::HashMap;
use std::sync::Arc;

/// Adaptive sampling strategy for optimizing computation graph evaluation
#[derive(Debug, Clone)]
pub struct AdaptiveSampling {
    /// Minimum sample count to try
    pub min_samples: usize,
    /// Maximum sample count allowed
    pub max_samples: usize,
    /// Relative error threshold for convergence
    pub error_threshold: f64,
    /// Factor to increase sample count on each iteration
    pub growth_factor: f64,
}

impl Default for AdaptiveSampling {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: 10000,
            error_threshold: 0.01,
            growth_factor: 1.5,
        }
    }
}

/// Caching strategy for computation graph nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachingStrategy {
    /// Cache all intermediate results
    Aggressive,
    /// Cache only expensive operations
    Conservative,
    /// Adaptive caching based on computation cost
    Adaptive,
}

/// Context for memoizing samples within a single evaluation to ensure
/// shared variables produce the same sample value throughout an evaluation
pub struct SampleContext {
    /// Memoized values indexed by node ID
    memoized_values: HashMap<uuid::Uuid, Box<dyn std::any::Any + Send>>,
    /// Current caching strategy
    caching_strategy: CachingStrategy,
    /// Adaptive sampling configuration
    adaptive_sampling: AdaptiveSampling,
}

impl SampleContext {
    /// Create a new empty sample context
    #[must_use]
    pub fn new() -> Self {
        Self {
            memoized_values: HashMap::new(),
            caching_strategy: CachingStrategy::Adaptive,
            adaptive_sampling: AdaptiveSampling::default(),
        }
    }

    /// Create a sample context with specific caching strategy
    #[must_use]
    pub fn with_caching_strategy(strategy: CachingStrategy) -> Self {
        Self {
            memoized_values: HashMap::new(),
            caching_strategy: strategy,
            adaptive_sampling: AdaptiveSampling::default(),
        }
    }

    /// Get a memoized value for a given node ID
    #[must_use]
    pub fn get_value<T: Clone + 'static>(&self, id: &uuid::Uuid) -> Option<T> {
        self.memoized_values.get(id)?.downcast_ref::<T>().cloned()
    }

    /// Set a memoized value for a given node ID
    pub fn set_value<T: Clone + Send + 'static>(&mut self, id: uuid::Uuid, value: T) {
        self.memoized_values.insert(id, Box::new(value));
    }

    /// Clear all memoized values
    pub fn clear(&mut self) {
        self.memoized_values.clear();
    }

    /// Get the number of memoized values
    #[must_use]
    pub fn len(&self) -> usize {
        self.memoized_values.len()
    }

    /// Check if the context is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memoized_values.is_empty()
    }

    /// Determine if a node should be cached based on strategy and cost
    #[must_use]
    pub fn should_cache_node(&self, node: &ComputationNode<impl Shareable>) -> bool {
        match self.caching_strategy {
            CachingStrategy::Aggressive => true,
            CachingStrategy::Conservative => {
                // Only cache expensive operations (depth > 2 or complex nodes)
                node.depth() > 2 || matches!(node, ComputationNode::Conditional { .. })
            }
            CachingStrategy::Adaptive => {
                let complexity = node.compute_complexity();
                complexity > 5 // Threshold for caching
            }
        }
    }

    /// Get the adaptive sampling configuration
    #[must_use]
    pub fn adaptive_sampling(&self) -> &AdaptiveSampling {
        &self.adaptive_sampling
    }

    /// Set adaptive sampling configuration
    pub fn set_adaptive_sampling(&mut self, config: AdaptiveSampling) {
        self.adaptive_sampling = config;
    }
}

impl Default for SampleContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Computation graph node for lazy evaluation using indirect enum
///
/// This enables building complex expressions like `(x + y) * 2.0 - z` as a computation
/// graph that's only evaluated when samples are needed, with proper memoization to
/// ensure shared variables use the same sample within a single evaluation.
#[derive(Clone)]
pub enum ComputationNode<T> {
    /// Leaf node representing a direct sampling function with unique ID
    Leaf {
        id: uuid::Uuid,
        sample: Arc<dyn Fn() -> T + Send + Sync>,
        /// The value this leaf is structurally known to always produce, if any.
        /// `Some` only for leaves built via [`ComputationNode::constant`] (and,
        /// transitively, `Uncertain::point` and the optimizer's own folded results) —
        /// storing the value directly (rather than re-sampling to check it) means the
        /// optimizer's constancy checks never call `sample` at all. Never set by
        /// sampling and comparing — a low-entropy distribution (e.g. `bernoulli(0.99)`)
        /// could return equal samples by chance and be silently, incorrectly folded.
        constant_value: Option<T>,
    },

    /// Binary operation node for combining two uncertain values
    BinaryOp {
        left: Box<ComputationNode<T>>,
        right: Box<ComputationNode<T>>,
        operation: BinaryOperation,
    },

    /// Unary operation node for transforming a single uncertain value
    UnaryOp {
        operand: Box<ComputationNode<T>>,
        operation: UnaryOperation<T>,
    },

    /// Conditional node for if-then-else logic
    Conditional {
        condition: Box<ComputationNode<bool>>,
        if_true: Box<ComputationNode<T>>,
        if_false: Box<ComputationNode<T>>,
    },
}

/// Unary operation types for computation graph
#[derive(Clone)]
pub enum UnaryOperation<T> {
    Map(Arc<dyn Fn(T) -> T + Send + Sync>),
    Filter(Arc<dyn Fn(&T) -> bool + Send + Sync>),
}

impl<T> ComputationNode<T>
where
    T: Shareable,
{
    /// Evaluates the computation graph node with memoization context
    ///
    /// This is the core evaluation method that respects memoization to ensure
    /// shared variables produce consistent samples within a single evaluation.
    ///
    /// # Errors
    ///
    /// Returns `UncertainError::UnsupportedNode` if called on a `BinaryOp` variant (use
    /// `evaluate_arithmetic` instead) or a `Conditional` variant (use `evaluate_arithmetic`,
    /// which handles conditionals, or `evaluate_bool` for boolean graphs).
    pub fn evaluate(&self, context: &mut SampleContext) -> Result<T, UncertainError> {
        match self {
            ComputationNode::Leaf { id, sample, .. } => {
                // Check if we already have a memoized value for this node
                if let Some(cached) = context.get_value::<T>(id) {
                    Ok(cached)
                } else {
                    // Generate new sample and memoize it
                    let value = sample();
                    context.set_value(*id, value.clone());
                    Ok(value)
                }
            }

            ComputationNode::UnaryOp { operand, operation } => {
                let operand_val = operand.evaluate(context)?;
                Ok(match operation {
                    UnaryOperation::Map(func) => func(operand_val),
                    UnaryOperation::Filter(_) => {
                        // Filter requires special handling with rejection sampling
                        // This is a simplified implementation
                        operand_val
                    }
                })
            }

            // These variants require special handling based on type constraints
            ComputationNode::BinaryOp { .. } => Err(UncertainError::unsupported_node(
                "Leaf or UnaryOp",
                "BinaryOp",
            )),

            ComputationNode::Conditional { .. } => Err(UncertainError::unsupported_node(
                "Leaf or UnaryOp",
                "Conditional",
            )),
        }
    }

    /// Evaluates arithmetic operations with proper trait bounds
    ///
    /// Also handles `Conditional` nodes: the condition is evaluated via `evaluate_bool`
    /// and the taken branch is evaluated arithmetically, so this is a total dispatcher
    /// over every `ComputationNode<T>` variant for arithmetic `T`.
    ///
    /// # Errors
    ///
    /// Returns `UncertainError::UnsupportedNode` if evaluating a `Conditional`'s boolean
    /// condition encounters an unsupported node (e.g. a boolean `BinaryOp`).
    pub fn evaluate_arithmetic(&self, context: &mut SampleContext) -> Result<T, UncertainError>
    where
        T: Arithmetic,
    {
        match self {
            ComputationNode::Leaf { id, sample, .. } => {
                if let Some(cached) = context.get_value::<T>(id) {
                    Ok(cached)
                } else {
                    let value = sample();
                    context.set_value(*id, value.clone());
                    Ok(value)
                }
            }

            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => {
                let left_val = left.evaluate_arithmetic(context)?;
                let right_val = right.evaluate_arithmetic(context)?;
                Ok(operation.apply(left_val, right_val))
            }

            ComputationNode::UnaryOp { operand, operation } => {
                let operand_val = operand.evaluate_arithmetic(context)?;
                Ok(match operation {
                    UnaryOperation::Map(func) => func(operand_val),
                    UnaryOperation::Filter(_) => operand_val,
                })
            }

            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                if condition.evaluate_bool(context)? {
                    if_true.evaluate_arithmetic(context)
                } else {
                    if_false.evaluate_arithmetic(context)
                }
            }
        }
    }

    /// Evaluates the computation graph node in a new context
    ///
    /// This creates a fresh context for evaluation, useful when you want
    /// independent samples without memoization effects.
    ///
    /// # Panics
    ///
    /// Panics if the graph contains a node that isn't evaluable in the arithmetic domain.
    /// This can't happen for graphs built via the public `Uncertain<T>` combinator API;
    /// it would only occur from misuse of the low-level `ComputationNode` constructors
    /// (e.g. directly constructing a boolean `BinaryOp`, which has no defined operation).
    #[must_use]
    pub fn evaluate_fresh(&self) -> T
    where
        T: Arithmetic,
    {
        let mut context = SampleContext::new();
        self.evaluate_arithmetic(&mut context)
            .expect("graphs built via the public Uncertain<T> API are always well-formed")
    }

    /// Creates a new leaf node from an arbitrary sampling function
    ///
    /// The result is never treated as structurally constant by the optimizer — use
    /// [`ComputationNode::constant`] for a leaf whose value is known not to vary.
    pub fn leaf<F>(sample: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        ComputationNode::Leaf {
            id: uuid::Uuid::new_v4(),
            sample: Arc::new(sample),
            constant_value: None,
        }
    }

    /// Creates a leaf node whose value is structurally known to be constant
    ///
    /// This is the only way to mark a node as constant for the optimizer's identity
    /// elimination and constant folding passes — deciding constancy by sampling and
    /// comparing is unsound (a low-entropy distribution can return equal samples by
    /// chance) and is never done.
    pub fn constant(value: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        let stored = value.clone();
        ComputationNode::Leaf {
            id: uuid::Uuid::new_v4(),
            sample: Arc::new(move || value.clone()),
            constant_value: Some(stored),
        }
    }

    /// Creates a new binary operation node
    #[must_use]
    pub fn binary_op(
        left: ComputationNode<T>,
        right: ComputationNode<T>,
        operation: BinaryOperation,
    ) -> Self {
        ComputationNode::BinaryOp {
            left: Box::new(left),
            right: Box::new(right),
            operation,
        }
    }

    /// Creates a new unary map operation node
    pub fn map<F>(operand: ComputationNode<T>, func: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        ComputationNode::UnaryOp {
            operand: Box::new(operand),
            operation: UnaryOperation::Map(Arc::new(func)),
        }
    }

    /// Creates a conditional node
    #[must_use]
    pub fn conditional(
        condition: ComputationNode<bool>,
        if_true: ComputationNode<T>,
        if_false: ComputationNode<T>,
    ) -> Self {
        ComputationNode::Conditional {
            condition: Box::new(condition),
            if_true: Box::new(if_true),
            if_false: Box::new(if_false),
        }
    }

    /// Counts the number of nodes in the computation graph
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            ComputationNode::Leaf { .. } => 1,
            ComputationNode::BinaryOp { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
            ComputationNode::UnaryOp { operand, .. } => 1 + operand.node_count(),
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => 1 + condition.node_count() + if_true.node_count() + if_false.node_count(),
        }
    }

    /// Gets the depth of the computation graph
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            ComputationNode::Leaf { .. } => 1,
            ComputationNode::BinaryOp { left, right, .. } => 1 + left.depth().max(right.depth()),
            ComputationNode::UnaryOp { operand, .. } => 1 + operand.depth(),
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => 1 + condition.depth().max(if_true.depth().max(if_false.depth())),
        }
    }

    /// Checks if the computation graph contains any conditional nodes
    #[must_use]
    pub fn has_conditionals(&self) -> bool {
        match self {
            ComputationNode::Leaf { .. } => false,
            ComputationNode::BinaryOp { left, right, .. } => {
                left.has_conditionals() || right.has_conditionals()
            }
            ComputationNode::UnaryOp { operand, .. } => operand.has_conditionals(),
            ComputationNode::Conditional { .. } => true,
        }
    }

    /// Estimate computational complexity of the node for caching decisions
    #[must_use]
    pub fn compute_complexity(&self) -> usize {
        match self {
            ComputationNode::Leaf { .. } => 1,
            ComputationNode::BinaryOp { left, right, .. } => {
                2 + left.compute_complexity() + right.compute_complexity()
            }
            ComputationNode::UnaryOp { operand, .. } => 1 + operand.compute_complexity(),
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                5 + condition.compute_complexity()
                    + if_true.compute_complexity()
                    + if_false.compute_complexity()
            }
        }
    }

    /// Generate a structural hash for computation graph caching
    #[must_use]
    pub fn structural_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        self.hash_structure(&mut hasher);
        hasher.finish()
    }

    fn hash_structure(&self, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;

        match self {
            ComputationNode::Leaf { id, .. } => {
                "leaf".hash(hasher);
                id.hash(hasher);
            }
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => {
                "binary".hash(hasher);
                operation.hash(hasher);
                left.hash_structure(hasher);
                right.hash_structure(hasher);
            }
            ComputationNode::UnaryOp { operand, operation } => {
                "unary".hash(hasher);
                unary_operation_identity(operation).hash(hasher);
                operand.hash_structure(hasher);
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                "conditional".hash(hasher);
                condition.hash_structure(hasher);
                if_true.hash_structure(hasher);
                if_false.hash_structure(hasher);
            }
        }
    }

    /// Structural identity key used by [`GraphOptimizer`]'s subexpression cache.
    ///
    /// Unlike [`ComputationNode::structural_hash`], equality of this key is checked via
    /// `PartialEq`/`Eq` (not just a `u64` comparison), so a `HashMap` keyed by it can
    /// never confuse two structurally different nodes even if their underlying hash
    /// values collide. Two nodes produce equal keys iff they are the same random
    /// variable (same leaf `id`) or fully deterministic composites over equal
    /// (sub-)keys — never merely because they *sample* the same values. A `UnaryOp`'s
    /// closure is identified by `Arc` pointer, since two distinct closures can't be
    /// proven equal in general; only literal `.clone()`s of the same node share a
    /// pointer.
    fn structural_key(&self) -> StructuralKey {
        match self {
            ComputationNode::Leaf { id, .. } => StructuralKey::Leaf(*id),
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => StructuralKey::Binary(
                operation.clone(),
                Box::new(left.structural_key()),
                Box::new(right.structural_key()),
            ),
            ComputationNode::UnaryOp { operand, operation } => StructuralKey::Unary(
                unary_operation_identity(operation),
                Box::new(operand.structural_key()),
            ),
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => StructuralKey::Conditional(
                Box::new(condition.structural_key()),
                Box::new(if_true.structural_key()),
                Box::new(if_false.structural_key()),
            ),
        }
    }
}

/// Identifies a `UnaryOperation`'s closure by `Arc` pointer address rather than by
/// value, since two distinct `Fn` closures can't be compared for equality in general.
fn unary_operation_identity<T>(operation: &UnaryOperation<T>) -> UnaryOpIdentity {
    match operation {
        UnaryOperation::Map(f) => UnaryOpIdentity::Map(Arc::as_ptr(f).cast::<()>() as usize),
        UnaryOperation::Filter(f) => UnaryOpIdentity::Filter(Arc::as_ptr(f).cast::<()>() as usize),
    }
}

/// A `UnaryOp`'s closure identity, for use in [`StructuralKey`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum UnaryOpIdentity {
    Map(usize),
    Filter(usize),
}

/// Collision-safe structural identity for a [`ComputationNode`], independent of `T`.
///
/// See [`ComputationNode::structural_key`] for the unification rule this encodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StructuralKey {
    Leaf(uuid::Uuid),
    Binary(BinaryOperation, Box<StructuralKey>, Box<StructuralKey>),
    Unary(UnaryOpIdentity, Box<StructuralKey>),
    Conditional(Box<StructuralKey>, Box<StructuralKey>, Box<StructuralKey>),
}

// Specialized implementation for handling conditionals with boolean conditions
impl ComputationNode<bool> {
    /// Evaluates boolean computation nodes
    ///
    /// # Errors
    ///
    /// Returns `UncertainError::UnsupportedNode` if called on a `BinaryOp` variant, since
    /// no boolean `BinaryOperation` is defined (`BinaryOperation` only covers arithmetic
    /// operations, and `bool` doesn't implement `Arithmetic`).
    pub fn evaluate_bool(&self, context: &mut SampleContext) -> Result<bool, UncertainError> {
        match self {
            ComputationNode::Leaf { id, sample, .. } => {
                if let Some(cached) = context.get_value::<bool>(id) {
                    Ok(cached)
                } else {
                    let value = sample();
                    context.set_value(*id, value);
                    Ok(value)
                }
            }
            ComputationNode::UnaryOp { operand, operation } => {
                let operand_val = operand.evaluate_bool(context)?;
                Ok(match operation {
                    UnaryOperation::Map(func) => func(operand_val),
                    UnaryOperation::Filter(_) => operand_val,
                })
            }
            ComputationNode::BinaryOp { .. } => Err(UncertainError::unsupported_node(
                "Leaf, UnaryOp, or Conditional",
                "BinaryOp",
            )),
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                if condition.evaluate_bool(context)? {
                    if_true.evaluate_bool(context)
                } else {
                    if_false.evaluate_bool(context)
                }
            }
        }
    }
}

/// Computation graph optimizer for improving evaluation performance
pub struct GraphOptimizer {
    /// Cache of optimized subexpressions, keyed by full structural identity (not a bare
    /// hash) so a lookup can never return the wrong node even under a hash collision.
    subexpression_cache: HashMap<StructuralKey, Box<dyn std::any::Any + Send + Sync>>,
    /// Number of times a cached subexpression was reused instead of rebuilt.
    cse_hits: usize,
}

impl GraphOptimizer {
    /// Create a new graph optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            subexpression_cache: HashMap::new(),
            cse_hits: 0,
        }
    }

    /// Number of subexpressions currently held in the CSE cache.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.subexpression_cache.len()
    }

    /// Number of times [`GraphOptimizer::eliminate_common_subexpressions`] reused a
    /// cached subexpression instead of rebuilding it.
    #[must_use]
    pub fn cse_hits(&self) -> usize {
        self.cse_hits
    }

    /// Optimizes a computation graph by applying various transformations
    #[must_use]
    pub fn optimize<T>(&mut self, node: ComputationNode<T>) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        let node = self.eliminate_common_subexpressions(node);
        let node = Self::eliminate_identity_operations(node);
        Self::constant_folding(node)
    }

    /// Eliminates common subexpressions by reusing nodes with same structure
    pub fn eliminate_common_subexpressions<T>(
        &mut self,
        node: ComputationNode<T>,
    ) -> ComputationNode<T>
    where
        T: Shareable,
    {
        let key = node.structural_key();

        // Check if we have a cached version of this subexpression. Keying by the full
        // `StructuralKey` (not a bare hash) means this can never return the wrong node:
        // `HashMap` always resolves same-bucket entries via `Eq` before returning one.
        if let Some(cached_node) = self.subexpression_cache.get(&key)
            && let Some(cached) = cached_node.downcast_ref::<ComputationNode<T>>()
        {
            self.cse_hits += 1;
            return cached.clone();
        }

        // Recursively optimize children and cache this node
        let optimized = match node {
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => {
                let left_opt = Box::new(self.eliminate_common_subexpressions(*left));
                let right_opt = Box::new(self.eliminate_common_subexpressions(*right));
                ComputationNode::BinaryOp {
                    left: left_opt,
                    right: right_opt,
                    operation,
                }
            }
            ComputationNode::UnaryOp { operand, operation } => {
                let operand_opt = Box::new(self.eliminate_common_subexpressions(*operand));
                ComputationNode::UnaryOp {
                    operand: operand_opt,
                    operation,
                }
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                let condition_opt = Box::new(self.eliminate_common_subexpressions(*condition));
                let if_true_opt = Box::new(self.eliminate_common_subexpressions(*if_true));
                let if_false_opt = Box::new(self.eliminate_common_subexpressions(*if_false));
                ComputationNode::Conditional {
                    condition: condition_opt,
                    if_true: if_true_opt,
                    if_false: if_false_opt,
                }
            }
            leaf @ ComputationNode::Leaf { .. } => leaf,
        };

        // Cache this subexpression for future use
        self.subexpression_cache
            .insert(key, Box::new(optimized.clone()));

        optimized
    }

    /// Eliminates identity operations like `x + 0` or `x * 1`
    #[allow(clippy::too_many_lines)]
    fn eliminate_identity_operations<T>(node: ComputationNode<T>) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        match node {
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => Self::eliminate_identity_operations_binary(*left, *right, operation),
            ComputationNode::UnaryOp { operand, operation } => {
                Self::eliminate_identity_operations_unary(*operand, operation)
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => Self::eliminate_identity_operations_conditional(*condition, *if_true, *if_false),
            ComputationNode::Leaf { .. } => node,
        }
    }

    /// Handles identity operation elimination for binary operations
    fn eliminate_identity_operations_binary<T>(
        left: ComputationNode<T>,
        right: ComputationNode<T>,
        operation: BinaryOperation,
    ) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        let left_opt = Self::eliminate_identity_operations(left);
        let right_opt = Self::eliminate_identity_operations(right);

        // Check for identity operations by operation type
        match operation {
            BinaryOperation::Add => {
                if let Some(result) = Self::check_addition_identities(&left_opt, &right_opt) {
                    return result;
                }
            }
            BinaryOperation::Sub => {
                if let Some(result) = Self::check_subtraction_identities(&left_opt, &right_opt) {
                    return result;
                }
            }
            BinaryOperation::Mul => {
                if let Some(result) = Self::check_multiplication_identities(&left_opt, &right_opt) {
                    return result;
                }
            }
            BinaryOperation::Div => {
                if let Some(result) = Self::check_division_identities(&left_opt, &right_opt) {
                    return result;
                }
            }
        }

        ComputationNode::BinaryOp {
            left: Box::new(left_opt),
            right: Box::new(right_opt),
            operation,
        }
    }

    /// Checks for addition identity operations: x + 0 = x, 0 + x = x
    fn check_addition_identities<T>(
        left: &ComputationNode<T>,
        right: &ComputationNode<T>,
    ) -> Option<ComputationNode<T>>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        if Self::is_constant_zero(right) {
            return Some(left.clone());
        }
        if Self::is_constant_zero(left) {
            return Some(right.clone());
        }
        None
    }

    /// Checks for subtraction identity operations: x - 0 = x
    fn check_subtraction_identities<T>(
        left: &ComputationNode<T>,
        right: &ComputationNode<T>,
    ) -> Option<ComputationNode<T>>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        if Self::is_constant_zero(right) {
            return Some(left.clone());
        }
        None
    }

    /// Checks for multiplication identity operations: x * 0 = 0, 0 * x = 0, x * 1 = x, 1 * x = x
    fn check_multiplication_identities<T>(
        left: &ComputationNode<T>,
        right: &ComputationNode<T>,
    ) -> Option<ComputationNode<T>>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        // x * 0 = 0, 0 * x = 0
        if Self::is_constant_zero(right) || Self::is_constant_zero(left) {
            return Some(ComputationNode::constant(T::zero()));
        }

        // x * 1 = x, 1 * x = x
        if Self::is_constant_one(right) {
            return Some(left.clone());
        }
        if Self::is_constant_one(left) {
            return Some(right.clone());
        }

        None
    }

    /// Checks for division identity operations: x / 1 = x
    fn check_division_identities<T>(
        left: &ComputationNode<T>,
        right: &ComputationNode<T>,
    ) -> Option<ComputationNode<T>>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        if Self::is_constant_one(right) {
            return Some(left.clone());
        }
        None
    }

    /// Handles identity operation elimination for unary operations
    fn eliminate_identity_operations_unary<T>(
        operand: ComputationNode<T>,
        operation: UnaryOperation<T>,
    ) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        let operand_opt = Self::eliminate_identity_operations(operand);
        ComputationNode::UnaryOp {
            operand: Box::new(operand_opt),
            operation,
        }
    }

    /// Handles identity operation elimination for conditional operations
    fn eliminate_identity_operations_conditional<T>(
        condition: ComputationNode<bool>,
        if_true: ComputationNode<T>,
        if_false: ComputationNode<T>,
    ) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + PartialEq + Clone,
    {
        // For conditionals, we need to handle the boolean condition separately
        let condition_opt = Self::eliminate_identity_operations_bool(condition);
        let if_true_opt = Self::eliminate_identity_operations(if_true);
        let if_false_opt = Self::eliminate_identity_operations(if_false);
        ComputationNode::Conditional {
            condition: Box::new(condition_opt),
            if_true: Box::new(if_true_opt),
            if_false: Box::new(if_false_opt),
        }
    }

    /// Eliminates identity operations for boolean types (no arithmetic operations)
    fn eliminate_identity_operations_bool(node: ComputationNode<bool>) -> ComputationNode<bool> {
        match node {
            ComputationNode::UnaryOp { operand, operation } => {
                let operand_opt = Self::eliminate_identity_operations_bool(*operand);
                ComputationNode::UnaryOp {
                    operand: Box::new(operand_opt),
                    operation,
                }
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                let condition_opt = Self::eliminate_identity_operations_bool(*condition);
                let if_true_opt = Self::eliminate_identity_operations_bool(*if_true);
                let if_false_opt = Self::eliminate_identity_operations_bool(*if_false);
                ComputationNode::Conditional {
                    condition: Box::new(condition_opt),
                    if_true: Box::new(if_true_opt),
                    if_false: Box::new(if_false_opt),
                }
            }
            ComputationNode::Leaf { .. } | ComputationNode::BinaryOp { .. } => node,
        }
    }

    /// Checks whether a node is structurally constant and equal to zero
    ///
    /// Never decided by sampling: a low-entropy distribution could return equal
    /// samples by chance and be silently, incorrectly folded. Only leaves built via
    /// [`ComputationNode::constant`] carry a `constant_value`; it's compared directly
    /// with no call to `sample` at all.
    fn is_constant_zero<T>(node: &ComputationNode<T>) -> bool
    where
        T: PartialEq + Arithmetic,
    {
        match node {
            ComputationNode::Leaf {
                constant_value: Some(value),
                ..
            } => *value == T::zero(),
            _ => false,
        }
    }

    /// Checks whether a node is structurally constant and equal to one
    ///
    /// See [`GraphOptimizer::is_constant_zero`] for why this is structural, not sampled.
    fn is_constant_one<T>(node: &ComputationNode<T>) -> bool
    where
        T: PartialEq + Arithmetic,
    {
        match node {
            ComputationNode::Leaf {
                constant_value: Some(value),
                ..
            } => *value == T::one(),
            _ => false,
        }
    }

    /// Performs constant folding for compile-time evaluation of constant expressions
    fn constant_folding<T>(node: ComputationNode<T>) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + Clone + PartialEq,
    {
        match node {
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => Self::constant_folding_binary_op(*left, *right, operation),
            ComputationNode::UnaryOp { operand, operation } => {
                Self::constant_folding_unary_op(*operand, operation)
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => Self::constant_folding_conditional(*condition, *if_true, *if_false),
            ComputationNode::Leaf { .. } => node,
        }
    }

    /// Handles constant folding for binary operations
    fn constant_folding_binary_op<T>(
        left: ComputationNode<T>,
        right: ComputationNode<T>,
        operation: BinaryOperation,
    ) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + Clone + PartialEq,
    {
        let left_opt = Self::constant_folding(left);
        let right_opt = Self::constant_folding(right);

        if let (
            ComputationNode::Leaf {
                constant_value: Some(left_val),
                ..
            },
            ComputationNode::Leaf {
                constant_value: Some(right_val),
                ..
            },
        ) = (&left_opt, &right_opt)
        {
            let (left_val, right_val) = (left_val.clone(), right_val.clone());
            let result = match operation {
                BinaryOperation::Add => left_val + right_val,
                BinaryOperation::Sub => left_val - right_val,
                BinaryOperation::Mul => left_val * right_val,
                BinaryOperation::Div => left_val / right_val,
            };
            return ComputationNode::constant(result);
        }

        ComputationNode::BinaryOp {
            left: Box::new(left_opt),
            right: Box::new(right_opt),
            operation,
        }
    }

    /// Handles constant folding for unary operations
    fn constant_folding_unary_op<T>(
        operand: ComputationNode<T>,
        operation: UnaryOperation<T>,
    ) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + Clone + PartialEq,
    {
        let operand_opt = Self::constant_folding(operand);

        if let ComputationNode::Leaf {
            constant_value: Some(operand_val),
            ..
        } = &operand_opt
        {
            let operand_val = operand_val.clone();
            let result = match operation {
                UnaryOperation::Map(func) => func(operand_val),
                UnaryOperation::Filter(_) => operand_val, // Filter doesn't change the value
            };
            return ComputationNode::constant(result);
        }

        ComputationNode::UnaryOp {
            operand: Box::new(operand_opt),
            operation,
        }
    }

    /// Handles constant folding for conditional operations
    fn constant_folding_conditional<T>(
        condition: ComputationNode<bool>,
        if_true: ComputationNode<T>,
        if_false: ComputationNode<T>,
    ) -> ComputationNode<T>
    where
        T: Shareable + Arithmetic + Clone + PartialEq,
    {
        let condition_opt = Self::constant_folding_bool(condition);
        let if_true_opt = Self::constant_folding(if_true);
        let if_false_opt = Self::constant_folding(if_false);

        // Check if condition is constant
        if let ComputationNode::Leaf {
            constant_value: Some(condition_val),
            ..
        } = &condition_opt
        {
            if *condition_val {
                return if_true_opt;
            }
            return if_false_opt;
        }

        ComputationNode::Conditional {
            condition: Box::new(condition_opt),
            if_true: Box::new(if_true_opt),
            if_false: Box::new(if_false_opt),
        }
    }

    /// Performs constant folding for boolean types
    fn constant_folding_bool(node: ComputationNode<bool>) -> ComputationNode<bool> {
        match node {
            ComputationNode::UnaryOp { operand, operation } => {
                Self::constant_folding_bool_unary_op(*operand, operation)
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => Self::constant_folding_bool_conditional(*condition, *if_true, *if_false),
            ComputationNode::Leaf { .. } | ComputationNode::BinaryOp { .. } => node,
        }
    }

    /// Handles constant folding for boolean unary operations
    fn constant_folding_bool_unary_op(
        operand: ComputationNode<bool>,
        operation: UnaryOperation<bool>,
    ) -> ComputationNode<bool> {
        let operand_opt = Self::constant_folding_bool(operand);

        if let ComputationNode::Leaf {
            constant_value: Some(operand_val),
            ..
        } = &operand_opt
        {
            let result = match operation {
                UnaryOperation::Map(func) => func(*operand_val),
                UnaryOperation::Filter(_) => *operand_val, // Filter doesn't change the value
            };
            return ComputationNode::constant(result);
        }

        ComputationNode::UnaryOp {
            operand: Box::new(operand_opt),
            operation,
        }
    }

    /// Handles constant folding for boolean conditional operations
    fn constant_folding_bool_conditional(
        condition: ComputationNode<bool>,
        if_true: ComputationNode<bool>,
        if_false: ComputationNode<bool>,
    ) -> ComputationNode<bool> {
        let condition_opt = Self::constant_folding_bool(condition);
        let if_true_opt = Self::constant_folding_bool(if_true);
        let if_false_opt = Self::constant_folding_bool(if_false);

        if let ComputationNode::Leaf {
            constant_value: Some(condition_val),
            ..
        } = &condition_opt
        {
            if *condition_val {
                return if_true_opt;
            }
            return if_false_opt;
        }

        ComputationNode::Conditional {
            condition: Box::new(condition_opt),
            if_true: Box::new(if_true_opt),
            if_false: Box::new(if_false_opt),
        }
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Computation graph visualizer for debugging and analysis
pub struct GraphVisualizer;

impl GraphVisualizer {
    /// Generates a DOT graph representation for visualization
    #[must_use]
    pub fn to_dot<T>(node: &ComputationNode<T>) -> String
    where
        T: Shareable,
    {
        let mut dot = String::from("digraph G {\n");
        let mut node_id = 0;
        Self::add_node_to_dot(node, &mut dot, &mut node_id);
        dot.push_str("}\n");
        dot
    }

    fn add_node_to_dot<T>(node: &ComputationNode<T>, dot: &mut String, node_id: &mut usize) -> usize
    where
        T: Shareable,
    {
        use std::fmt::Write;
        let current_id = *node_id;
        *node_id += 1;

        match node {
            ComputationNode::Leaf { .. } => {
                writeln!(dot, "  {current_id} [label=\"Leaf\", shape=circle];").unwrap();
            }
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => {
                let op_name = match operation {
                    BinaryOperation::Add => "Add",
                    BinaryOperation::Sub => "Sub",
                    BinaryOperation::Mul => "Mul",
                    BinaryOperation::Div => "Div",
                };
                writeln!(dot, "  {current_id} [label=\"{op_name}\", shape=box];").unwrap();

                let left_id = Self::add_node_to_dot(left, dot, node_id);
                let right_id = Self::add_node_to_dot(right, dot, node_id);

                writeln!(dot, "  {current_id} -> {left_id};").unwrap();
                writeln!(dot, "  {current_id} -> {right_id};").unwrap();
            }
            ComputationNode::UnaryOp { operand, .. } => {
                writeln!(dot, "  {current_id} [label=\"UnaryOp\", shape=box];").unwrap();
                let operand_id = Self::add_node_to_dot(operand, dot, node_id);
                writeln!(dot, "  {current_id} -> {operand_id};").unwrap();
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                writeln!(dot, "  {current_id} [label=\"If\", shape=diamond];").unwrap();

                let cond_id = Self::add_node_to_dot(condition, dot, node_id);
                let true_id = Self::add_node_to_dot(if_true, dot, node_id);
                let false_id = Self::add_node_to_dot(if_false, dot, node_id);

                writeln!(dot, "  {current_id} -> {cond_id} [label=\"cond\"];").unwrap();
                writeln!(dot, "  {current_id} -> {true_id} [label=\"true\"];").unwrap();
                writeln!(dot, "  {current_id} -> {false_id} [label=\"false\"];").unwrap();
            }
        }

        current_id
    }

    /// Prints a text-based representation of the computation graph
    pub fn print_tree<T>(node: &ComputationNode<T>, indent: usize)
    where
        T: Shareable,
    {
        let prefix = "  ".repeat(indent);

        match node {
            ComputationNode::Leaf { id, .. } => {
                println!("{prefix}Leaf({id})");
            }
            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => {
                let op_name = match operation {
                    BinaryOperation::Add => "Add",
                    BinaryOperation::Sub => "Sub",
                    BinaryOperation::Mul => "Mul",
                    BinaryOperation::Div => "Div",
                };
                println!("{prefix}{op_name}");
                Self::print_tree(left, indent + 1);
                Self::print_tree(right, indent + 1);
            }
            ComputationNode::UnaryOp { operand, .. } => {
                println!("{prefix}UnaryOp");
                Self::print_tree(operand, indent + 1);
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                println!("{prefix}Conditional");
                println!("{prefix}  Condition:");
                Self::print_tree(condition, indent + 2);
                println!("{prefix}  If True:");
                Self::print_tree(if_true, indent + 2);
                println!("{prefix}  If False:");
                Self::print_tree(if_false, indent + 2);
            }
        }
    }
}

/// Performance profiler for computation graphs
pub struct GraphProfiler {
    execution_times: HashMap<String, Vec<std::time::Duration>>,
}

impl GraphProfiler {
    /// Create a new profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
        }
    }

    /// Profile the execution of a computation graph
    pub fn profile_execution<T, F>(&mut self, name: &str, func: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = func();
        let duration = start.elapsed();

        self.execution_times
            .entry(name.to_string())
            .or_default()
            .push(duration);

        result
    }

    /// Get profiling statistics
    #[must_use]
    pub fn get_stats(&self, name: &str) -> Option<ProfileStats> {
        let times = self.execution_times.get(name)?;
        if times.is_empty() {
            return None;
        }

        let total: std::time::Duration = times.iter().sum();
        let count = times.len();
        let average = total / u32::try_from(count).unwrap_or(1);

        let mut sorted_times = times.clone();
        sorted_times.sort();
        let median = sorted_times[count / 2];
        let min = sorted_times[0];
        let max = sorted_times[count - 1];

        Some(ProfileStats {
            count,
            total,
            average,
            median,
            min,
            max,
        })
    }

    /// Print all profiling results
    pub fn print_report(&self) {
        println!("=== Computation Graph Profiling Report ===");
        for name in self.execution_times.keys() {
            if let Some(stats) = self.get_stats(name) {
                println!("\n{name}:");
                println!("  Count: {}", stats.count);
                println!("  Total: {:?}", stats.total);
                println!("  Average: {:?}", stats.average);
                println!("  Median: {:?}", stats.median);
                println!("  Min: {:?}", stats.min);
                println!("  Max: {:?}", stats.max);
            }
        }
    }
}

impl Default for GraphProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from profiling computation graph execution
#[derive(Debug, Clone)]
pub struct ProfileStats {
    /// Number of executions
    pub count: usize,
    /// Total execution time
    pub total: std::time::Duration,
    /// Average execution time
    pub average: std::time::Duration,
    /// Median execution time
    pub median: std::time::Duration,
    /// Minimum execution time
    pub min: std::time::Duration,
    /// Maximum execution time
    pub max: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uncertain;

    #[test]
    fn test_sample_context_memoization() {
        let mut context = SampleContext::new();
        let id = uuid::Uuid::new_v4();

        // Set a value
        context.set_value(id, 42.0);

        // Should get the same value back
        assert_eq!(context.get_value::<f64>(&id), Some(42.0));

        // Different ID should return None
        let other_id = uuid::Uuid::new_v4();
        assert_eq!(context.get_value::<f64>(&other_id), None);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_computation_node_evaluation() {
        let left = ComputationNode::leaf(|| 5.0);
        let right = ComputationNode::leaf(|| 3.0);
        let add_node = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        let result = add_node.evaluate_fresh();
        assert_eq!(result, 8.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_shared_variable_memoization() {
        let mut context = SampleContext::new();

        // Create a leaf node
        let leaf_id = uuid::Uuid::new_v4();
        let leaf = ComputationNode::Leaf {
            id: leaf_id,
            sample: Arc::new(rand::random::<f64>),
            constant_value: None,
        };

        // Evaluate twice with the same context
        let val1 = leaf.evaluate(&mut context).unwrap();
        let val2 = leaf.evaluate(&mut context).unwrap();

        // Should get the same value due to memoization
        assert_eq!(val1, val2);
    }

    #[test]
    fn test_computation_graph_metrics() {
        let left = ComputationNode::leaf(|| 1.0);
        let right = ComputationNode::leaf(|| 2.0);
        let add_node = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        assert_eq!(add_node.node_count(), 3); // 2 leaves + 1 binary op
        assert_eq!(add_node.depth(), 2); // Binary op -> leaves
        assert!(!add_node.has_conditionals());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_conditional_node() {
        let condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| 10.0);
        let if_false = ComputationNode::leaf(|| 20.0);

        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let result = conditional.evaluate_fresh();
        assert_eq!(result, 10.0); // Should pick the true branch
        assert!(conditional.has_conditionals());
    }

    #[test]
    fn test_graph_visualizer_dot_output() {
        let left = ComputationNode::leaf(|| 1.0);
        let right = ComputationNode::leaf(|| 2.0);
        let add_node = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        let dot = GraphVisualizer::to_dot(&add_node);

        assert!(dot.contains("digraph G"));
        assert!(dot.contains("Add"));
        assert!(dot.contains("Leaf"));
    }

    #[test]
    fn test_profiler() {
        let mut profiler = GraphProfiler::new();

        let result = profiler.profile_execution("test", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        let stats = profiler.get_stats("test").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_complex_computation_graph() {
        // Test (x + y) * (x - y) where x and y are uncertain values
        let x = Uncertain::normal(5.0, 1.0).unwrap();
        let y = Uncertain::normal(3.0, 1.0).unwrap();

        // This should build a computation graph
        let sum = x.clone() + y.clone();
        let diff = x - y;
        let product = sum * diff;

        // The computation graph should have multiple nodes
        assert!(product.node.node_count() > 5);
        assert!(product.node.depth() > 2);

        // Should be able to evaluate multiple times
        let sample1 = product.sample();
        let sample2 = product.sample();

        // Values should be reasonable (approximately (5+3)*(5-3) = 16 with some variance)
        // With normal distributions (5±1) and (3±1), the result can vary significantly
        // Allow for wide variance due to unbounded normal distributions and multiplication
        // Statistical analysis shows 99.9% of values fall within [-50, 150]
        assert!(sample1 > -50.0 && sample1 < 150.0);
        assert!(sample2 > -50.0 && sample2 < 150.0);
    }

    #[test]
    fn test_sample_context_clear() {
        let mut context = SampleContext::new();
        let id = uuid::Uuid::new_v4();

        context.set_value(id, 42.0);
        assert_eq!(context.len(), 1);
        assert!(!context.is_empty());

        context.clear();
        assert_eq!(context.len(), 0);
        assert!(context.is_empty());
        assert_eq!(context.get_value::<f64>(&id), None);
    }

    #[test]
    fn test_sample_context_default() {
        let context = SampleContext::default();
        assert!(context.is_empty());
        assert_eq!(context.len(), 0);
    }

    #[test]
    fn test_evaluate_err_on_binary_op() {
        let left = ComputationNode::leaf(|| 1.0);
        let right = ComputationNode::leaf(|| 2.0);
        let binary_op = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        let mut context = SampleContext::new();
        let err = binary_op.evaluate(&mut context).unwrap_err();
        assert_eq!(
            err,
            UncertainError::unsupported_node("Leaf or UnaryOp", "BinaryOp")
        );
    }

    #[test]
    fn test_evaluate_err_on_conditional() {
        let condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| 10.0);
        let if_false = ComputationNode::leaf(|| 20.0);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let mut context = SampleContext::new();
        let err = conditional.evaluate(&mut context).unwrap_err();
        assert_eq!(
            err,
            UncertainError::unsupported_node("Leaf or UnaryOp", "Conditional")
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_evaluate_arithmetic_handles_conditional() {
        let condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| 10.0);
        let if_false = ComputationNode::leaf(|| 20.0);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let mut context = SampleContext::new();
        let result = conditional.evaluate_arithmetic(&mut context).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_evaluate_bool_err_on_binary_op() {
        let left = ComputationNode::leaf(|| true);
        let right = ComputationNode::leaf(|| false);
        let binary_op = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        let mut context = SampleContext::new();
        let err = binary_op.evaluate_bool(&mut context).unwrap_err();
        assert_eq!(
            err,
            UncertainError::unsupported_node("Leaf, UnaryOp, or Conditional", "BinaryOp")
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_unary_map_operation() {
        let operand = ComputationNode::leaf(|| 5.0);
        let mapped = ComputationNode::map(operand, |x| x * 2.0);

        let result = mapped.evaluate_fresh();
        assert_eq!(result, 10.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_unary_filter_operation() {
        let operand = ComputationNode::leaf(|| 42.0);
        let filtered = ComputationNode::UnaryOp {
            operand: Box::new(operand),
            operation: UnaryOperation::Filter(Arc::new(|x: &f64| *x > 0.0)),
        };

        let mut context = SampleContext::new();
        let result = filtered.evaluate(&mut context).unwrap();
        assert_eq!(result, 42.0); // Filter currently just passes through
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_evaluate_unary_map_operation() {
        let operand = ComputationNode::leaf(|| 5.0);
        let mapped = ComputationNode::map(operand, |x| x * 2.0);

        let mut context = SampleContext::new();
        let result = mapped.evaluate(&mut context).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_evaluate_arithmetic_unary_filter_operation() {
        let operand = ComputationNode::leaf(|| 42.0);
        let filtered = ComputationNode::UnaryOp {
            operand: Box::new(operand),
            operation: UnaryOperation::Filter(Arc::new(|x: &f64| *x > 0.0)),
        };

        let mut context = SampleContext::new();
        let result = filtered.evaluate_arithmetic(&mut context).unwrap();
        assert_eq!(result, 42.0); // Filter currently just passes through
    }

    #[test]
    fn test_graph_optimizer() {
        let node = ComputationNode::leaf(|| 1.0);
        let mut optimizer = GraphOptimizer::new();
        let optimized_node = optimizer.optimize(node);
        assert_eq!(optimized_node.node_count(), 1);
    }

    #[test]
    fn test_common_subexpression_elimination() {
        let mut optimizer = GraphOptimizer::new();

        // Create a common subexpression: (x + y) * (x + y)
        let x = ComputationNode::leaf(|| 2.0);
        let y = ComputationNode::leaf(|| 3.0);
        let sum = ComputationNode::binary_op(x.clone(), y.clone(), BinaryOperation::Add);

        // Create the expression: (x + y) * (x + y)
        let expr = ComputationNode::binary_op(sum.clone(), sum, BinaryOperation::Mul);

        // First optimization should cache the sum subexpression
        let optimized1 = optimizer.eliminate_common_subexpressions(expr.clone());

        // Second optimization should reuse the cached sum subexpression
        let optimized2 = optimizer.eliminate_common_subexpressions(expr);

        // Both should produce the same result
        let result1: f64 = optimized1.evaluate_fresh();
        let result2: f64 = optimized2.evaluate_fresh();
        assert!((result1 - result2).abs() < f64::EPSILON);

        // The cache should contain the sum subexpression
        assert!(optimizer.cache_size() > 0);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_common_subexpression_elimination_complex() {
        let mut optimizer = GraphOptimizer::new();

        // Create a more complex expression with multiple common subexpressions
        let a = ComputationNode::leaf(|| 1.0);
        let b = ComputationNode::leaf(|| 2.0);
        let c = ComputationNode::leaf(|| 3.0);

        // Common subexpression: a + b
        let sum_ab = ComputationNode::binary_op(a.clone(), b.clone(), BinaryOperation::Add);

        // Expression: (a + b) * (a + b) + (a + b) * c
        let expr1 =
            ComputationNode::binary_op(sum_ab.clone(), sum_ab.clone(), BinaryOperation::Mul);
        let expr2 = ComputationNode::binary_op(sum_ab.clone(), c.clone(), BinaryOperation::Mul);
        let final_expr = ComputationNode::binary_op(expr1, expr2, BinaryOperation::Add);

        let optimized = optimizer.eliminate_common_subexpressions(final_expr);

        // Should produce correct result
        let result: f64 = optimized.evaluate_fresh();
        let expected = (1.0 + 2.0) * (1.0 + 2.0) + (1.0 + 2.0) * 3.0;
        assert!((result - expected).abs() < f64::EPSILON);

        // Cache should contain every distinct subexpression visited: leaves a, b, c,
        // sum_ab, expr1, expr2, final_expr — 7 in total — with a cache hit for each of
        // sum_ab's 2 repeat visits (inside expr1 and expr2).
        assert_eq!(optimizer.cache_size(), 7);
        assert_eq!(optimizer.cse_hits(), 2);
    }

    #[test]
    fn test_cse_unifies_clones_of_the_same_leaf() {
        // x.clone() + x.clone(): both operands are the SAME random variable (shared
        // leaf id), so within one evaluation they must unify to a single draw, making
        // the sum exactly 2x — deterministically, regardless of what x samples.
        let x = Uncertain::normal(5.0, 2.0).unwrap();
        let expr = x.clone() + x.clone();

        let mut optimizer = GraphOptimizer::new();
        let optimized_node = optimizer.eliminate_common_subexpressions(expr.node.clone());

        let mut context = SampleContext::new();
        let x_val = x.node.evaluate_arithmetic(&mut context).unwrap();
        let sum_val = optimized_node.evaluate_arithmetic(&mut context).unwrap();

        assert!((sum_val - 2.0 * x_val).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cse_never_unifies_independently_constructed_leaves() {
        // Two independently-built normal(0, 1) leaves are different random variables:
        // unifying them into "2x" would change Var(x + y) from 2 to 4.
        let x = Uncertain::normal(0.0, 1.0).unwrap();
        let y = Uncertain::normal(0.0, 1.0).unwrap();
        let expr = x + y;

        let mut optimizer = GraphOptimizer::new();
        let optimized_node = optimizer.eliminate_common_subexpressions(expr.node.clone());
        let optimized = Uncertain::with_node(optimized_node);

        let variance = optimized.variance(50_000).unwrap();
        assert!(
            (variance - 2.0).abs() < 0.4,
            "expected variance ~2 (independent sum), got {variance}"
        );
    }

    #[test]
    fn test_cse_deduplicates_deterministic_subexpression_built_twice() {
        // The same subexpression, built twice from the same leaves, must collapse to
        // one shared node rather than two structurally-identical-but-distinct copies.
        let a = ComputationNode::leaf(|| 1.0);
        let b = ComputationNode::leaf(|| 2.0);

        let built_once = ComputationNode::binary_op(a.clone(), b.clone(), BinaryOperation::Add);
        let built_again = ComputationNode::binary_op(a, b, BinaryOperation::Add);

        let mut optimizer = GraphOptimizer::new();
        let _ = optimizer.eliminate_common_subexpressions(built_once);
        assert_eq!(optimizer.cse_hits(), 0);

        let _ = optimizer.eliminate_common_subexpressions(built_again);
        assert_eq!(
            optimizer.cse_hits(),
            1,
            "rebuilding the identical expression from the same leaves should hit the cache"
        );
    }

    #[test]
    fn test_cse_does_not_confuse_unary_ops_with_different_closures() {
        // Under the old hash (which ignored a UnaryOp's closure entirely), these two
        // nodes were an engineered hash collision: same operand, same "unary" shape,
        // different function. A naive hash-only cache would return f's cached result
        // for g's lookup. The structural key must distinguish them by closure identity.
        let leaf = ComputationNode::leaf(|| 3.0);
        let mapped_add_one = ComputationNode::map(leaf.clone(), |v| v + 1.0);
        let mapped_times_ten = ComputationNode::map(leaf, |v| v * 10.0);

        assert_ne!(
            mapped_add_one.structural_hash(),
            mapped_times_ten.structural_hash(),
            "different closures over the same operand must not hash the same"
        );

        let mut optimizer = GraphOptimizer::new();
        let opt_add_one = optimizer.eliminate_common_subexpressions(mapped_add_one);
        let opt_times_ten = optimizer.eliminate_common_subexpressions(mapped_times_ten);

        let result_add_one: f64 = opt_add_one.evaluate_fresh();
        let result_times_ten: f64 = opt_times_ten.evaluate_fresh();
        assert!((result_add_one - 4.0).abs() < f64::EPSILON);
        assert!((result_times_ten - 30.0).abs() < f64::EPSILON);
        // Exactly one hit: the shared `leaf` operand (same id, legitimately the same
        // variable) unifies, but the two UnaryOp nodes wrapping it — different
        // closures — never do.
        assert_eq!(optimizer.cse_hits(), 1);
    }

    #[test]
    fn test_structural_key_lookup_is_correct_even_under_a_forced_hash_collision() {
        // A HashMap keyed by StructuralKey is collision-safe by construction (Eq is
        // always checked before a bucket entry is returned), independent of hash
        // quality. Prove it by forcing every key into the same bucket with a hasher
        // that discards all input, then confirming lookups never cross-contaminate.
        #[derive(Default)]
        struct AlwaysCollideHasher;
        impl std::hash::Hasher for AlwaysCollideHasher {
            fn finish(&self) -> u64 {
                0
            }
            fn write(&mut self, _bytes: &[u8]) {}
        }

        let mut map: HashMap<
            StructuralKey,
            &'static str,
            std::hash::BuildHasherDefault<AlwaysCollideHasher>,
        > = HashMap::default();

        let leaf_a = ComputationNode::<f64>::leaf(|| 1.0);
        let leaf_b = ComputationNode::<f64>::leaf(|| 2.0);
        let key_a = leaf_a.structural_key();
        let key_b = leaf_b.structural_key();
        assert_ne!(key_a, key_b);

        map.insert(key_a.clone(), "a");
        map.insert(key_b.clone(), "b");

        assert_eq!(map.get(&key_a), Some(&"a"));
        assert_eq!(map.get(&key_b), Some(&"b"));
    }

    #[test]
    fn test_identity_operation_elimination() {
        // Test x + 0 = x
        let x = ComputationNode::leaf(|| 5.0);
        let zero = ComputationNode::constant(0.0);
        let add_zero = ComputationNode::binary_op(x.clone(), zero, BinaryOperation::Add);

        let optimized = GraphOptimizer::eliminate_identity_operations(add_zero);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);

        // Test x * 1 = x
        let one = ComputationNode::constant(1.0);
        let mul_one = ComputationNode::binary_op(x.clone(), one, BinaryOperation::Mul);

        let optimized = GraphOptimizer::eliminate_identity_operations(mul_one);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);

        // Test x - 0 = x
        let zero2 = ComputationNode::constant(0.0);
        let sub_zero = ComputationNode::binary_op(x.clone(), zero2, BinaryOperation::Sub);

        let optimized = GraphOptimizer::eliminate_identity_operations(sub_zero);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);

        // Test x / 1 = x
        let one2 = ComputationNode::constant(1.0);
        let div_one = ComputationNode::binary_op(x.clone(), one2, BinaryOperation::Div);

        let optimized = GraphOptimizer::eliminate_identity_operations(div_one);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);

        // Test x * 0 = 0
        let zero3 = ComputationNode::constant(0.0);
        let mul_zero = ComputationNode::binary_op(x.clone(), zero3, BinaryOperation::Mul);

        let optimized = GraphOptimizer::eliminate_identity_operations(mul_zero);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_constant_folding() {
        // Test constant addition: 2 + 3 = 5
        let two = ComputationNode::constant(2.0);
        let three = ComputationNode::constant(3.0);
        let add_const = ComputationNode::binary_op(two, three, BinaryOperation::Add);

        let optimized = GraphOptimizer::constant_folding(add_const);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);

        // Test constant multiplication: 4 * 5 = 20
        let four = ComputationNode::constant(4.0);
        let five = ComputationNode::constant(5.0);
        let mul_const = ComputationNode::binary_op(four, five, BinaryOperation::Mul);

        let optimized = GraphOptimizer::constant_folding(mul_const);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 20.0).abs() < f64::EPSILON);

        // Test constant division: 10 / 2 = 5
        let ten = ComputationNode::constant(10.0);
        let two_div = ComputationNode::constant(2.0);
        let div_const = ComputationNode::binary_op(ten, two_div, BinaryOperation::Div);

        let optimized = GraphOptimizer::constant_folding(div_const);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);

        // Test constant subtraction: 8 - 3 = 5
        let eight = ComputationNode::constant(8.0);
        let three_sub = ComputationNode::constant(3.0);
        let sub_const = ComputationNode::binary_op(eight, three_sub, BinaryOperation::Sub);

        let optimized = GraphOptimizer::constant_folding(sub_const);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_constant_folding_conditional() {
        // Test constant condition: if true then 10 else 20 = 10
        let true_condition = ComputationNode::constant(true);
        let if_true = ComputationNode::constant(10.0);
        let if_false = ComputationNode::constant(20.0);
        let conditional = ComputationNode::conditional(true_condition, if_true, if_false);

        let optimized = GraphOptimizer::constant_folding(conditional);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 10.0).abs() < f64::EPSILON);

        // Test constant condition: if false then 10 else 20 = 20
        let false_condition = ComputationNode::constant(false);
        let if_true2 = ComputationNode::constant(10.0);
        let if_false2 = ComputationNode::constant(20.0);
        let conditional2 = ComputationNode::conditional(false_condition, if_true2, if_false2);

        let optimized = GraphOptimizer::constant_folding(conditional2);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_constant_folding_unary() {
        // Test constant unary operation: map(|x| x * 2) on constant 5 = 10
        let five = ComputationNode::constant(5.0);
        let double = ComputationNode::map(five, |x| x * 2.0);

        let optimized = GraphOptimizer::constant_folding(double);
        let result: f64 = optimized.evaluate_fresh();
        assert!((result - 10.0).abs() < f64::EPSILON);
    }

    // Spec 07 acceptance tests: constant folding/identity elimination must decide
    // constancy structurally, never by sampling.

    #[test]
    fn test_low_entropy_distribution_never_treated_as_constant() {
        // Simulates `bernoulli(0.99)` mapped to `f64`: P(1.0) = 0.99. Under the old
        // sample-3x-and-compare semantics, ~96% of runs would see three consecutive
        // 1.0s and be silently (and incorrectly) folded into a constant. Structural
        // constancy never inspects sampled values at all, so this is now a
        // deterministic guarantee rather than a statistical one -- true regardless
        // of how "constant-looking" any given run's samples happen to be.
        let low_entropy = ComputationNode::leaf(|| {
            if rand::random::<f64>() < 0.99 {
                1.0
            } else {
                0.0
            }
        });
        assert!(!GraphOptimizer::is_constant_zero(&low_entropy));
        assert!(!GraphOptimizer::is_constant_one(&low_entropy));

        // Downstream: an identity check against it must not fire either.
        let x = ComputationNode::leaf(|| 5.0);
        assert!(GraphOptimizer::check_addition_identities(&x, &low_entropy).is_none());
    }

    #[test]
    fn test_point_zero_addition_folds_to_the_other_operand() {
        let x = Uncertain::normal(5.0, 1.0).unwrap();
        let expr = x.clone() + Uncertain::point(0.0);

        let optimized = GraphOptimizer::new().optimize(expr.node.clone());

        // x + 0 collapses back down to just x's own node -- same shape, same size.
        assert_eq!(optimized.node_count(), x.node.node_count());
        assert!(matches!(optimized, ComputationNode::Leaf { .. }));
    }

    #[test]
    fn test_point_times_point_folds_to_a_single_constant() {
        let expr: Uncertain<f64> = Uncertain::point(2.0) * Uncertain::point(3.0);

        let optimized = GraphOptimizer::new().optimize(expr.node.clone());

        match &optimized {
            ComputationNode::Leaf {
                constant_value: Some(value),
                ..
            } => assert!((value - 6.0).abs() < f64::EPSILON),
            _ => panic!("expected a folded constant leaf"),
        }
    }

    #[test]
    fn test_tiny_variance_normal_is_not_folded() {
        // A `normal(0, 1e-12)` has genuinely nonzero variance -- it must never be
        // folded, no matter how close to a point mass it looks. `Uncertain::normal`
        // always builds a plain (non-`constant`) leaf, so this holds structurally.
        let tiny_variance = Uncertain::normal(0.0, 1e-12).unwrap();
        assert!(!GraphOptimizer::is_constant_zero(&tiny_variance.node));
    }

    #[test]
    fn test_optimized_and_unoptimized_graphs_sample_identically_under_same_seed() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // A moderately complex expression mixing genuine identity/constant
        // opportunities (`+ point(0.0)`, `* point(1.0)`) with real randomness.
        let x = Uncertain::normal(3.0, 1.0).unwrap();
        let y = Uncertain::normal(2.0, 0.5).unwrap();
        let expr = ((x.clone() + y.clone()) + Uncertain::point(0.0))
            * (x.clone() - y.clone())
            * Uncertain::point(1.0);

        let optimized_node = GraphOptimizer::new().optimize(expr.node.clone());
        let optimized = Uncertain::with_node(optimized_node);

        let mut rng_a = ChaCha8Rng::seed_from_u64(2024);
        let mut rng_b = ChaCha8Rng::seed_from_u64(2024);

        for _ in 0..50 {
            let unoptimized_sample = expr.sample_with(&mut rng_a);
            let optimized_sample = optimized.sample_with(&mut rng_b);
            assert!((unoptimized_sample - optimized_sample).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_graph_visualizer_print_tree() {
        let left = ComputationNode::leaf(|| 1.0);
        let right = ComputationNode::leaf(|| 2.0);
        let add_node = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        // This mainly tests that print_tree doesn't panic
        GraphVisualizer::print_tree(&add_node, 0);
    }

    #[test]
    fn test_graph_visualizer_dot_conditional() {
        let condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| 10.0);
        let if_false = ComputationNode::leaf(|| 20.0);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let dot = GraphVisualizer::to_dot(&conditional);

        assert!(dot.contains("digraph G"));
        assert!(dot.contains("If"));
        assert!(dot.contains("diamond"));
        assert!(dot.contains("cond"));
        assert!(dot.contains("true"));
        assert!(dot.contains("false"));
    }

    #[test]
    fn test_graph_visualizer_dot_unary_op() {
        let operand = ComputationNode::leaf(|| 5.0);
        let unary = ComputationNode::map(operand, |x| x * 2.0);

        let dot = GraphVisualizer::to_dot(&unary);

        assert!(dot.contains("digraph G"));
        assert!(dot.contains("UnaryOp"));
        assert!(dot.contains("Leaf"));
    }

    #[test]
    fn test_profiler_default() {
        let profiler = GraphProfiler::default();
        assert!(profiler.get_stats("nonexistent").is_none());
    }

    #[test]
    fn test_profiler_get_stats_nonexistent() {
        let profiler = GraphProfiler::new();
        assert!(profiler.get_stats("nonexistent").is_none());
    }

    #[test]
    fn test_profiler_multiple_executions() {
        let mut profiler = GraphProfiler::new();

        profiler.profile_execution("test", || {
            std::thread::sleep(std::time::Duration::from_millis(1));
        });
        profiler.profile_execution("test", || {
            std::thread::sleep(std::time::Duration::from_millis(2));
        });
        profiler.profile_execution("test", || {
            std::thread::sleep(std::time::Duration::from_millis(3));
        });

        let stats = profiler.get_stats("test").unwrap();
        assert_eq!(stats.count, 3);
        assert!(stats.min <= stats.median);
        assert!(stats.median <= stats.max);
        assert!(stats.average.as_nanos() > 0);

        profiler.print_report();
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_conditional_evaluation_false_branch() {
        let condition = ComputationNode::leaf(|| false);
        let if_true = ComputationNode::leaf(|| 10.0);
        let if_false = ComputationNode::leaf(|| 20.0);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let result = conditional.evaluate_fresh();
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_bool_conditional_evaluation() {
        let condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| true);
        let if_false = ComputationNode::leaf(|| false);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let mut context = SampleContext::new();
        let result = conditional.evaluate_bool(&mut context).unwrap();
        assert!(result);
    }

    #[test]
    fn test_bool_unary_operation() {
        let operand = ComputationNode::leaf(|| true);
        let mapped = ComputationNode::map(operand, |x| !x);

        let mut context = SampleContext::new();
        let result = mapped.evaluate_bool(&mut context).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_bool_conditional_evaluation_false_branch() {
        let condition = ComputationNode::leaf(|| false);
        let if_true = ComputationNode::leaf(|| true);
        let if_false = ComputationNode::leaf(|| false);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let mut context = SampleContext::new();
        let result = conditional.evaluate_bool(&mut context).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_bool_leaf_memoization() {
        let mut context = SampleContext::new();
        let leaf_id = uuid::Uuid::new_v4();
        let leaf = ComputationNode::Leaf {
            id: leaf_id,
            sample: Arc::new(rand::random::<bool>),
            constant_value: None,
        };

        let val1 = leaf.evaluate_bool(&mut context).unwrap();
        let val2 = leaf.evaluate_bool(&mut context).unwrap();
        assert_eq!(val1, val2);
    }

    #[test]
    fn test_bool_unary_filter_operation() {
        let operand = ComputationNode::leaf(|| true);
        let filtered = ComputationNode::UnaryOp {
            operand: Box::new(operand),
            operation: UnaryOperation::Filter(Arc::new(|x: &bool| *x)),
        };

        let mut context = SampleContext::new();
        let result = filtered.evaluate_bool(&mut context).unwrap();
        assert!(result); // Filter currently just passes through
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_binary_operations_subtraction() {
        let left = ComputationNode::leaf(|| 10.0);
        let right = ComputationNode::leaf(|| 3.0);
        let sub_node = ComputationNode::binary_op(left, right, BinaryOperation::Sub);

        let result = sub_node.evaluate_fresh();
        assert_eq!(result, 7.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_binary_operations_multiplication() {
        let left = ComputationNode::leaf(|| 4.0);
        let right = ComputationNode::leaf(|| 5.0);
        let mul_node = ComputationNode::binary_op(left, right, BinaryOperation::Mul);

        let result = mul_node.evaluate_fresh();
        assert_eq!(result, 20.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_binary_operations_division() {
        let left = ComputationNode::leaf(|| 15.0);
        let right = ComputationNode::leaf(|| 3.0);
        let div_node = ComputationNode::binary_op(left, right, BinaryOperation::Div);

        let result = div_node.evaluate_fresh();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_nested_conditional_depth() {
        let condition1 = ComputationNode::leaf(|| true);
        let condition2 = ComputationNode::leaf(|| false);
        let leaf1 = ComputationNode::leaf(|| 1.0);
        let _leaf2 = ComputationNode::leaf(|| 2.0);
        let leaf3 = ComputationNode::leaf(|| 3.0);
        let leaf4 = ComputationNode::leaf(|| 4.0);

        let inner_conditional = ComputationNode::conditional(condition2, leaf3, leaf4);
        let outer_conditional = ComputationNode::conditional(condition1, leaf1, inner_conditional);

        assert_eq!(outer_conditional.depth(), 3);
        assert_eq!(outer_conditional.node_count(), 7);
        assert!(outer_conditional.has_conditionals());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_evaluate_arithmetic_total_dispatch() {
        let condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| 42.0);
        let if_false = ComputationNode::leaf(|| 24.0);
        let conditional = ComputationNode::conditional(condition, if_true, if_false);

        let mut context = SampleContext::new();
        let result = conditional.evaluate_arithmetic(&mut context).unwrap();
        assert_eq!(result, 42.0);

        let leaf = ComputationNode::leaf(|| 99.0);
        let result = leaf.evaluate_arithmetic(&mut context).unwrap();
        assert_eq!(result, 99.0);
    }

    #[test]
    fn test_sample_context_different_types() {
        let mut context = SampleContext::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        context.set_value(id1, 42.0_f64);
        context.set_value(id2, 100_i32);

        assert_eq!(context.get_value::<f64>(&id1), Some(42.0));
        assert_eq!(context.get_value::<i32>(&id2), Some(100));
        assert_eq!(context.get_value::<f64>(&id2), None); // Wrong type
        assert_eq!(context.get_value::<i32>(&id1), None); // Wrong type

        assert_eq!(context.len(), 2);
    }

    #[test]
    fn test_profile_stats_debug() {
        let stats = ProfileStats {
            count: 5,
            total: std::time::Duration::from_millis(100),
            average: std::time::Duration::from_millis(20),
            median: std::time::Duration::from_millis(18),
            min: std::time::Duration::from_millis(15),
            max: std::time::Duration::from_millis(30),
        };

        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("ProfileStats"));

        let cloned = stats.clone();
        assert_eq!(cloned.count, stats.count);
    }

    #[test]
    fn test_with_caching_strategy_sets_strategy() {
        let context = SampleContext::with_caching_strategy(CachingStrategy::Aggressive);
        let trivial = ComputationNode::leaf(|| 1.0);
        // Aggressive => always cache, even for a trivial leaf that Adaptive/Conservative
        // would refuse. Distinguishes from `Default::default()` (which is Adaptive).
        assert!(context.should_cache_node(&trivial));
    }

    #[test]
    fn test_should_cache_node_conservative_boundary() {
        let context = SampleContext::with_caching_strategy(CachingStrategy::Conservative);

        let depth2 = ComputationNode::binary_op(
            ComputationNode::leaf(|| 1.0),
            ComputationNode::leaf(|| 2.0),
            BinaryOperation::Add,
        );
        assert_eq!(depth2.depth(), 2);
        assert!(!context.should_cache_node(&depth2));

        let depth3 =
            ComputationNode::binary_op(depth2, ComputationNode::leaf(|| 3.0), BinaryOperation::Add);
        assert_eq!(depth3.depth(), 3);
        assert!(context.should_cache_node(&depth3));
    }

    #[test]
    fn test_should_cache_node_adaptive_boundary() {
        let context = SampleContext::with_caching_strategy(CachingStrategy::Adaptive);

        let complexity5 = ComputationNode::map(
            ComputationNode::binary_op(
                ComputationNode::leaf(|| 1.0),
                ComputationNode::leaf(|| 2.0),
                BinaryOperation::Add,
            ),
            |x| x,
        );
        assert_eq!(complexity5.compute_complexity(), 5);
        assert!(!context.should_cache_node(&complexity5));

        let complexity6 = ComputationNode::binary_op(
            ComputationNode::map(ComputationNode::leaf(|| 1.0), |x| x),
            ComputationNode::map(ComputationNode::leaf(|| 2.0), |x| x),
            BinaryOperation::Add,
        );
        assert_eq!(complexity6.compute_complexity(), 6);
        assert!(context.should_cache_node(&complexity6));
    }

    #[test]
    fn test_adaptive_sampling_get_set() {
        let mut context = SampleContext::new();
        assert_eq!(context.adaptive_sampling().min_samples, 100);

        let custom = AdaptiveSampling {
            min_samples: 42,
            max_samples: 999,
            error_threshold: 0.05,
            growth_factor: 2.0,
        };
        context.set_adaptive_sampling(custom);
        assert_eq!(context.adaptive_sampling().min_samples, 42);
        assert_eq!(context.adaptive_sampling().max_samples, 999);
    }

    #[test]
    fn test_node_shape_metrics_leaf() {
        let leaf = ComputationNode::leaf(|| 1.0);
        assert_eq!(leaf.node_count(), 1);
        assert_eq!(leaf.depth(), 1);
        assert_eq!(leaf.compute_complexity(), 1);
        assert!(!leaf.has_conditionals());
    }

    #[test]
    fn test_node_shape_metrics_binary_op() {
        let node = ComputationNode::binary_op(
            ComputationNode::leaf(|| 1.0),
            ComputationNode::leaf(|| 2.0),
            BinaryOperation::Add,
        );
        assert_eq!(node.node_count(), 3);
        assert_eq!(node.depth(), 2);
        assert_eq!(node.compute_complexity(), 4);
        assert!(!node.has_conditionals());
    }

    #[test]
    fn test_node_shape_metrics_unary_op() {
        let node = ComputationNode::map(ComputationNode::leaf(|| 1.0), |x| x * 2.0);
        assert_eq!(node.node_count(), 2);
        assert_eq!(node.depth(), 2);
        assert_eq!(node.compute_complexity(), 2);
        assert!(!node.has_conditionals());
    }

    #[test]
    fn test_node_shape_metrics_nested_binary_op() {
        let inner = ComputationNode::binary_op(
            ComputationNode::leaf(|| 1.0),
            ComputationNode::leaf(|| 2.0),
            BinaryOperation::Add,
        );
        let outer =
            ComputationNode::binary_op(inner, ComputationNode::leaf(|| 3.0), BinaryOperation::Mul);
        assert_eq!(outer.node_count(), 5);
        assert_eq!(outer.depth(), 3);
        assert_eq!(outer.compute_complexity(), 7);
    }

    #[test]
    fn test_node_shape_metrics_conditional() {
        let node = ComputationNode::conditional(
            ComputationNode::leaf(|| true),
            ComputationNode::leaf(|| 1.0),
            ComputationNode::leaf(|| 2.0),
        );
        assert_eq!(node.node_count(), 4);
        assert_eq!(node.depth(), 2);
        assert_eq!(node.compute_complexity(), 8);
        assert!(node.has_conditionals());
    }

    #[test]
    fn test_has_conditionals_propagates_through_binary_op() {
        // Left is a conditional (true), right is a plain leaf (false): only `||`
        // (not `&&`) makes this observably true.
        let conditional = ComputationNode::conditional(
            ComputationNode::leaf(|| true),
            ComputationNode::leaf(|| 1.0),
            ComputationNode::leaf(|| 2.0),
        );
        let wrapped = ComputationNode::binary_op(
            conditional,
            ComputationNode::leaf(|| 3.0),
            BinaryOperation::Add,
        );
        assert!(wrapped.has_conditionals());
    }

    #[test]
    fn test_check_addition_identities_both_orders_and_no_match() {
        let x = ComputationNode::leaf(|| 5.0);
        let zero = ComputationNode::constant(0.0);
        let nonzero = ComputationNode::constant(3.0);
        // The "x + 0" arm matches whenever the RIGHT side is structurally constant
        // zero, regardless of the left side, so it shadows the "0 + x" arm unless
        // the right side is something that isn't.
        let non_constant =
            ComputationNode::binary_op(x.clone(), nonzero.clone(), BinaryOperation::Add);

        // x + 0 = x
        assert!(GraphOptimizer::check_addition_identities(&x, &zero).is_some());
        // 0 + x = x
        assert!(GraphOptimizer::check_addition_identities(&zero, &non_constant).is_some());
        // neither side is zero: no identity applies
        assert!(GraphOptimizer::check_addition_identities(&x, &nonzero).is_none());
        // left is constant but not zero, right isn't constant at all: the "0 + x"
        // arm's guard must actually check is_constant_zero, not always match.
        assert!(GraphOptimizer::check_addition_identities(&nonzero, &non_constant).is_none());
    }

    #[test]
    fn test_check_addition_identities_never_fires_on_sampled_zero() {
        // Soundness: a leaf that always happens to sample 0.0 is not structurally
        // constant (it wasn't built via `ComputationNode::constant`), so it must
        // never be folded away -- constancy is never decided by sampling and
        // comparing, since a low-entropy distribution could look constant by chance.
        let x = ComputationNode::leaf(|| 5.0);
        let looks_like_zero = ComputationNode::leaf(|| 0.0);
        assert!(GraphOptimizer::check_addition_identities(&x, &looks_like_zero).is_none());
    }

    #[test]
    fn test_check_subtraction_identities_match_and_no_match() {
        let x = ComputationNode::leaf(|| 5.0);
        let zero = ComputationNode::constant(0.0);
        let nonzero = ComputationNode::constant(3.0);

        // x - 0 = x
        assert!(GraphOptimizer::check_subtraction_identities(&x, &zero).is_some());
        assert!(GraphOptimizer::check_subtraction_identities(&x, &nonzero).is_none());
    }

    #[test]
    fn test_check_multiplication_identities_all_orders_and_no_match() {
        let x = ComputationNode::leaf(|| 5.0);
        let zero = ComputationNode::constant(0.0);
        let one = ComputationNode::constant(1.0);
        let nonzero = ComputationNode::constant(3.0);
        // Same shadowing consideration as addition: the "x op constant" arms match
        // whenever the right side is structurally constant, so the "constant op x"
        // arms need a non-constant right side to be reachable.
        let non_constant =
            ComputationNode::binary_op(x.clone(), nonzero.clone(), BinaryOperation::Add);

        // x * 0 = 0
        assert!(GraphOptimizer::check_multiplication_identities(&x, &zero).is_some());
        // 0 * x = 0
        assert!(GraphOptimizer::check_multiplication_identities(&zero, &non_constant).is_some());
        // x * 1 = x
        assert!(GraphOptimizer::check_multiplication_identities(&x, &one).is_some());
        // 1 * x = x
        assert!(GraphOptimizer::check_multiplication_identities(&one, &non_constant).is_some());
        // no identity
        assert!(GraphOptimizer::check_multiplication_identities(&x, &nonzero).is_none());
        // left is constant but neither zero nor one, right isn't constant at all:
        // both the "0 * x" and "1 * x" arms' guards must actually check their
        // respective is_constant_zero/is_constant_one, not always match.
        assert!(GraphOptimizer::check_multiplication_identities(&nonzero, &non_constant).is_none());
    }

    #[test]
    fn test_check_division_identities_match_and_no_match() {
        let x = ComputationNode::leaf(|| 5.0);
        let one = ComputationNode::constant(1.0);
        let nonzero = ComputationNode::constant(3.0);

        // x / 1 = x
        assert!(GraphOptimizer::check_division_identities(&x, &one).is_some());
        assert!(GraphOptimizer::check_division_identities(&x, &nonzero).is_none());
    }

    #[test]
    fn test_is_constant_zero_and_one() {
        let zero = ComputationNode::constant(0.0);
        let one = ComputationNode::constant(1.0);
        let nonzero = ComputationNode::constant(3.0);

        assert!(GraphOptimizer::is_constant_zero(&zero));
        assert!(!GraphOptimizer::is_constant_zero(&one));
        assert!(GraphOptimizer::is_constant_one(&one));
        assert!(!GraphOptimizer::is_constant_one(&zero));
        assert!(!GraphOptimizer::is_constant_zero(&nonzero));
        assert!(!GraphOptimizer::is_constant_one(&nonzero));

        // Soundness: never decided by sampling -- a leaf that always samples 0.0/1.0
        // but wasn't built via `ComputationNode::constant` is not treated as constant.
        assert!(!GraphOptimizer::is_constant_zero(&ComputationNode::leaf(
            || 0.0
        )));
        assert!(!GraphOptimizer::is_constant_one(&ComputationNode::leaf(
            || 1.0
        )));
    }

    #[test]
    fn test_constant_folding_never_fires_on_sampled_constant() {
        // A leaf that always returns the same value by construction (not a genuine
        // distribution) but wasn't built via `ComputationNode::constant` must not be
        // folded -- constant_folding_binary_op only checks the structural flag.
        let const_looking = ComputationNode::leaf(|| 7.0);
        let other = ComputationNode::leaf(|| 7.0);
        let sum = ComputationNode::binary_op(const_looking, other, BinaryOperation::Add);
        let folded = GraphOptimizer::constant_folding(sum);
        assert!(matches!(folded, ComputationNode::BinaryOp { .. }));
    }

    #[test]
    fn test_constant_folding_bool_never_fires_on_sampled_constant() {
        let const_looking_condition = ComputationNode::leaf(|| true);
        let if_true = ComputationNode::leaf(|| 10.0);
        let if_false = ComputationNode::leaf(|| 20.0);
        let conditional = ComputationNode::conditional(const_looking_condition, if_true, if_false);
        let folded = GraphOptimizer::constant_folding(conditional);
        assert!(matches!(folded, ComputationNode::Conditional { .. }));
    }

    #[test]
    fn test_to_dot_node_ids_increment_sequentially() {
        // If add_node_to_dot's `*node_id += 1` regressed to `*= 1`, every node
        // would be assigned id 0 instead of sequential ids.
        let left = ComputationNode::leaf(|| 1.0);
        let right = ComputationNode::leaf(|| 2.0);
        let add = ComputationNode::binary_op(left, right, BinaryOperation::Add);

        let dot = GraphVisualizer::to_dot(&add);

        assert!(dot.contains("0 [label=\"Add\""));
        assert!(dot.contains("1 [label=\"Leaf\""));
        assert!(dot.contains("2 [label=\"Leaf\""));
        assert!(dot.contains("0 -> 1;"));
        assert!(dot.contains("0 -> 2;"));
    }

    #[test]
    fn test_profiler_get_stats_average_matches_total_over_count() {
        let mut profiler = GraphProfiler::new();
        profiler.profile_execution("op", || 1 + 1);
        profiler.profile_execution("op", || 2 + 2);
        profiler.profile_execution("op", || 3 + 3);

        let stats = profiler.get_stats("op").unwrap();
        assert_eq!(stats.count, 3);
        // Independently recompute the average/count relationship; catches
        // `average = total / count` regressing to `total * count`.
        assert_eq!(
            stats.average,
            stats.total / u32::try_from(stats.count).unwrap()
        );
    }
}
