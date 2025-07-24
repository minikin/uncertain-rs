use crate::operations::{Arithmetic, arithmetic::BinaryOperation};
use std::collections::HashMap;
use std::sync::Arc;

/// Context for memoizing samples within a single evaluation to ensure
/// shared variables produce the same sample value throughout an evaluation
pub struct SampleContext {
    /// Memoized values indexed by node ID
    memoized_values: HashMap<uuid::Uuid, Box<dyn std::any::Any + Send>>,
}

impl SampleContext {
    /// Create a new empty sample context
    #[must_use]
    pub fn new() -> Self {
        Self {
            memoized_values: HashMap::new(),
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
    T: Clone + Send + Sync + 'static,
{
    /// Evaluates the computation graph node with memoization context
    ///
    /// This is the core evaluation method that respects memoization to ensure
    /// shared variables produce consistent samples within a single evaluation.
    ///
    /// # Panics
    ///
    /// - Panics if called on a `BinaryOp` variant. Use `evaluate_arithmetic` instead for binary operations.
    /// - Panics if called on a `Conditional` variant. Use `evaluate_conditional` instead for conditional operations.
    pub fn evaluate(&self, context: &mut SampleContext) -> T {
        match self {
            ComputationNode::Leaf { id, sample } => {
                // Check if we already have a memoized value for this node
                if let Some(cached) = context.get_value::<T>(id) {
                    cached
                } else {
                    // Generate new sample and memoize it
                    let value = sample();
                    context.set_value(*id, value.clone());
                    value
                }
            }

            ComputationNode::UnaryOp { operand, operation } => {
                let operand_val = operand.evaluate(context);
                match operation {
                    UnaryOperation::Map(func) => func(operand_val),
                    UnaryOperation::Filter(_) => {
                        // Filter requires special handling with rejection sampling
                        // This is a simplified implementation
                        operand_val
                    }
                }
            }

            // These variants require special handling based on type constraints
            ComputationNode::BinaryOp { .. } => {
                panic!(
                    "BinaryOp evaluation requires arithmetic trait bounds. Use evaluate_arithmetic instead."
                )
            }

            ComputationNode::Conditional { .. } => {
                panic!(
                    "Conditional evaluation requires specific handling. Use evaluate_conditional instead."
                )
            }
        }
    }

    /// Evaluates arithmetic operations with proper trait bounds
    ///
    /// # Panics
    ///
    /// Panics if called on a `Conditional` variant with a boolean condition, as this is not supported in arithmetic context.
    pub fn evaluate_arithmetic(&self, context: &mut SampleContext) -> T
    where
        T: Arithmetic,
    {
        match self {
            ComputationNode::Leaf { id, sample } => {
                if let Some(cached) = context.get_value::<T>(id) {
                    cached
                } else {
                    let value = sample();
                    context.set_value(*id, value.clone());
                    value
                }
            }

            ComputationNode::BinaryOp {
                left,
                right,
                operation,
            } => {
                let left_val = left.evaluate_arithmetic(context);
                let right_val = right.evaluate_arithmetic(context);
                operation.apply(left_val, right_val)
            }

            ComputationNode::UnaryOp { operand, operation } => {
                let operand_val = operand.evaluate_arithmetic(context);
                match operation {
                    UnaryOperation::Map(func) => func(operand_val),
                    UnaryOperation::Filter(_) => operand_val,
                }
            }

            ComputationNode::Conditional {
                condition: _,
                if_true: _,
                if_false: _,
            } => {
                panic!(
                    "Conditional evaluation with bool condition not supported in arithmetic context"
                )
            }
        }
    }

    /// Evaluates the computation graph node in a new context
    ///
    /// This creates a fresh context for evaluation, useful when you want
    /// independent samples without memoization effects.
    #[must_use]
    pub fn evaluate_fresh(&self) -> T
    where
        T: Arithmetic,
    {
        let mut context = SampleContext::new();
        self.evaluate_conditional_with_arithmetic(&mut context)
    }

    /// Creates a new leaf node
    pub fn leaf<F>(sample: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        ComputationNode::Leaf {
            id: uuid::Uuid::new_v4(),
            sample: Arc::new(sample),
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
}

// Specialized implementation for handling conditionals with boolean conditions
impl ComputationNode<bool> {
    /// Evaluates boolean computation nodes
    ///
    /// # Panics
    ///
    /// Panics if called on a `BinaryOp` variant as boolean binary operations are not implemented.
    pub fn evaluate_bool(&self, context: &mut SampleContext) -> bool {
        match self {
            ComputationNode::Leaf { id, sample } => {
                if let Some(cached) = context.get_value::<bool>(id) {
                    cached
                } else {
                    let value = sample();
                    context.set_value(*id, value);
                    value
                }
            }
            ComputationNode::UnaryOp { operand, operation } => {
                let operand_val = operand.evaluate_bool(context);
                match operation {
                    UnaryOperation::Map(func) => func(operand_val),
                    UnaryOperation::Filter(_) => operand_val,
                }
            }
            ComputationNode::BinaryOp { .. } => {
                panic!("Boolean binary operations not implemented")
            }
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                let condition_val = condition.evaluate_bool(context);
                if condition_val {
                    if_true.evaluate_bool(context)
                } else {
                    if_false.evaluate_bool(context)
                }
            }
        }
    }
}

// Add a specialized method for evaluating conditionals with arithmetic return types
impl<T> ComputationNode<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Evaluates conditional nodes where condition is bool and branches return T
    pub fn evaluate_conditional_with_arithmetic(&self, context: &mut SampleContext) -> T
    where
        T: Arithmetic,
    {
        match self {
            ComputationNode::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                let condition_val = condition.evaluate_bool(context);
                if condition_val {
                    if_true.evaluate_arithmetic(context)
                } else {
                    if_false.evaluate_arithmetic(context)
                }
            }
            _ => self.evaluate_arithmetic(context),
        }
    }
}

/// Computation graph optimizer for improving evaluation performance
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Optimizes a computation graph by applying various transformations
    #[must_use]
    pub fn optimize<T>(node: ComputationNode<T>) -> ComputationNode<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        // Apply optimizations in order
        let node = Self::eliminate_identity_operations(node);
        Self::constant_folding(node)
    }

    /// Eliminates identity operations like `x + 0` or `x * 1`
    fn eliminate_identity_operations<T>(node: ComputationNode<T>) -> ComputationNode<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        // This is a simplified version - in practice, you'd need more sophisticated
        // pattern matching and constant detection
        node
    }

    /// Performs constant folding for compile-time evaluation of constant expressions
    fn constant_folding<T>(node: ComputationNode<T>) -> ComputationNode<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        // This is a simplified version - in practice, you'd detect constant
        // sub-expressions and pre-evaluate them
        node
    }
}

/// Computation graph visualizer for debugging and analysis
pub struct GraphVisualizer;

impl GraphVisualizer {
    /// Generates a DOT graph representation for visualization
    #[must_use]
    pub fn to_dot<T>(node: &ComputationNode<T>) -> String
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut dot = String::from("digraph G {\n");
        let mut node_id = 0;
        Self::add_node_to_dot(node, &mut dot, &mut node_id);
        dot.push_str("}\n");
        dot
    }

    fn add_node_to_dot<T>(node: &ComputationNode<T>, dot: &mut String, node_id: &mut usize) -> usize
    where
        T: Clone + Send + Sync + 'static,
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
        T: Clone + Send + Sync + 'static,
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
    ///
    /// # Panics
    ///
    /// Panics if the internal state is corrupted and the times vector is empty
    /// when it shouldn't be (this should never happen in normal usage).
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
        let min = *sorted_times
            .first()
            .expect("Times vector should not be empty");
        let max = *sorted_times
            .last()
            .expect("Times vector should not be empty");

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
        };

        // Evaluate twice with the same context
        let val1 = leaf.evaluate(&mut context);
        let val2 = leaf.evaluate(&mut context);

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
        let x = Uncertain::normal(5.0, 1.0);
        let y = Uncertain::normal(3.0, 1.0);

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
        // Allow for even wider variance due to the multiplication of uncertain values
        assert!(sample1 > -20.0 && sample1 < 100.0);
        assert!(sample2 > -20.0 && sample2 < 100.0);
    }
}
