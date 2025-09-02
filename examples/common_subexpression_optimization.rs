use uncertain_rs::{
    Uncertain, computation::GraphOptimizer, operations::arithmetic::BinaryOperation,
};

fn main() {
    println!("🔧 Common Subexpression Elimination Demo");
    println!("=======================================\n");

    // Create a complex expression with common subexpressions
    let x = Uncertain::normal(2.0, 0.1);
    let y = Uncertain::normal(3.0, 0.1);
    let z = Uncertain::normal(1.0, 0.1);

    // Expression: (x + y) * (x + y) + (x + y) * z
    // This has the common subexpression (x + y) repeated 3 times
    let sum = x.clone() + y.clone();
    let expr = (sum.clone() * sum.clone()) + (sum * z);

    println!("Original expression: (x + y) * (x + y) + (x + y) * z");
    println!("Expected result: (2 + 3) * (2 + 3) + (2 + 3) * 1 = 25 + 5 = 30\n");

    // Evaluate without optimization
    let result_without_opt = expr.expected_value(1000);
    println!("Result without optimization: {result_without_opt:.2}");

    // Now let's demonstrate the optimization
    println!("\n🔄 Applying Common Subexpression Elimination...");

    let mut optimizer = GraphOptimizer::new();

    // Create the same expression structure for optimization
    let x_node = uncertain_rs::computation::ComputationNode::leaf(|| 2.0);
    let y_node = uncertain_rs::computation::ComputationNode::leaf(|| 3.0);
    let z_node = uncertain_rs::computation::ComputationNode::leaf(|| 1.0);

    let sum_node = uncertain_rs::computation::ComputationNode::binary_op(
        x_node.clone(),
        y_node.clone(),
        BinaryOperation::Add,
    );

    let expr_node = uncertain_rs::computation::ComputationNode::binary_op(
        uncertain_rs::computation::ComputationNode::binary_op(
            sum_node.clone(),
            sum_node.clone(),
            BinaryOperation::Mul,
        ),
        uncertain_rs::computation::ComputationNode::binary_op(
            sum_node.clone(),
            z_node,
            BinaryOperation::Mul,
        ),
        BinaryOperation::Add,
    );

    // Apply optimization
    let optimized_node = optimizer.eliminate_common_subexpressions(expr_node);

    println!(
        "Cache size after optimization: {}",
        optimizer.subexpression_cache.len()
    );
    println!("Optimized node count: {}", optimized_node.node_count());

    // Evaluate optimized expression
    let result_with_opt = optimized_node.evaluate_fresh();
    println!("Result with optimization: {result_with_opt:.2}");

    println!("\n✅ Optimization successful!");
    println!("The common subexpression (x + y) was cached and reused.");
    println!("Both results should be approximately 30.0");
}
