use uncertain_rs::{
    Uncertain, computation::ComputationNode, computation::GraphOptimizer,
    operations::arithmetic::BinaryOperation,
};

fn main() {
    println!("🔧 Graph Optimization Demo");
    println!("=========================\n");

    // Create a complex expression with multiple optimization opportunities
    let x = Uncertain::normal(2.0, 0.1);
    let y = Uncertain::normal(3.0, 0.1);
    let z = Uncertain::normal(1.0, 0.1);

    // Expression: (x + y) * (x + y) + (x + y) * z + (x + 0) * 1
    // This has:
    // 1. Common subexpression: (x + y) appears 3 times
    // 2. Identity operations: (x + 0) and (* 1)
    // 3. Constant folding opportunities: 0, 1
    let sum = x.clone() + y.clone();
    let expr = (sum.clone() * sum.clone())
        + (sum * z)
        + ((x + Uncertain::point(0.0)) * Uncertain::point(1.0));

    println!("Original expression: (x + y) * (x + y) + (x + y) * z + (x + 0) * 1");
    println!("Expected result: (2 + 3) * (2 + 3) + (2 + 3) * 1 + (2 + 0) * 1 = 25 + 5 + 2 = 32\n");

    let result_without_opt = expr.expected_value(1000);
    println!("Result without optimization: {result_without_opt:.2}");

    println!("\n🔄 Applying Graph Optimizations...");

    let mut optimizer = GraphOptimizer::new();

    // Create the same expression structure for optimization
    let x_node = ComputationNode::leaf(|| 2.0);
    let y_node = ComputationNode::leaf(|| 3.0);
    let z_node = ComputationNode::leaf(|| 1.0);
    let zero_node = ComputationNode::leaf(|| 0.0);
    let one_node = ComputationNode::leaf(|| 1.0);

    let sum_node = ComputationNode::binary_op(x_node.clone(), y_node.clone(), BinaryOperation::Add);

    // Create the complex expression
    let expr1 =
        ComputationNode::binary_op(sum_node.clone(), sum_node.clone(), BinaryOperation::Mul);

    let expr2 = ComputationNode::binary_op(sum_node.clone(), z_node, BinaryOperation::Mul);

    let expr3 = ComputationNode::binary_op(
        ComputationNode::binary_op(x_node.clone(), zero_node, BinaryOperation::Add),
        one_node,
        BinaryOperation::Mul,
    );

    let final_expr = ComputationNode::binary_op(
        ComputationNode::binary_op(expr1, expr2, BinaryOperation::Add),
        expr3,
        BinaryOperation::Add,
    );

    // Apply all optimizations
    let optimized_node = optimizer.optimize(final_expr);

    println!("Optimized node count: {}", optimized_node.node_count());

    let result_with_opt = optimized_node.evaluate_fresh();
    println!("Result with optimization: {result_with_opt:.2}");

    println!("\n✅ Optimization successful!");
    println!("The optimizations applied:");
    println!("  - Common subexpression elimination: (x + y) was cached and reused");
    println!("  - Identity operation elimination: (x + 0) → x, (* 1) → removed");
    println!("  - Constant folding: 0 and 1 were evaluated at compile time");
    println!("Both results should be approximately 32.0");
}
