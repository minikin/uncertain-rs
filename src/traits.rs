/// A trait alias for types that can be safely shared across threads and have stable references.
/// This combines Clone + Send + Sync + 'static which are required for concurrent uncertain computations.
pub trait Shareable: Clone + Send + Sync + 'static {}

// Blanket implementation for all types that satisfy the bounds
impl<T> Shareable for T where T: Clone + Send + Sync + 'static {}
