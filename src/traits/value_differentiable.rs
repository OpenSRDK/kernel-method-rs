use crate::{KernelError, PositiveDefiniteKernel, Value};

pub trait ValueDifferentiableKernel<T>: PositiveDefiniteKernel<T>
where
    T: Value,
{
    fn diff_value(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError>;
}
