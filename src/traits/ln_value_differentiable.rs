use crate::{KernelError, PositiveDefiniteKernel, Value};

pub trait LogValueDifferentiableKernel<T>: PositiveDefiniteKernel<T>
where
    T: Value,
{
    fn ln_diff_value(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError>;
}
