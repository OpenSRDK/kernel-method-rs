use crate::{KernelError, PositiveDefiniteKernel, Value};

pub trait ParamsDifferentiableKernel<T>: PositiveDefiniteKernel<T>
where
    T: Value,
{
    fn diff_params(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError>;
}
