use crate::{KernelError, PositiveDefiniteKernel, Value};

pub trait ParamsDifferentiable<T>: PositiveDefiniteKernel<T>
where
    T: Value,
{
    fn ln_diff_params(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError>;
}
