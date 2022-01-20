use crate::{KernelError, PositiveDefiniteKernel, Value};

pub trait ValueDifferentiable: PositiveDefiniteKernel<Vec<f64>>
{
    fn ln_diff_value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<(Vec<f64>, f64), KernelError>;
}
