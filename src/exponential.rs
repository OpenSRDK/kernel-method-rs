use super::PositiveDefiniteKernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{ops::Add, ops::Mul};

const PARAMS_LEN: usize = 1;

#[derive(Clone, Debug)]
pub struct Exponential;

impl Exponential {
    fn norm(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
        if params.len() != PARAMS_LEN {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != xprime.len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let v = x
            .par_iter()
            .zip(xprime.par_iter())
            .map(|(x_i, xprime_i)| (x_i - xprime_i).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(v)
    }
}

impl PositiveDefiniteKernel<Vec<f64>> for Exponential {
    fn params_len(&self) -> usize {
        PARAMS_LEN
    }

    fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
        let norm = self.norm(params, x, xprime)?;

        let fx = (-norm / params[0]).exp();

        Ok(fx)
    }
}

impl<R> Add<R> for Exponential
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for Exponential
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = Exponential;

        let test_value = kernel
            .value(&[1.0], &vec![1.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0])
            .unwrap();

        assert_eq!(test_value, (-1f64).exp());
    }
}
