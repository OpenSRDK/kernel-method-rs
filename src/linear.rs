use super::PositiveDefiniteKernel;
use crate::{KernelAdd, KernelError, KernelMul, ValueDifferentiableKernel, ParamsDifferentiableKernel};
use rayon::prelude::*;
use std::{ops::Add, ops::Mul};
use opensrdk_linear_algebra::*;


const PARAMS_LEN: usize = 0;

#[derive(Clone, Debug)]
pub struct Linear;

impl PositiveDefiniteKernel<Vec<f64>> for Linear {
    fn params_len(&self) -> usize {
        PARAMS_LEN
    }

    fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
        if params.len() != PARAMS_LEN {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != xprime.len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let fx = x
            .par_iter()
            .zip(xprime.par_iter())
            .map(|(x_i, xprime_i)| x_i * xprime_i)
            .sum();

        Ok(fx)
    }
}

impl<R> Add<R> for Linear
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for Linear
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl ValueDifferentiableKernel<Vec<f64>> for Linear {
    fn ln_diff_value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
    ) -> Result<(Vec<f64>, f64), KernelError> {
        let value = &self.value(params, x, xprime).unwrap();
        let diff = (2.0 / value * x.clone().col_mat()).vec();
        Ok((diff, *value))
    }
}

impl ParamsDifferentiableKernel<Vec<f64>> for Linear {
  fn ln_diff_params(
      &self,
      params: &[f64],
      x: &Vec<f64>,
      xprime: &Vec<f64>,
  ) -> Result<(Vec<f64>, f64), KernelError> {
      let diff = vec![];
      let value = &self.value(params, x, xprime).unwrap();
      Ok((diff, *value))
  }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = Linear;

        let test_value = kernel
            .value(&[], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
            .unwrap();

        assert_eq!(test_value, 10.0);
    }
}
