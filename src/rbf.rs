use super::Kernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{ops::Add, ops::Mul};

const PARAMS_LEN: usize = 2;

#[derive(Clone, Debug)]
pub struct RBF;

impl RBF {
  fn norm_pow(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
    if params.len() != PARAMS_LEN {
      return Err(KernelError::ParametersLengthMismatch.into());
    }
    if x.len() != xprime.len() {
      return Err(KernelError::InvalidArgument.into());
    }

    let norm_pow = x
      .par_iter()
      .zip(xprime.par_iter())
      .map(|(x_i, xprime_i)| (x_i - xprime_i).powi(2))
      .sum();

    Ok(norm_pow)
  }
}

impl Kernel<Vec<f64>> for RBF {
  fn params_len(&self) -> usize {
    PARAMS_LEN
  }

  fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
    let norm_pow = self.norm_pow(params, x, xprime)?;

    let fx = params[0] * (-norm_pow / params[1]).exp();

    Ok(fx)
  }

  fn value_with_grad(
    &self,
    params: &[f64],
    x: &Vec<f64>,
    xprime: &Vec<f64>,
  ) -> Result<(f64, Vec<f64>), KernelError> {
    let norm_pow = self.norm_pow(params, x, xprime)?;

    let fx = params[0] * (-norm_pow / params[1]).exp();

    let gfx = vec![
      (-norm_pow / params[1]).exp(),
      params[0] * (-norm_pow / params[1]).exp() * (norm_pow / params[1].powi(2)),
    ];

    Ok((fx, gfx))
  }
}

impl<R> Add<R> for RBF
where
  R: Kernel<Vec<f64>>,
{
  type Output = KernelAdd<Self, R, Vec<f64>>;

  fn add(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}

impl<R> Mul<R> for RBF
where
  R: Kernel<Vec<f64>>,
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
    let kernel = RBF;

    let (func, grad) = kernel
      .value_with_grad(&[1.0, 1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
      .unwrap();

    println!("{}", func);
    println!("{:#?}", grad);
  }
}
