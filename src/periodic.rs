use super::Kernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{ops::Add, ops::Mul};

const PARAMS_LEN: usize = 2;

#[derive(Clone, Debug)]
pub struct Periodic;

impl Periodic {
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

impl Kernel<Vec<f64>> for Periodic {
  fn params_len(&self) -> usize {
    PARAMS_LEN
  }

  fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
    let norm = self.norm(params, x, xprime)?;

    let fx = (params[0] * (norm / params[1]).cos()).exp();

    Ok(fx)
  }

  fn value_with_grad(
    &self,
    params: &[f64],
    x: &Vec<f64>,
    xprime: &Vec<f64>,
  ) -> Result<(f64, Vec<f64>), KernelError> {
    let norm = self.norm(params, x, xprime)?;

    let fx = (params[0] * (norm / params[1]).cos()).exp();

    let gfx = vec![
      (params[0] * (norm / params[1]).cos()).exp() * (norm / params[1]).cos(),
      (params[0] * (norm / params[1]).cos()).exp()
        * params[0]
        * (norm / params[1]).sin()
        * (norm / params[1].powi(2)),
    ];

    Ok((fx, gfx))
  }
}

impl<R> Add<R> for Periodic
where
  R: Kernel<Vec<f64>>,
{
  type Output = KernelAdd<Self, R, Vec<f64>>;

  fn add(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}

impl<R> Mul<R> for Periodic
where
  R: Kernel<Vec<f64>>,
{
  type Output = KernelMul<Self, R, Vec<f64>>;

  fn mul(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}
