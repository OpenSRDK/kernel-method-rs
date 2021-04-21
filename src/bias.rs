use super::Kernel;
use crate::Value;
use crate::{KernelAdd, KernelError, KernelMul};
use std::fmt::Debug;
use std::{ops::Add, ops::Mul};

const PARAMS_LEN: usize = 1;

#[derive(Clone, Debug)]
pub struct Bias;

impl<T> Kernel<T> for Bias
where
  T: Value,
{
  fn params_len(&self) -> usize {
    PARAMS_LEN
  }

  fn value(&self, params: &[f64], _: &T, _: &T) -> Result<f64, KernelError> {
    if params.len() != PARAMS_LEN {
      return Err(KernelError::ParametersLengthMismatch.into());
    }

    let fx = params[0];

    Ok(fx)
  }

  fn value_with_grad(
    &self,
    params: &[f64],
    x: &T,
    xprime: &T,
  ) -> Result<(f64, Vec<f64>), KernelError> {
    let fx = self.value(params, x, xprime)?;
    let gfx = vec![1.0];

    Ok((fx, gfx))
  }
}

impl<R> Add<R> for Bias
where
  R: Kernel<Vec<f64>>,
{
  type Output = KernelAdd<Self, R, Vec<f64>>;

  fn add(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}

impl<R> Mul<R> for Bias
where
  R: Kernel<Vec<f64>>,
{
  type Output = KernelMul<Self, R, Vec<f64>>;

  fn mul(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}
