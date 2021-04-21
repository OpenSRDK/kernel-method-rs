use crate::Value;
use crate::{Kernel, KernelError};
use rayon::prelude::*;
use std::fmt::Debug;

pub trait Convolutable: Value {
  fn parts_len(&self) -> usize;
  fn part(&self, index: usize) -> &Vec<f64>;
  fn data_len(&self) -> usize;
}

impl Convolutable for Vec<f64> {
  fn parts_len(&self) -> usize {
    1
  }

  fn part(&self, _: usize) -> &Vec<f64> {
    self
  }

  fn data_len(&self) -> usize {
    self.len()
  }
}

#[derive(Clone, Debug)]
pub struct Convolutional<K>
where
  K: Kernel<Vec<f64>>,
{
  kernel: K,
}

impl<K> Convolutional<K>
where
  K: Kernel<Vec<f64>>,
{
  pub fn new(kernel: K) -> Self {
    Self { kernel }
  }

  pub fn kernel_ref(&self) -> &K {
    &self.kernel
  }
}

impl<T, K> Kernel<T> for Convolutional<K>
where
  T: Convolutable,
  K: Kernel<Vec<f64>>,
{
  fn params_len(&self) -> usize {
    self.kernel.params_len()
  }

  fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError> {
    if params.len() != self.kernel.params_len() {
      return Err(KernelError::ParametersLengthMismatch.into());
    }
    let p = x.parts_len();
    if p != xprime.parts_len() {
      return Err(KernelError::InvalidArgument.into());
    }

    let fx = (0..p)
      .into_par_iter()
      .map(|pi| self.kernel.value(params, x.part(pi), xprime.part(pi)))
      .sum::<Result<f64, KernelError>>()?;

    Ok(fx)
  }

  fn value_with_grad(
    &self,
    params: &[f64],
    x: &T,
    xprime: &T,
  ) -> Result<(f64, Vec<f64>), KernelError> {
    if params.len() != self.kernel.params_len() {
      return Err(KernelError::ParametersLengthMismatch.into());
    }
    let p = x.parts_len();
    if p != xprime.parts_len() {
      return Err(KernelError::InvalidArgument.into());
    }

    let (fx, gfx): (f64, Vec<f64>) = (0..p)
      .into_iter()
      .map(|pi| {
        self
          .kernel
          .value_with_grad(params, x.part(pi), xprime.part(pi))
      })
      .try_fold::<(f64, Vec<f64>), _, Result<(f64, Vec<f64>), KernelError>>(
        (0.0, vec![]),
        |a, b| {
          let b = b?;
          Ok((a.0 + b.0, [a.1, b.1].concat()))
        },
      )?;

    Ok((fx, gfx))
  }
}
