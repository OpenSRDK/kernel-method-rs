use crate::KernelError;
use crate::Value;
use crate::{Kernel, KernelAdd};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ops::Add;
use std::{fmt::Debug, ops::Mul};

#[derive(Clone, Debug)]
pub struct KernelMul<L, R, T>
where
  L: Kernel<T>,
  R: Kernel<T>,
  T: Value,
{
  lhs: L,
  rhs: R,
  phantom: PhantomData<T>,
}

impl<L, R, T> KernelMul<L, R, T>
where
  L: Kernel<T>,
  R: Kernel<T>,
  T: Value,
{
  pub fn new(lhs: L, rhs: R) -> Self {
    Self {
      lhs,
      rhs,
      phantom: PhantomData,
    }
  }
}

impl<L, R, T> Kernel<T> for KernelMul<L, R, T>
where
  L: Kernel<T>,
  R: Kernel<T>,
  T: Value,
{
  fn params_len(&self) -> usize {
    self.lhs.params_len() + self.rhs.params_len()
  }

  fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError> {
    let lhs_params_len = self.lhs.params_len();
    let fx = self.lhs.value(&params[..lhs_params_len], x, xprime)?;
    let gx = self.rhs.value(&params[lhs_params_len..], x, xprime)?;

    let hx = fx * gx;

    Ok(hx)
  }

  fn value_with_grad(
    &self,
    params: &[f64],
    x: &T,
    xprime: &T,
  ) -> Result<(f64, Vec<f64>), KernelError> {
    let lhs_params_len = self.lhs.params_len();
    let (fx, dfx) = self
      .lhs
      .value_with_grad(&params[..lhs_params_len], x, xprime)?;
    let (gx, dgx) = self
      .rhs
      .value_with_grad(&params[lhs_params_len..], x, xprime)?;

    let hx = fx * gx;

    let ghx = dfx
      .par_iter()
      .map(|dfxi| dfxi * gx)
      .chain(dgx.par_iter().map(|dgxi| fx * dgxi))
      .collect::<Vec<_>>();

    Ok((hx, ghx))
  }
}

impl<Rhs, L, R, T> Add<Rhs> for KernelMul<L, R, T>
where
  Rhs: Kernel<T>,
  L: Kernel<T>,
  R: Kernel<T>,
  T: Value,
{
  type Output = KernelAdd<Self, Rhs, T>;

  fn add(self, rhs: Rhs) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}

impl<Rhs, L, R, T> Mul<Rhs> for KernelMul<L, R, T>
where
  Rhs: Kernel<T>,
  L: Kernel<T>,
  R: Kernel<T>,
  T: Value,
{
  type Output = KernelMul<Self, Rhs, T>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}
