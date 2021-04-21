use super::Kernel;
use crate::KernelError;
use crate::Value;
use crate::{KernelAdd, KernelMul};
use std::{fmt::Debug, ops::Add, ops::Mul};

#[derive(Clone)]
pub struct InstantKernel<'a, T>
where
  T: Value,
{
  params_len: usize,
  value: &'a (dyn Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Send + Sync),
  value_with_grad:
    &'a (dyn Fn(&[f64], &T, &T) -> Result<(f64, Vec<f64>), KernelError> + Send + Sync),
}

impl<'a, T> InstantKernel<'a, T>
where
  T: Value,
{
  pub fn new(
    params_len: usize,
    value: &'a (dyn Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Send + Sync),
    value_with_grad: &'a (dyn Fn(&[f64], &T, &T) -> Result<(f64, Vec<f64>), KernelError>
           + Send
           + Sync),
  ) -> Self {
    Self {
      params_len,
      value,
      value_with_grad,
    }
  }
}

impl<'a, T> Debug for InstantKernel<'a, T>
where
  T: Value,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "InstantKernel {{ params_len: {} }}", self.params_len)
  }
}

impl<'a, T> Kernel<T> for InstantKernel<'a, T>
where
  T: Value,
{
  fn params_len(&self) -> usize {
    self.params_len
  }

  fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError> {
    (self.value)(params, x, xprime)
  }

  fn value_with_grad(
    &self,
    params: &[f64],
    x: &T,
    xprime: &T,
  ) -> Result<(f64, Vec<f64>), KernelError> {
    (self.value_with_grad)(params, x, xprime)
  }
}

impl<'a, T, R> Add<R> for InstantKernel<'a, T>
where
  T: Value,
  R: Kernel<T>,
{
  type Output = KernelAdd<Self, R, T>;

  fn add(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}

impl<'a, T, R> Mul<R> for InstantKernel<'a, T>
where
  T: Value,
  R: Kernel<T>,
{
  type Output = KernelMul<Self, R, T>;

  fn mul(self, rhs: R) -> Self::Output {
    Self::Output::new(self, rhs)
  }
}

#[cfg(test)]
mod tests {
  use crate::*;
  #[test]
  fn it_works() {
    let kernel = RBF + InstantKernel::new(0, &|_, _, _| Ok(0.0), &|_, _, _| Ok((0.0, vec![])));

    let (func, grad) = kernel
      .value_with_grad(&[1.0, 1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
      .unwrap();

    println!("{}", func);
    println!("{:#?}", grad);
  }
}
