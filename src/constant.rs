use super::PositiveDefiniteKernel;
use crate::Value;
use crate::{KernelAdd, KernelError, KernelMul};
use std::fmt::Debug;
use std::{ops::Add, ops::Mul};

const PARAMS_LEN: usize = 1;

#[derive(Clone, Debug)]
pub struct Constant;

impl<T> PositiveDefiniteKernel<T> for Constant
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
}

impl<R> Add<R> for Constant
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for Constant
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
        let kernel = Constant;

        let test_value = kernel
            .value(&[1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
            .unwrap();

        assert_eq!(test_value, 1.0);
    }
}
