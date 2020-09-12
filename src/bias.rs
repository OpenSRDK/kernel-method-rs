use super::Kernel;
use crate::{KernelAdd, KernelError, KernelMul};
use std::fmt::Debug;
use std::{error::Error, ops::Add, ops::Mul};

const PARAMS_LEN: usize = 1;

#[derive(Clone, Debug)]
pub struct Bias;

impl<T> Kernel<T> for Bias
where
    T: Clone + Debug,
{
    fn params_len(&self) -> usize {
        PARAMS_LEN
    }

    fn value(
        &self,
        params: &[f64],
        _: &T,
        _: &T,
        _: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        if params.len() != PARAMS_LEN {
            return Err(KernelError::ParametersLengthMismatch.into());
        }

        let fx = params[0];
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
