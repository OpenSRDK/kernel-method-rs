use super::Kernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{error::Error, ops::Add, ops::Mul};

const PARAMS_LEN: usize = 0;

#[derive(Clone, Debug)]
pub struct Linear;

impl Kernel<Vec<f64>> for Linear {
    fn params_len(&self) -> usize {
        PARAMS_LEN
    }

    fn value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
        _: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
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

        let gfx = vec![];

        Ok((fx, gfx))
    }
}

impl<R> Add<R> for Linear
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for Linear
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}
