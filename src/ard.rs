use super::Kernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{error::Error, ops::Add, ops::Mul};

fn weighted_norm_pow(params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> f64 {
    params
        .par_iter()
        .zip(x.par_iter())
        .zip(xprime.par_iter())
        .map(|((relevance, xi), xprimei)| relevance * (xi - xprimei).powi(2))
        .sum()
}
#[derive(Clone, Debug)]
pub struct ARD(pub usize);

impl Kernel<Vec<f64>> for ARD {
    fn params_len(&self) -> usize {
        self.0
    }

    fn value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
        with_grad: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        if params.len() != self.0 {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != self.0 || xprime.len() != self.0 {
            return Err(KernelError::InvalidArgument.into());
        }

        let fx = (-weighted_norm_pow(&params, x, xprime)).exp();

        let grad = if !with_grad {
            vec![]
        } else {
            let mut gfx = vec![f64::default(); self.0];

            gfx.par_iter_mut()
                .zip(x.par_iter())
                .zip(xprime.par_iter())
                .for_each(|((gfxi, &xi), &xprimei)| *gfxi = -(xi - xprimei).powi(2));

            gfx
        };

        Ok((fx, grad))
    }
}

impl<R> Add<R> for ARD
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for ARD
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}
