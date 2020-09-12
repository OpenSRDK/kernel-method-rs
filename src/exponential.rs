use super::Kernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{error::Error, ops::Add, ops::Mul};

const PARAMS_LEN: usize = 1;

fn norm(x: &Vec<f64>, xprime: &Vec<f64>) -> f64 {
    x.par_iter()
        .zip(xprime.par_iter())
        .map(|(x_i, xprime_i)| (x_i - xprime_i).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[derive(Clone, Debug)]
pub struct Exponential;

impl Kernel<Vec<f64>> for Exponential {
    fn params_len(&self) -> usize {
        PARAMS_LEN
    }

    fn value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
        with_grad: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        if params.len() != PARAMS_LEN {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != xprime.len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let norm = norm(x, xprime);

        let fx = (-norm / params[0]).exp();

        let gfx = if !with_grad {
            vec![]
        } else {
            let mut gfx = vec![f64::default(); PARAMS_LEN];

            gfx[0] = (-norm / params[0]).exp() / params[0].powi(2);

            gfx
        };

        Ok((fx, gfx))
    }
}

impl<R> Add<R> for Exponential
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for Exponential
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}
