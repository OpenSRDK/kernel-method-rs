use super::PositiveDefiniteKernel;
use crate::{
    KernelAdd, KernelError, KernelMul, ParamsDifferentiableKernel, ValueDifferentiableKernel,
};
use rayon::prelude::*;
use std::{ops::Add, ops::Mul};

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

impl PositiveDefiniteKernel<Vec<f64>> for ARD {
    fn params_len(&self) -> usize {
        self.0
    }

    fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
        if params.len() != self.0 {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != self.0 || xprime.len() != self.0 {
            return Err(KernelError::InvalidArgument.into());
        }

        let fx = (-weighted_norm_pow(&params, x, xprime)).exp();

        Ok(fx)
    }
}

impl<R> Add<R> for ARD
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for ARD
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl ValueDifferentiableKernel<Vec<f64>> for ARD {
    fn ln_diff_value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
    ) -> Result<Vec<f64>, KernelError> {
        let diff = params
            .par_iter()
            .zip(x.par_iter())
            .zip(xprime.par_iter())
            .map(|((relevance, xi), xprimei)| -2.0 * relevance * (xi - xprimei))
            .collect::<Vec<f64>>();
        Ok(diff)
    }
}

impl ParamsDifferentiableKernel<Vec<f64>> for ARD {
    fn ln_diff_params(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
    ) -> Result<Vec<f64>, KernelError> {
        let diff = params
            .par_iter()
            .zip(x.par_iter())
            .zip(xprime.par_iter())
            .map(|((_relevance, xi), xprimei)| -(xi - xprimei).powi(2))
            .collect::<Vec<f64>>();
        Ok(diff)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = ARD(3);

        let test_value = kernel
            .value(&[1.0, 0.0, 0.0], &vec![1.0, 2.0, 3.0], &vec![0.0, 2.0, 1.0])
            .unwrap();

        assert_eq!(test_value, (-1f64).exp());
    }
}
