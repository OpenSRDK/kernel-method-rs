use super::PositiveDefiniteKernel;
use crate::{KernelAdd, KernelError, KernelMul};
use rayon::prelude::*;
use std::{f64::consts::PI, ops::Add, ops::Mul};

/// http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf
#[derive(Clone, Debug)]
pub struct SpectralMixture {
    p: usize,
    q: usize,
}

impl SpectralMixture {
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }
}

impl PositiveDefiniteKernel<Vec<f64>> for SpectralMixture {
    fn params_len(&self) -> usize {
        self.q + self.p * self.q + self.p * self.q
    }

    fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
        if params.len() != self.params_len() {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if self.p != x.len() {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != xprime.len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let w = &params[0..self.q];
        let v = &params[self.q..self.q + self.p * self.q];
        let mu = &params[self.q + self.p * self.q..self.q + self.p * self.q + self.p * self.q];

        let fx = (0..self.q)
            .into_par_iter()
            .map(|q| {
                w[q] * (0..self.p)
                    .into_par_iter()
                    .map(|p| {
                        (-2.0 * PI.powi(2) * (x[p] - xprime[p]).powi(2) * v[self.p * p + q]).exp()
                            * (2.0 * PI * (x[p] - xprime[p]) * mu[self.p * p + q]).cos()
                    })
                    .product::<f64>()
            })
            .sum();

        Ok(fx)
    }
}

impl<R> Add<R> for SpectralMixture
where
    R: PositiveDefiniteKernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for SpectralMixture
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
        let kernel = SpectralMixture::new(1, 2);

        let test_value = kernel.value(&[1.0, 1.0], &vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);

        match test_value {
            Err(KernelError::ParametersLengthMismatch) => (),
            _ => panic!(),
        };
    }
}
