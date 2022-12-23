use super::PositiveDefiniteKernel;
use crate::{
    KernelAdd, KernelError, KernelMul, LogParamsDifferentiableKernel, LogValueDifferentiableKernel,
};
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

impl LogValueDifferentiableKernel<Vec<f64>> for SpectralMixture {
    fn ln_diff_value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
    ) -> Result<Vec<f64>, KernelError> {
        let value = self.value(params, x, xprime).unwrap();
        let w = &params[0..self.q];
        let v = &params[self.q..self.q + self.p * self.q];
        let mu = &params[self.q + self.p * self.q..self.q + self.p * self.q + self.p * self.q];

        let diff = (0..self.p)
            .into_par_iter()
            .map(|p| {
                (0..self.q)
                    .into_par_iter()
                    .map(|q| {
                        let each_wd = w[q]
                            * (0..self.p)
                                .into_par_iter()
                                .map(|i| {
                                    (-2.0
                                        * PI.powi(2)
                                        * (x[i] - xprime[i]).powi(2)
                                        * v[self.p * i + q])
                                        .exp()
                                        * (2.0 * PI * (x[i] - xprime[i]) * mu[self.p * i + q]).cos()
                                })
                                .product::<f64>();
                        let diff_d = (4.0 * PI.powi(2) * (x[p] - xprime[p]) * v[self.p * p + q])
                            + (2.0 * PI * (x[p] - xprime[p]) * mu[self.p * p + q]).tan()
                                * ((-2.0) * PI * (x[p] - xprime[p]) * mu[self.p * p + q]);
                        diff_d * each_wd / value
                    })
                    .sum()
            })
            .collect::<Vec<f64>>();

        Ok(diff)
    }
}

impl LogParamsDifferentiableKernel<Vec<f64>> for SpectralMixture {
    fn ln_diff_params(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
    ) -> Result<Vec<f64>, KernelError> {
        let value = self.value(params, x, xprime).unwrap();
        let w = &params[0..self.q];
        let v = &params[self.q..self.q + self.p * self.q];
        let mu = &params[self.q + self.p * self.q..self.q + self.p * self.q + self.p * self.q];

        let diff_w = (0..self.q)
            .into_par_iter()
            .map(|q| {
                (0..self.p)
                    .into_par_iter()
                    .map(|i| {
                        (-2.0 * PI.powi(2) * (x[i] - xprime[i]).powi(2) * v[self.p * i + q]).exp()
                            * (2.0 * PI * (x[i] - xprime[i]) * mu[self.p * i + q]).cos()
                    })
                    .product::<f64>()
            })
            .collect::<Vec<f64>>();

        let diff_mu = (0..self.q)
            .into_par_iter()
            .map(|q| {
                let each_wd = w[q]
                    * (0..self.p)
                        .into_par_iter()
                        .map(|i| {
                            (-2.0 * PI.powi(2) * (x[i] - xprime[i]).powi(2) * v[self.p * i + q])
                                .exp()
                                * (2.0 * PI * (x[i] - xprime[i]) * mu[self.p * i + q]).cos()
                        })
                        .product::<f64>();
                (0..self.p)
                    .into_par_iter()
                    .map(|p| {
                        let diff_d = (2.0 * PI * (x[p] - xprime[p]) * mu[self.p * p + q]).tan()
                            * ((-2.0) * PI * (x[p] - xprime[p]));
                        diff_d * each_wd / value
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
            .concat();

        let diff_v = (0..self.q)
            .into_par_iter()
            .map(|q| {
                let each_wd = w[q]
                    * (0..self.p)
                        .into_par_iter()
                        .map(|i| {
                            (-2.0 * PI.powi(2) * (x[i] - xprime[i]).powi(2) * v[self.p * i + q])
                                .exp()
                                * (2.0 * PI * (x[i] - xprime[i]) * mu[self.p * i + q]).cos()
                        })
                        .product::<f64>();
                (0..self.p)
                    .into_par_iter()
                    .map(|p| {
                        let diff_d = 4.0 * PI.powi(2) * (x[p] - xprime[p]).powi(2);
                        diff_d * each_wd / value
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
            .concat();

        let diff = [diff_w, diff_v, diff_mu].concat();

        Ok(diff)
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
