use super::Kernel;
use crate::KernelError;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Debug;

const PARAMS_LEN: usize = 1;

#[derive(Clone, Debug)]
pub struct Exponential {
    params: [f64; PARAMS_LEN],
}

impl Exponential {
    pub fn new(params: [f64; PARAMS_LEN]) -> Self {
        Self { params }
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self::new([1.0])
    }
}

impl Kernel<Vec<f64>> for Exponential {
    fn get_params(&self) -> &[f64] {
        &self.params
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), Box<dyn Error>> {
        if params.len() != PARAMS_LEN {
            return Err(Box::new(KernelError::DimensionMismatch));
        }
        self.params
            .par_iter_mut()
            .zip(params.par_iter())
            .for_each(|(s, &p)| *s = p);

        Ok(())
    }

    fn value(&self, x: &Vec<f64>, x_prime: &Vec<f64>) -> Result<f64, Box<dyn Error>> {
        if x.len() != x_prime.len() {
            return Err(Box::new(KernelError::DimensionMismatch));
        }

        let norm: f64 = x
            .par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok((-norm / self.params[0]).exp())
    }

    fn grad(
        &self,
        x: &Vec<f64>,
        x_prime: &Vec<f64>,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, Box<dyn Error>>>, Box<dyn Error>> {
        if x.len() != x_prime.len() {
            return Err(Box::new(KernelError::DimensionMismatch));
        }

        let norm: f64 = x
            .par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(Box::new(move |params: &[f64]| {
            if params.len() != PARAMS_LEN {
                return Err(Box::new(KernelError::DimensionMismatch));
            }
            let mut grad = vec![f64::default(); PARAMS_LEN];

            grad[0] = (-norm / params[0]).exp() / params[0].powi(2);

            Ok(grad)
        }))
    }
}
