use super::Kernel;
use rayon::prelude::*;
use std::fmt::Debug;

const PARAMS_LEN: usize = 2;

#[derive(Clone, Debug)]
pub struct RBF {
    params: [f64; PARAMS_LEN],
}

impl RBF {
    pub fn new(params: [f64; PARAMS_LEN]) -> Self {
        Self { params }
    }
}

impl Default for RBF {
    fn default() -> Self {
        Self::new([1.0, 1.0])
    }
}

impl Kernel<Vec<f64>> for RBF {
    fn get_params(&self) -> &[f64] {
        &self.params
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), String> {
        if params.len() != PARAMS_LEN {
            return Err("dimension mismatch".to_owned());
        }
        self.params
            .par_iter_mut()
            .zip(params.par_iter())
            .for_each(|(s, &p)| *s = p);

        Ok(())
    }

    fn value(&self, x: &Vec<f64>, x_prime: &Vec<f64>) -> Result<f64, String> {
        if x.len() != x_prime.len() {
            return Err("dimension mismatch".to_owned());
        }

        let norm_pow: f64 = x
            .par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
            .sum();

        Ok(self.params[0] * (-norm_pow / self.params[1]).exp())
    }

    fn grad(
        &self,
        x: &Vec<f64>,
        x_prime: &Vec<f64>,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, String>>, String> {
        if x.len() != x_prime.len() {
            return Err("dimension mismatch".to_owned());
        }

        let norm_pow: f64 = x
            .par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
            .sum();

        Ok(Box::new(move |params: &[f64]| {
            if params.len() != PARAMS_LEN {
                return Err("dimension mismatch".to_owned());
            }
            let mut grad = vec![f64::default(); PARAMS_LEN];

            grad[0] = (-norm_pow / params[1]).exp();
            grad[1] = params[0] * (-norm_pow / params[1]).exp() * (norm_pow / params[1].powi(2));

            Ok(grad)
        }))
    }
}
