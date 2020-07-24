use super::Kernel;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct RBF {
    params: [f64; 2],
}

impl RBF {
    pub fn new(params: [f64; 2]) -> Self {
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
        let n = params.len();
        if n != 2 {
            return Err("dimension mismatch".to_owned());
        }
        for i in 0..n {
            self.params[i] = params[i];
        }

        Ok(())
    }

    fn value(&self, x: &Vec<f64>, x_prime: &Vec<f64>) -> Result<f64, String> {
        if x.len() != x_prime.len() {
            return Err("dimension mismatch".to_owned());
        }

        let norm: f64 = x
            .par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
            .sum();

        Ok(self.params[0] * (-1.0 * norm / self.params[1]).exp())
    }

    fn grad(
        &self,
        x: &Vec<f64>,
        x_prime: &Vec<f64>,
    ) -> Result<Box<dyn Fn(&[f64]) -> Vec<f64>>, String> {
        if x.len() != x_prime.len() {
            return Err("dimension mismatch".to_owned());
        }

        let norm: f64 = x
            .par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
            .sum();

        Ok(Box::new(move |params| {
            let mut grad = vec![f64::default(); 2];

            grad[0] = (-1.0 * norm / params[1]).exp();
            grad[1] = params[0] * norm * (-norm / params[1]).exp() / params[1].powi(2);

            grad
        }))
    }
}
