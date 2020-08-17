use crate::Kernel;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct ARD {
    data_len: usize,
    params: Vec<f64>,
}

impl ARD {
    pub fn new(data_len: usize, params: Vec<f64>) -> Self {
        Self { data_len, params }
    }

    fn weighted_norm(x: &Vec<f64>, x_prime: &Vec<f64>, params: &[f64]) -> f64 {
        let relevances = &params[3..];

        x.par_iter()
            .zip(x_prime.par_iter())
            .zip(relevances.par_iter())
            .map(|((x_i, x_prime_i), relevance)| relevance * (x_i - x_prime_i).powi(2))
            .sum()
    }

    fn prod(x: &Vec<f64>, x_prime: &Vec<f64>) -> f64 {
        x.par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| x_i * x_prime_i)
            .sum()
    }
}

impl Kernel<Vec<f64>> for ARD {
    fn get_params(&self) -> &[f64] {
        &self.params
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), String> {
        let n = params.len();
        if n != 3 + self.data_len {
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

        Ok(
            self.params[0] * (-ARD::weighted_norm(x, x_prime, &self.params)).exp()
                + self.params[1]
                + self.params[2] * ARD::prod(x, x_prime),
        )
    }

    fn grad(
        &self,
        x: &Vec<f64>,
        x_prime: &Vec<f64>,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, String>>, String> {
        if x.len() != x_prime.len() {
            return Err("dimension mismatch".to_owned());
        }

        let params_len = self.params.len();
        let x = x.to_vec();
        let x_prime = x_prime.to_vec();

        Ok(Box::new(move |params: &[f64]| {
            if params.len() != params_len {
                return Err("dimension mismatch".to_owned());
            }
            let mut grad = vec![f64::default(); params_len];

            grad[0] = (-ARD::weighted_norm(&x, &x_prime, params)).exp();
            grad[1] = 1.0;
            grad[2] = ARD::prod(&x, &x_prime);

            let relevances_grad = &mut grad[3..];

            relevances_grad
                .par_iter_mut()
                .zip(x.par_iter())
                .zip(x_prime.par_iter())
                .for_each(|((s, &l), &r)| *s = -(l - r).powi(2));

            Ok(grad)
        }))
    }
}
