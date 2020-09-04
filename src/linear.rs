use super::Kernel;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Debug;

#[derive(Clone, Debug, Default)]
pub struct Linear;

impl Kernel<Vec<f64>> for Linear {
    fn get_params(&self) -> &[f64] {
        &[]
    }

    fn set_params(&mut self, _: &[f64]) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn value(&self, x: &Vec<f64>, x_prime: &Vec<f64>) -> Result<f64, Box<dyn Error>> {
        Ok(x.par_iter()
            .zip(x_prime.par_iter())
            .map(|(x_i, x_prime_i)| x_i * x_prime_i)
            .sum())
    }

    fn grad(
        &self,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, Box<dyn Error>>>, Box<dyn Error>> {
        Ok(Box::new(|_| Ok(vec![])))
    }
}
