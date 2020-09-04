extern crate rayon;
extern crate thiserror;

pub use crate::{ard::ARD, exponential::Exponential, linear::Linear, periodic::Periodic, rbf::RBF};
use std::error::Error;
use std::fmt::Debug;

pub mod ard;
pub mod exponential;
pub mod linear;
pub mod periodic;
pub mod rbf;

pub trait Kernel<T>: Clone + Debug + Send + Sync {
    fn get_params(&self) -> &[f64];
    fn set_params(&mut self, params: &[f64]) -> Result<(), Box<dyn Error>>;
    fn value(&self, x: &T, x_prime: &T) -> Result<f64, Box<dyn Error>>;
    fn grad(
        &self,
        x: &T,
        x_prime: &T,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, Box<dyn Error>>>, Box<dyn Error>>;
}

#[derive(thiserror::Error, Debug)]
pub enum KernelError {
    #[error("dimension mismatch")]
    DimensionMismatch,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
