extern crate rayon;

pub mod ard;
pub mod linear;
pub mod rbf;

use std::fmt::Debug;

pub trait Kernel<T>: Clone + Debug + Send + Sync {
    fn get_params(&self) -> &[f64];
    fn set_params(&mut self, params: &[f64]) -> Result<(), String>;
    fn value(&self, x: &T, x_prime: &T) -> Result<f64, String>;
    fn grad(&self, x: &T, x_prime: &T) -> Result<Box<dyn Fn(&[f64]) -> Vec<f64>>, String>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
