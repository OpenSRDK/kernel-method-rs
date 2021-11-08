extern crate rayon;
extern crate thiserror;

pub use add::*;
pub use ard::*;
pub use bias::*;
pub use convolutional::*;
pub use exponential::*;
pub use instant::*;
pub use linear::*;
pub use mul::*;
pub use neural_network::{deep_neural_network::*, relu::*};
pub use periodic::*;
pub use rbf::*;
pub use spectral_mixture::*;

use std::fmt::Debug;

pub mod add;
pub mod ard;
pub mod bias;
pub mod convolutional;
pub mod exponential;
pub mod instant;
pub mod linear;
pub mod mul;
pub mod neural_network;
pub mod periodic;
pub mod rbf;
pub mod spectral_mixture;

pub trait Value: Clone + Debug + Send + Sync {}
impl<T> Value for T where T: Clone + Debug + Send + Sync {}

pub trait Kernel<T>: Clone + Debug + Send + Sync
where
    T: Value,
{
    fn params_len(&self) -> usize;

    fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError>;

    fn value_with_grad(
        &self,
        params: &[f64],
        x: &T,
        xprime: &T,
    ) -> Result<(f64, Vec<f64>), KernelError>;
}

#[derive(thiserror::Error, Debug)]
pub enum KernelError {
    #[error("parameters length mismatch")]
    ParametersLengthMismatch,
    #[error("invalid parameter")]
    InvalidParameter,
    #[error("invalid argument")]
    InvalidArgument,
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = RBF + Bias * Linear + Bias * Periodic + Bias * ARD(3);
        let (func, grad) = kernel
            .value_with_grad(
                &vec![1.0; kernel.params_len()],
                &vec![1.0, 2.0, 3.0],
                &vec![10.0, 20.0, 30.0],
            )
            .unwrap();

        println!("{}", func);
        println!("{:#?}", grad);
    }
}
