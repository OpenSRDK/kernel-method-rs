extern crate rayon;
extern crate thiserror;

pub use ard::ard;
pub use bias::bias;
pub use exponential::exponential;
pub use linear::linear;
pub use periodic::periodic;
pub use rbf::rbf;
use std::error::Error;
use std::fmt::Debug;

pub mod ard;
pub mod bias;
pub mod exponential;
pub mod linear;
pub mod ops;
pub mod periodic;
pub mod rbf;

pub type Func<T> =
    Box<dyn Fn(&T, &T, bool, &[f64]) -> Result<(f64, Option<Vec<f64>>), Box<dyn Error>>>;
pub struct Kernel<T>
where
    T: ?Sized,
{
    params: Vec<f64>,
    func: Func<T>,
}

impl<T> Debug for Kernel<T>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.params)
    }
}

unsafe impl<T> Send for Kernel<T> where T: ?Sized {}
unsafe impl<T> Sync for Kernel<T> where T: ?Sized {}

impl<T> Kernel<T>
where
    T: ?Sized,
{
    pub fn new(params: Vec<f64>, func: Func<T>) -> Self {
        Self { params, func }
    }

    pub fn params(&self) -> &[f64] {
        &self.params
    }

    pub fn with_params(mut self, params: &[f64]) -> Result<Self, Box<dyn Error>> {
        if self.params.len() != params.len() {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        self.params.clone_from_slice(params);

        Ok(self)
    }

    pub fn func(
        &self,
        x: &T,
        x_prime: &T,
        with_grad: bool,
        rewrite_params: Option<&[f64]>,
    ) -> Result<(f64, Option<Vec<f64>>), Box<dyn Error>> {
        (self.func)(
            x,
            x_prime,
            with_grad,
            match rewrite_params {
                None => &self.params,
                Some(v) => {
                    if self.params.len() != v.len() {
                        return Err(KernelError::ParametersLengthMismatch.into());
                    }
                    v
                }
            },
        )
    }
}

#[derive(thiserror::Error, Debug)]
pub enum KernelError {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("invalid parameter")]
    InvalidParameter,
    #[error("parameters length mismatch")]
    ParametersLengthMismatch,
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = bias() + bias() * linear() + bias() * rbf() + bias() * periodic();
        let (func, grad) = kernel
            .func(&vec![1.0, 2.0, 3.0], &vec![10.0, 20.0, 30.0], true, None)
            .unwrap();

        println!("{}", func);
        println!("{:#?}", grad);
    }
}
