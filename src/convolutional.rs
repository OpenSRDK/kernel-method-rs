use crate::Value;
use crate::{KernelError, PositiveDefiniteKernel};
use rayon::prelude::*;
use std::fmt::Debug;

pub trait Convolutable: Value {
    fn parts_len(&self) -> usize;
    fn part(&self, index: usize) -> &Vec<f64>;
    fn data_len(&self) -> usize;
}

impl Convolutable for Vec<f64> {
    fn parts_len(&self) -> usize {
        1
    }

    fn part(&self, _: usize) -> &Vec<f64> {
        self
    }

    fn data_len(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, Debug)]
pub struct Convolutional<K>
where
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    kernel: K,
}

impl<K> Convolutional<K>
where
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    pub fn new(kernel: K) -> Self {
        Self { kernel }
    }

    pub fn kernel_ref(&self) -> &K {
        &self.kernel
    }
}

impl<T, K> PositiveDefiniteKernel<T> for Convolutional<K>
where
    T: Convolutable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn params_len(&self) -> usize {
        self.kernel.params_len()
    }

    fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError> {
        if params.len() != self.kernel.params_len() {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        let p = x.parts_len();
        if p != xprime.parts_len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let fx = (0..p)
            .into_par_iter()
            .map(|pi| self.kernel.value(params, x.part(pi), xprime.part(pi)))
            .sum::<Result<f64, KernelError>>()?;

        Ok(fx)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = Convolutional::new(RBF);

        let test_value = kernel.value(&[1.0], &vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);

        match test_value {
            Err(KernelError::ParametersLengthMismatch) => (),
            _ => panic!(),
        };
    }
}
