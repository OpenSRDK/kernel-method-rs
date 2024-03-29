use super::PositiveDefiniteKernel;
use crate::KernelError;
use crate::Value;
use crate::{KernelAdd, KernelMul};
use std::marker::PhantomData;
use std::{fmt::Debug, ops::Add, ops::Mul};

#[derive(Clone)]
pub struct InstantKernel<T, F>
where
    T: Value,
    F: Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Clone + Send + Sync,
{
    params_len: usize,
    value_function: F,
    phantom: PhantomData<T>,
}

impl<T, F> InstantKernel<T, F>
where
    T: Value,
    F: Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Clone + Send + Sync,
{
    pub fn new(params_len: usize, value_function: F) -> Self {
        Self {
            params_len,
            value_function,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Debug for InstantKernel<T, F>
where
    T: Value,
    F: Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InstantKernel {{ params_len: {} }}", self.params_len)
    }
}

impl<T, F> PositiveDefiniteKernel<T> for InstantKernel<T, F>
where
    T: Value,
    F: Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Clone + Send + Sync,
{
    fn params_len(&self) -> usize {
        self.params_len
    }

    fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError> {
        (self.value_function)(params, x, xprime)
    }
}

impl<T, R, F> Add<R> for InstantKernel<T, F>
where
    T: Value,
    R: PositiveDefiniteKernel<T>,
    F: Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Clone + Send + Sync,
{
    type Output = KernelAdd<Self, R, T>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<T, R, F> Mul<R> for InstantKernel<T, F>
where
    T: Value,
    R: PositiveDefiniteKernel<T>,
    F: Fn(&[f64], &T, &T) -> Result<f64, KernelError> + Clone + Send + Sync,
{
    type Output = KernelMul<Self, R, T>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = RBF + InstantKernel::new(0, |_, _, _| Ok(0.0));

        //let (func, grad) = kernel
        //    .value_with_grad(&[1.0, 1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
        //    .unwrap();

        //println!("{}", func);
        //println!("{:#?}", grad);

        let test_value = kernel
            .value(&[1.0, 1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
            .unwrap();

        println!("{}", test_value);
    }
}
