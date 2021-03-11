use super::Kernel;
use crate::{KernelAdd, KernelMul};
use std::{error::Error, fmt::Debug, ops::Add, ops::Mul};

pub type ValueFn<T> = dyn Fn(&[f64], &T, &T, bool) -> Result<(f64, Vec<f64>), Box<dyn Error>>;

#[derive(Clone)]
pub struct InstantKernel<'a, T>
where
    T: Clone + Debug,
{
    params_len: usize,
    value: &'a ValueFn<T>,
}

impl<'a, T> InstantKernel<'a, T>
where
    T: Clone + Debug,
{
    pub fn new(params_len: usize, value: &'a ValueFn<T>) -> Self {
        Self { params_len, value }
    }
}

impl<'a, T> Debug for InstantKernel<'a, T>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InstantKernel {{ params_len: {} }}", self.params_len)
    }
}

impl<'a, T> Kernel<T> for InstantKernel<'a, T>
where
    T: Clone + Debug,
{
    fn params_len(&self) -> usize {
        self.params_len
    }

    fn value(
        &self,
        params: &[f64],
        x: &T,
        xprime: &T,
        with_grad: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        (self.value)(params, x, xprime, with_grad)
    }
}

impl<'a, T, R> Add<R> for InstantKernel<'a, T>
where
    T: Clone + Debug,
    R: Kernel<T>,
{
    type Output = KernelAdd<Self, R, T>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<'a, T, R> Mul<R> for InstantKernel<'a, T>
where
    T: Clone + Debug,
    R: Kernel<T>,
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
        let kernel = RBF + InstantKernel::new(0, &|_, _, _, _| Ok((0.0, vec![])));

        let (func, grad) = kernel
            .value(
                &[1.0, 1.0],
                &vec![1.0, 2.0, 3.0],
                &vec![3.0, 2.0, 1.0],
                true,
            )
            .unwrap();

        println!("{}", func);
        println!("{:#?}", grad);
    }
}
