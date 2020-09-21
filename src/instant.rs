use super::Kernel;
use crate::{KernelAdd, KernelMul};
use std::{error::Error, fmt::Debug, ops::Add, ops::Mul};

pub type ValueFn =
    dyn Fn(&[f64], &Vec<f64>, &Vec<f64>, bool) -> Result<(f64, Vec<f64>), Box<dyn Error>>;

#[derive(Clone)]
pub struct InstantKernel<'a> {
    params_len: usize,
    value: &'a ValueFn,
}

impl<'a> InstantKernel<'a> {
    pub fn new(params_len: usize, value: &'a ValueFn) -> Self {
        Self { params_len, value }
    }
}

impl<'a> Debug for InstantKernel<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instant {{ params_len: {} }}", self.params_len)
    }
}

impl<'a> Kernel<Vec<f64>> for InstantKernel<'a> {
    fn params_len(&self) -> usize {
        self.params_len
    }

    fn value(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
        with_grad: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        (self.value)(params, x, xprime, with_grad)
    }
}

impl<'a, R> Add<R> for InstantKernel<'a>
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<'a, R> Mul<R> for InstantKernel<'a>
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}
