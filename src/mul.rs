use opensrdk_linear_algebra::Vector;

use crate::KernelError;
use crate::ParamsDifferentiableKernel;
use crate::Value;
use crate::ValueDifferentiableKernel;
use crate::{KernelAdd, PositiveDefiniteKernel};
use std::marker::PhantomData;
use std::ops::Add;
use std::{fmt::Debug, ops::Mul};

#[derive(Clone, Debug)]
pub struct KernelMul<L, R, T>
where
    L: PositiveDefiniteKernel<T>,
    R: PositiveDefiniteKernel<T>,
    T: Value,
{
    lhs: L,
    rhs: R,
    phantom: PhantomData<T>,
}

impl<L, R, T> KernelMul<L, R, T>
where
    L: PositiveDefiniteKernel<T>,
    R: PositiveDefiniteKernel<T>,
    T: Value,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self {
            lhs,
            rhs,
            phantom: PhantomData,
        }
    }
}

impl<L, R, T> PositiveDefiniteKernel<T> for KernelMul<L, R, T>
where
    L: PositiveDefiniteKernel<T>,
    R: PositiveDefiniteKernel<T>,
    T: Value,
{
    fn params_len(&self) -> usize {
        self.lhs.params_len() + self.rhs.params_len()
    }

    fn value(&self, params: &[f64], x: &T, xprime: &T) -> Result<f64, KernelError> {
        let lhs_params_len = self.lhs.params_len();
        let fx = self.lhs.value(&params[..lhs_params_len], x, xprime)?;
        let gx = self.rhs.value(&params[lhs_params_len..], x, xprime)?;

        let hx = fx * gx;

        Ok(hx)
    }
}

impl<Rhs, L, R, T> Add<Rhs> for KernelMul<L, R, T>
where
    Rhs: PositiveDefiniteKernel<T>,
    L: PositiveDefiniteKernel<T>,
    R: PositiveDefiniteKernel<T>,
    T: Value,
{
    type Output = KernelAdd<Self, Rhs, T>;

    fn add(self, rhs: Rhs) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<Rhs, L, R, T> Mul<Rhs> for KernelMul<L, R, T>
where
    Rhs: PositiveDefiniteKernel<T>,
    L: PositiveDefiniteKernel<T>,
    R: PositiveDefiniteKernel<T>,
    T: Value,
{
    type Output = KernelMul<Self, Rhs, T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<L, R, T> ValueDifferentiableKernel<T> for KernelMul<L, R, T>
where
    L: ValueDifferentiableKernel<T>,
    R: ValueDifferentiableKernel<T>,
    T: Value,
{
    fn ln_diff_value(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError> {
        let diff_rhs = &self
            .rhs
            .ln_diff_value(params, x, xprime)
            .unwrap()
            .clone()
            .col_mat();
        let diff_lhs = &self
            .lhs
            .ln_diff_value(params, x, xprime)
            .unwrap()
            .clone()
            .col_mat();
        let diff = (diff_rhs + diff_lhs.clone()).vec();
        Ok(diff)
    }
}

impl<L, R, T> ParamsDifferentiableKernel<T> for KernelMul<L, R, T>
where
    L: ParamsDifferentiableKernel<T>,
    R: ParamsDifferentiableKernel<T>,
    T: Value,
{
    fn ln_diff_params(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError> {
        let diff_rhs = &self
            .rhs
            .ln_diff_params(params, x, xprime)
            .unwrap()
            .clone()
            .col_mat();
        let diff_lhs = &self
            .lhs
            .ln_diff_params(params, x, xprime)
            .unwrap()
            .clone()
            .col_mat();
        let diff = (diff_rhs + diff_lhs.clone()).vec();
        Ok(diff)
    }
}
