use crate::KernelError;
use crate::LogParamsDifferentiableKernel;
use crate::LogValueDifferentiableKernel;
use crate::Value;
use crate::{KernelMul, PositiveDefiniteKernel};
use opensrdk_linear_algebra::Vector;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::{ops::Add, ops::Mul};

#[derive(Clone, Debug)]
pub struct KernelAdd<L, R, T>
where
    L: PositiveDefiniteKernel<T>,
    R: PositiveDefiniteKernel<T>,
    T: Value,
{
    lhs: L,
    rhs: R,
    phantom: PhantomData<T>,
}

impl<L, R, T> KernelAdd<L, R, T>
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

impl<L, R, T> PositiveDefiniteKernel<T> for KernelAdd<L, R, T>
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

        let hx = fx + gx;

        Ok(hx)
    }
}

impl<Rhs, L, R, T> Add<Rhs> for KernelAdd<L, R, T>
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

impl<Rhs, L, R, T> Mul<Rhs> for KernelAdd<L, R, T>
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

impl<L, R, T> LogValueDifferentiableKernel<T> for KernelAdd<L, R, T>
where
    L: LogValueDifferentiableKernel<T>,
    R: LogValueDifferentiableKernel<T>,
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
        let value_rhs = vec![self.rhs.value(params, x, xprime).unwrap()].col_mat();
        let value_lhs = vec![self.lhs.value(params, x, xprime).unwrap()].col_mat();
        let diff = ((&value_rhs * diff_rhs + &value_lhs * diff_lhs)
            * (&value_rhs + value_lhs)[(0, 0)].powi(-1))
        .vec();
        Ok(diff)
    }
}

impl<L, R, T> LogParamsDifferentiableKernel<T> for KernelAdd<L, R, T>
where
    L: LogParamsDifferentiableKernel<T>,
    R: LogParamsDifferentiableKernel<T>,
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
        let value_rhs = vec![self.rhs.value(params, x, xprime).unwrap()].col_mat();
        let value_lhs = vec![self.lhs.value(params, x, xprime).unwrap()].col_mat();
        let diff = ((&value_rhs * diff_rhs + &value_lhs * diff_lhs)
            * (&value_rhs + value_lhs)[(0, 0)].powi(-1))
        .vec();
        Ok(diff)
    }
}
