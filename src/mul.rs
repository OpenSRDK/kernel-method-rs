use crate::{KernelAdd, KernelError, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;
use std::ops::Add;
use std::{fmt::Debug, ops::Mul};

#[derive(Clone, Debug)]
pub struct KernelMul<L, R>
where
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    lhs: L,
    rhs: R,
}

impl<L, R> KernelMul<L, R>
where
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self { lhs, rhs }
    }
}

impl<L, R> PositiveDefiniteKernel for KernelMul<L, R>
where
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    fn params_len(&self) -> usize {
        self.lhs.params_len() + self.rhs.params_len()
    }
    fn expression(
        &self,
        x: Expression,
        x_prime: Expression,
        params: &[Expression],
    ) -> Result<Expression, KernelError> {
        let lhs_params_len = self.lhs.params_len();
        let fx = self.lhs.expression(&params[..lhs_params_len], x, x_prime)?;
        let gx = self.rhs.expression(&params[lhs_params_len..], x, x_prime)?;

        let hx = fx * gx;

        Ok(hx)
    }
}

impl<Rhs, L, R> Add<Rhs> for KernelMul<L, R>
where
    Rhs: PositiveDefiniteKernel,
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    type Output = KernelAdd<Self, Rhs>;

    fn add(self, rhs: Rhs) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<Rhs, L, R> Mul<Rhs> for KernelMul<L, R>
where
    Rhs: PositiveDefiniteKernel,
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    type Output = KernelMul<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

// impl<L, R> ValueDifferentiableKernel for KernelMul<L, R>
// where
//     L: ValueDifferentiableKernel,
//     R: ValueDifferentiableKernel<T>,
//     T: Value,
// {
//     fn ln_diff_value(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError> {
//         let diff_rhs = &self
//             .rhs
//             .ln_diff_value(params, x, xprime)
//             .unwrap()
//             .clone()
//             .col_mat();
//         let diff_lhs = &self
//             .lhs
//             .ln_diff_value(params, x, xprime)
//             .unwrap()
//             .clone()
//             .col_mat();
//         let diff = (diff_rhs + diff_lhs.clone()).vec();
//         Ok(diff)
//     }
// }

// impl<L, R, T> ParamsDifferentiableKernel<T> for KernelMul<L, R, T>
// where
//     L: ParamsDifferentiableKernel<T>,
//     R: ParamsDifferentiableKernel<T>,
//     T: Value,
// {
//     fn ln_diff_params(&self, params: &[f64], x: &T, xprime: &T) -> Result<Vec<f64>, KernelError> {
//         let diff_rhs = &self
//             .rhs
//             .ln_diff_params(params, x, xprime)
//             .unwrap()
//             .clone()
//             .col_mat();
//         let diff_lhs = &self
//             .lhs
//             .ln_diff_params(params, x, xprime)
//             .unwrap()
//             .clone()
//             .col_mat();
//         let diff = (diff_rhs + diff_lhs.clone()).vec();
//         Ok(diff)
//     }
// }
