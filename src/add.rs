use crate::{KernelMul, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;
use std::fmt::Debug;
use std::{ops::Add, ops::Mul};

#[derive(Clone, Debug)]
pub struct KernelAdd<L, R>
where
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    lhs: L,
    rhs: R,
}

impl<L, R> KernelAdd<L, R>
where
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self { lhs, rhs }
    }
}

impl<L, R> PositiveDefiniteKernel for KernelAdd<L, R>
where
    L: PositiveDefiniteKernel,
    R: PositiveDefiniteKernel,
{
    fn params_len(&self) -> usize {
        self.lhs.params_len() + self.rhs.params_len()
    }

    fn expression(&self, x: Expression, x_prime: Expression, params: &[Expression]) -> Expression {
        let lhs_params_len = self.lhs.params_len();
        let fx = self.lhs.expression(&params[..lhs_params_len], x, x_prime)?;
        let gx = self.rhs.expression(&params[lhs_params_len..], x, x_prime)?;

        let hx = fx + gx;

        Ok(hx)
    }
}

impl<Rhs, L, R> Add<Rhs> for KernelAdd<L, R>
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

impl<Rhs, L, R> Mul<Rhs> for KernelAdd<L, R>
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

// impl<L, R> ValueDifferentiableKernel for KernelAdd<L, R>
// where
//     L: ValueDifferentiableKernel<T>,
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
//         let value_rhs = vec![self.rhs.value(params, x, xprime).unwrap()].col_mat();
//         let value_lhs = vec![self.lhs.value(params, x, xprime).unwrap()].col_mat();
//         let diff = ((&value_rhs * diff_rhs + &value_lhs * diff_lhs)
//             * (&value_rhs + value_lhs)[(0, 0)].powi(-1))
//         .vec();
//         Ok(diff)
//     }
// }

// impl<L, R, T> ParamsDifferentiableKernel<T> for KernelAdd<L, R, T>
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
//         let value_rhs = vec![self.rhs.value(params, x, xprime).unwrap()].col_mat();
//         let value_lhs = vec![self.lhs.value(params, x, xprime).unwrap()].col_mat();
//         let diff = ((&value_rhs * diff_rhs + &value_lhs * diff_lhs)
//             * (&value_rhs + value_lhs)[(0, 0)].powi(-1))
//         .vec();
//         Ok(diff)
//     }
// }
