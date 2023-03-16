use std::ops::{Add, Mul};

use super::{KernelAdd, KernelMul, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;

#[derive(Clone, Debug)]
pub struct Linear;

impl PositiveDefiniteKernel for Linear {
    fn expression(&self, x: Expression, x_prime: Expression, params: &[Expression]) -> Expression {
        x.clone().dot(x_prime, &[[0, 0]])
    }

    fn params_len(&self) -> usize {
        0
    }
}

impl<R> Add<R> for Linear
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        KernelAdd::new(self, rhs)
    }
}

impl<R> Mul<R> for Linear
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelMul<Self, R>;

    fn mul(self, rhs: R) -> Self::Output {
        KernelMul::new(self, rhs)
    }
}

// use super::PositiveDefiniteKernel;
// use crate::{
//     KernelAdd, KernelError, KernelMul, ParamsDifferentiableKernel, ValueDifferentiableKernel,
// };
// use opensrdk_linear_algebra::*;
// use rayon::prelude::*;
// use std::{ops::Add, ops::Mul};

// const PARAMS_LEN: usize = 0;

// #[derive(Clone, Debug)]
// pub struct Linear;

// impl PositiveDefiniteKernel<Vec<f64>> for Linear {
//     fn params_len(&self) -> usize {
//         PARAMS_LEN
//     }

//     fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
//         if params.len() != PARAMS_LEN {
//             return Err(KernelError::ParametersLengthMismatch.into());
//         }
//         if x.len() != xprime.len() {
//             return Err(KernelError::InvalidArgument.into());
//         }

//         let fx = x
//             .par_iter()
//             .zip(xprime.par_iter())
//             .map(|(x_i, xprime_i)| x_i * xprime_i)
//             .sum();

//         Ok(fx)
//     }
// }

// impl<R> Add<R> for Linear
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelAdd<Self, R, Vec<f64>>;

//     fn add(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl<R> Mul<R> for Linear
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelMul<Self, R, Vec<f64>>;

//     fn mul(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl ValueDifferentiableKernel<Vec<f64>> for Linear {
//     fn ln_diff_value(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let value = &self.value(params, x, xprime)?;
//         let diff = (2.0 / value * x.clone().col_mat()).vec();
//         Ok(diff)
//     }
// }

// impl ParamsDifferentiableKernel<Vec<f64>> for Linear {
//     fn ln_diff_params(
//         &self,
//         _params: &[f64],
//         _x: &Vec<f64>,
//         _xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff = vec![];
//         Ok(diff)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     #[test]
//     fn it_works() {
//         let kernel = Linear;

//         let test_value = kernel
//             .value(&[], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
//             .unwrap();

//         assert_eq!(test_value, 10.0);
//     }
// }
