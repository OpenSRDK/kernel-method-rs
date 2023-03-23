use std::ops::{Add, Mul};

use crate::KernelError;

use super::{KernelAdd, KernelMul, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;

const PARAMS_LEN: usize = 1;

#[derive(Clone, Debug)]
pub struct Constant;

impl PositiveDefiniteKernel for Constant {
    fn expression(
        &self,
        x: Expression,
        x_prime: Expression,
        params: &[Expression],
    ) -> Result<Expression, KernelError> {
        if params.len() != PARAMS_LEN {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        // if x.len() != x_prime.len() {
        //     return Err(KernelError::InvalidArgument.into());
        // }
        Ok(params[0].clone())
    }

    fn params_len(&self) -> usize {
        1
    }
}

impl<R> Add<R> for Constant
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        KernelAdd::new(self, rhs)
    }
}

impl<R> Mul<R> for Constant
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelMul<Self, R>;

    fn mul(self, rhs: R) -> Self::Output {
        KernelMul::new(self, rhs)
    }
}

// use super::PositiveDefiniteKernel;
// use crate::{KernelAdd, KernelError, KernelMul};
// use crate::{ParamsDifferentiableKernel, Value, ValueDifferentiableKernel};
// use std::fmt::Debug;
// use std::{ops::Add, ops::Mul};

// const PARAMS_LEN: usize = 1;

// #[derive(Clone, Debug)]
// pub struct Constant;

// impl<T> PositiveDefiniteKernel<T> for Constant
// where
//     T: Value,
// {
//     fn params_len(&self) -> usize {
//         PARAMS_LEN
//     }

//     fn value(&self, params: &[f64], _: &T, _: &T) -> Result<f64, KernelError> {
//         if params.len() != PARAMS_LEN {
//             return Err(KernelError::ParametersLengthMismatch.into());
//         }

//         let fx = params[0];

//         Ok(fx)
//     }
// }

// impl<R> Add<R> for Constant
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelAdd<Self, R, Vec<f64>>;

//     fn add(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl<R> Mul<R> for Constant
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelMul<Self, R, Vec<f64>>;

//     fn mul(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl ValueDifferentiableKernel<Vec<f64>> for Constant {
//     fn ln_diff_value(
//         &self,
//         _params: &[f64],
//         x: &Vec<f64>,
//         _xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff = vec![0.0; x.len()];
//         Ok(diff)
//     }
// }

// impl ParamsDifferentiableKernel<Vec<f64>> for Constant {
//     fn ln_diff_params(
//         &self,
//         _params: &[f64],
//         _x: &Vec<f64>,
//         _xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff = vec![1.0];
//         Ok(diff)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     #[test]
//     fn it_works() {
//         let kernel = Constant;

//         let test_value = kernel
//             .value(&[1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
//             .unwrap();

//         assert_eq!(test_value, 1.0);
//     }
// }
