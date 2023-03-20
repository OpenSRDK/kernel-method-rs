use std::ops::{Add, Mul};

use crate::KernelError;

use super::{KernelAdd, KernelMul, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;

const PARAMS_LEN: usize = 2;

#[derive(Clone, Debug)]
pub struct Periodic;

impl PositiveDefiniteKernel for Periodic {
    fn expression(
        &self,
        x: Expression,
        x_prime: Expression,
        params: &[Expression],
    ) -> Result<Expression, KernelError> {
        if params.len() != PARAMS_LEN {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != x_prime.len() {
            return Err(KernelError::InvalidArgument.into());
        }
        let diff = x - x_prime;

        Ok((params[0] * (diff.clone().dot(diff, &[[0, 0]]).sqrt() / params[1]).cos()).exp())
    }

    fn params_len(&self) -> usize {
        2
    }
}

impl<R> Add<R> for Periodic
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        KernelAdd::new(self, rhs)
    }
}

impl<R> Mul<R> for Periodic
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
// use opensrdk_linear_algebra::Vector;
// use rayon::prelude::*;
// use std::{ops::Add, ops::Mul};

// const PARAMS_LEN: usize = 2;

// #[derive(Clone, Debug)]
// pub struct Periodic;

// impl Periodic {
//     fn norm(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
//         if params.len() != PARAMS_LEN {
//             return Err(KernelError::ParametersLengthMismatch.into());
//         }
//         if x.len() != xprime.len() {
//             return Err(KernelError::InvalidArgument.into());
//         }

//         let v = x
//             .par_iter()
//             .zip(xprime.par_iter())
//             .map(|(x_i, xprime_i)| (x_i - xprime_i).powi(2))
//             .sum::<f64>()
//             .sqrt();

//         Ok(v)
//     }
// }

// impl PositiveDefiniteKernel<Vec<f64>> for Periodic {
//     fn params_len(&self) -> usize {
//         PARAMS_LEN
//     }

//     fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
//         let norm = self.norm(params, x, xprime)?;

//         let fx = (params[0] * (norm / params[1]).cos()).exp();

//         Ok(fx)
//     }
// }

// impl<R> Add<R> for Periodic
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelAdd<Self, R, Vec<f64>>;

//     fn add(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl<R> Mul<R> for Periodic
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelMul<Self, R, Vec<f64>>;

//     fn mul(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl ValueDifferentiableKernel<Vec<f64>> for Periodic {
//     fn ln_diff_value(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let value = &self.value(params, x, xprime)?;
//         let diff = (-value.sin() * 2.0 / params[1]
//             * (x.clone().col_mat() - xprime.clone().col_mat()))
//         .vec();
//         Ok(diff)
//     }
// }

// impl ParamsDifferentiableKernel<Vec<f64>> for Periodic {
//     fn ln_diff_params(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let value = &self.value(params, x, xprime)?;
//         let diff0 = 1.0 / params[0];
//         let diff1 = value.sin() * 2.0 * params[1].powi(-2) * &self.norm(params, x, xprime)?;
//         let diff = vec![diff0, diff1];
//         Ok(diff)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     #[test]
//     fn it_works() {
//         let kernel = Periodic;

//         let test_value = kernel.value(&[1.0], &vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);

//         match test_value {
//             Err(KernelError::ParametersLengthMismatch) => (),
//             _ => panic!(),
//         };
//     }

//     #[test]
//     fn it_works2() {
//         let kernel = Periodic;

//         let test_value = kernel
//             .value(&[1.0, 1.0], &vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0])
//             .unwrap();

//         assert_eq!(test_value, 1f64.exp());
//     }
// }
