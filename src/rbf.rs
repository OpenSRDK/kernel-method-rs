use std::ops::{Add, Mul};

use super::{KernelAdd, KernelMul, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;

#[derive(Clone, Debug)]
pub struct RBF;

impl PositiveDefiniteKernel for RBF {
    fn expression(&self, x: Expression, x_prime: Expression, params: &[Expression]) -> Expression {
        let diff = x - x_prime;

        (-diff.clone().dot(diff, &[[0, 0]]) / (2.0 * params[0].clone().pow(2.0.into()))).exp()
    }

    fn params_len(&self) -> usize {
        1
    }
}

impl<R> Add<R> for RBF
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        KernelAdd::new(self, rhs)
    }
}

impl<R> Mul<R> for RBF
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
// pub struct RBF;

// impl RBF {
//     fn norm_pow(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<f64, KernelError> {
//         if params.len() != PARAMS_LEN {
//             return Err(KernelError::ParametersLengthMismatch.into());
//         }
//         if x.len() != xprime.len() {
//             return Err(KernelError::InvalidArgument.into());
//         }

//         let norm_pow = x
//             .par_iter()
//             .zip(xprime.par_iter())
//             .map(|(x_i, xprime_i)| (x_i - xprime_i).powi(2))
//             .sum();

//         Ok(norm_pow)
//     }
// }

// impl PositiveDefiniteKernel<Vec<f64>> for RBF {
//     fn params_len(&self) -> usize {
//         PARAMS_LEN
//     }

//     fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
//         let norm_pow = self.norm_pow(params, x, xprime)?;

//         let fx = params[0] * (-norm_pow / params[1]).exp();

//         Ok(fx)
//     }
// }

// impl<R> Add<R> for RBF
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelAdd<Self, R, Vec<f64>>;

//     fn add(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl<R> Mul<R> for RBF
// where
//     R: PositiveDefiniteKernel<Vec<f64>>,
// {
//     type Output = KernelMul<Self, R, Vec<f64>>;

//     fn mul(self, rhs: R) -> Self::Output {
//         Self::Output::new(self, rhs)
//     }
// }

// impl ValueDifferentiableKernel<Vec<f64>> for RBF {
//     fn ln_diff_value(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff = (-2.0 / params[1] * (x.clone().col_mat() - xprime.clone().col_mat())).vec();
//         Ok(diff)
//     }
// }

// impl ParamsDifferentiableKernel<Vec<f64>> for RBF {
//     fn ln_diff_params(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff0 = 1.0 / params[0];
//         let diff1 = 2.0 * params[1].powi(-2) * &self.norm_pow(params, x, xprime).unwrap();
//         let diff = vec![diff0, diff1];
//         Ok(diff)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     #[test]
//     fn it_works() {
//         let kernel = RBF;

//         //let (func, grad) = kernel
//         //    .value_with_grad(&[1.0, 1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
//         //    .unwrap();

//         //println!("{}", func);
//         //println!("{:#?}", grad);

//         let test_value = kernel
//             .value(&[1.0, 1.0], &vec![1.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0])
//             .unwrap();

//         assert_eq!(test_value, (-1f64).exp());
//     }
//     #[test]
//     fn it_works2() {
//         let kernel = RBF;

//         //let (func, grad) = kernel
//         //    .value_with_grad(&[1.0, 1.0], &vec![1.0, 2.0, 3.0], &vec![3.0, 2.0, 1.0])
//         //    .unwrap();

//         //println!("{}", func);
//         //println!("{:#?}", grad);

//         let test_value = kernel
//             .ln_diff_value(&[1.0, 1.0], &vec![1.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0])
//             .unwrap();

//         println!("{:?}", test_value);
//     }
// }
