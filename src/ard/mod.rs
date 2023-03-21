use super::PositiveDefiniteKernel;
use crate::{KernelAdd, KernelError, KernelMul};
use opensrdk_symbolic_computation::Expression;
use rayon::prelude::*;
use std::{ops::Add, ops::Mul};

fn weighted_norm_pow(x: Expression, x_prime: Expression, params: &[Expression]) -> Expression {
    params
        .par_iter()
        .zip(x.par_iter())
        .zip(x_prime.par_iter())
        .map(|((relevance, xi), x_primei)| relevance * (xi - x_primei).powi(2))
        .sum()
}
//must rewite this function!

#[derive(Clone, Debug)]
pub struct ARD(pub usize);

impl PositiveDefiniteKernel for ARD {
    fn params_len(&self) -> usize {
        self.0
    }

    fn expression(
        &self,
        x: Expression,
        x_prime: Expression,
        params: &[Expression],
    ) -> Result<Expression, KernelError> {
        if params.len() != self.0 {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        // if x.len() != self.0 || x_prime.len() != self.0 {
        //     return Err(KernelError::InvalidArgument.into());
        // }

        let fx = (-weighted_norm_pow(&params, x, x_prime)).exp();

        Ok(fx)
    }
}

impl<R> Add<R> for ARD
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<R> Mul<R> for ARD
where
    R: PositiveDefiniteKernel,
{
    type Output = KernelMul<Self, R>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

// impl ValueDifferentiableKernel<Vec<f64>> for ARD {
//     fn ln_diff_value(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff = params
//             .par_iter()
//             .zip(x.par_iter())
//             .zip(xprime.par_iter())
//             .map(|((relevance, xi), xprimei)| -2.0 * relevance * (xi - xprimei))
//             .collect::<Vec<f64>>();
//         Ok(diff)
//     }
// }

// impl ParamsDifferentiableKernel<Vec<f64>> for ARD {
//     fn ln_diff_params(
//         &self,
//         params: &[f64],
//         x: &Vec<f64>,
//         xprime: &Vec<f64>,
//     ) -> Result<Vec<f64>, KernelError> {
//         let diff = params
//             .par_iter()
//             .zip(x.par_iter())
//             .zip(xprime.par_iter())
//             .map(|((_relevance, xi), xprimei)| -(xi - xprimei).powi(2))
//             .collect::<Vec<f64>>();
//         Ok(diff)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     #[test]
//     fn it_works() {
//         let kernel = ARD(3);

//         let test_value = kernel
//             .value(&[1.0, 0.0, 0.0], &vec![1.0, 2.0, 3.0], &vec![0.0, 2.0, 1.0])
//             .unwrap();

//         assert_eq!(test_value, (-1f64).exp());
//     }
// }
