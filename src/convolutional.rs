use crate::{KernelError, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::Expression;
use rayon::prelude::*;
use std::fmt::Debug;

pub trait Convolutable {
    fn parts_len(&self) -> usize;
    fn part(&self, index: usize) -> &Expression;
}

impl Convolutable for Expression {
    fn parts_len(&self) -> usize {
        1
    }

    fn part(&self, _: usize) -> &Expression {
        self
    }
}

#[derive(Clone, Debug)]
pub struct Convolutional<K>
where
    K: PositiveDefiniteKernel,
{
    kernel: K,
}

impl<K> Convolutional<K>
where
    K: PositiveDefiniteKernel,
{
    pub fn new(kernel: K) -> Self {
        Self { kernel }
    }

    pub fn kernel_ref(&self) -> &K {
        &self.kernel
    }
}

impl<K> PositiveDefiniteKernel for Convolutional<K>
where
    K: PositiveDefiniteKernel,
{
    fn params_len(&self) -> usize {
        self.kernel.params_len()
    }

    fn expression(
        &self,
        x: &Expression,
        x_prime: &Expression,
        params: &[Expression],
    ) -> Result<Expression, KernelError> {
        if params.len() != self.kernel.params_len() {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        let p = x.parts_len();
        if p != x_prime.parts_len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let fx = (0..p)
            .into_par_iter()
            .map(|pi| self.kernel.expression(x.part(pi), x_prime.part(pi), params))
            .sum::<Result<Expression, KernelError>>()?;

        Ok(fx)
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     #[test]
//     fn it_works() {
//         let kernel = Convolutional::new(RBF);

//         let test_value = kernel.value(&[1.0], &vec![0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0]);

//         match test_value {
//             Err(KernelError::ParametersLengthMismatch) => (),
//             _ => panic!(),
//         };
//     }
// }
