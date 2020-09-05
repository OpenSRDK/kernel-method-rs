use super::Kernel;
use crate::KernelError;
use rayon::prelude::*;

pub fn linear() -> Kernel<[f64]> {
    Kernel::<[f64]>::new(
        vec![],
        Box::new(|x: &[f64], x_prime: &[f64], with_grad: bool, _: &[f64]| {
            if x.len() != x_prime.len() {
                return Err(Box::new(KernelError::InvalidArgument));
            }

            let func = x
                .par_iter()
                .zip(x_prime.par_iter())
                .map(|(x_i, x_prime_i)| x_i * x_prime_i)
                .sum();

            let grad = if !with_grad { None } else { Some(vec![]) };

            Ok((func, grad))
        }),
    )
}
