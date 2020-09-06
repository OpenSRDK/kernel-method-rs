use super::Kernel;
use crate::KernelError;
use rayon::prelude::*;

const PARAMS_LEN: usize = 1;

fn norm(x: &[f64], x_prime: &[f64]) -> f64 {
    x.par_iter()
        .zip(x_prime.par_iter())
        .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn exponential(params: [f64; PARAMS_LEN]) -> Kernel<[f64]> {
    Kernel::<[f64]>::new(
        params.to_vec(),
        Box::new(
            |x: &[f64], x_prime: &[f64], with_grad: bool, params: &[f64]| {
                if x.len() != x_prime.len() {
                    return Err(KernelError::InvalidArgument.into());
                }

                let norm = norm(x, x_prime);

                let func = (-norm / params[0]).exp();

                let grad = if !with_grad {
                    None
                } else {
                    let mut grad = vec![f64::default(); PARAMS_LEN];

                    grad[0] = (-norm / params[0]).exp() / params[0].powi(2);

                    Some(grad)
                };

                Ok((func, grad))
            },
        ),
    )
}
