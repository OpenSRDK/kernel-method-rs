use super::Kernel;
use crate::KernelError;
use rayon::prelude::*;

fn weighted_norm_pow(x: &[f64], x_prime: &[f64], params: &[f64]) -> f64 {
    x.par_iter()
        .zip(x_prime.par_iter())
        .zip(params.par_iter())
        .map(|((x_i, x_prime_i), relevance)| relevance * (x_i - x_prime_i).powi(2))
        .sum()
}

pub fn ard(data_len: usize) -> Kernel<[f64]> {
    Kernel::<[f64]>::new(
        vec![1.0; data_len],
        Box::new(
            move |x: &[f64], x_prime: &[f64], with_grad: bool, params: &[f64]| {
                if x.len() != data_len || x_prime.len() != data_len {
                    return Err(KernelError::InvalidArgument.into());
                }

                let func = (-weighted_norm_pow(x, x_prime, &params)).exp();

                let grad = if !with_grad {
                    None
                } else {
                    let mut grad = vec![f64::default(); data_len];

                    grad.par_iter_mut()
                        .zip(x.par_iter())
                        .zip(x_prime.par_iter())
                        .for_each(|((s, &l), &r)| *s = -(l - r).powi(2));

                    Some(grad)
                };

                Ok((func, grad))
            },
        ),
    )
}
