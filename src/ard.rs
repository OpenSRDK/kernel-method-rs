use super::Kernel;
use crate::KernelError;
use rayon::prelude::*;

const PARAMS_LEN: usize = 3;

fn weighted_norm_pow(x: &[f64], x_prime: &[f64], params: &[f64]) -> f64 {
    let relevances = &params[PARAMS_LEN..];

    x.par_iter()
        .zip(x_prime.par_iter())
        .zip(relevances.par_iter())
        .map(|((x_i, x_prime_i), relevance)| relevance * (x_i - x_prime_i).powi(2))
        .sum()
}

fn prod(x: &[f64], x_prime: &[f64]) -> f64 {
    x.par_iter()
        .zip(x_prime.par_iter())
        .map(|(x_i, x_prime_i)| x_i * x_prime_i)
        .sum()
}

pub fn ard(params: [f64; PARAMS_LEN], data_len: usize) -> Kernel<[f64]> {
    Kernel::<[f64]>::new(
        [params.to_vec(), vec![1.0; data_len]].concat(),
        Box::new(
            move |x: &[f64], x_prime: &[f64], with_grad: bool, params: &[f64]| {
                if x.len() != data_len || x_prime.len() != data_len {
                    return Err(KernelError::InvalidArgument.into());
                }

                let func = params[0] * (-weighted_norm_pow(x, x_prime, &params)).exp()
                    + params[1]
                    + params[2] * prod(x, x_prime);

                let grad = if !with_grad {
                    None
                } else {
                    let mut grad = vec![f64::default(); PARAMS_LEN + data_len];

                    grad[0] = (-weighted_norm_pow(&x, &x_prime, params)).exp();
                    grad[1] = 1.0;
                    grad[2] = prod(&x, &x_prime);

                    let relevances_grad = &mut grad[PARAMS_LEN..];

                    relevances_grad
                        .par_iter_mut()
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
