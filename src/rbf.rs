use super::Kernel;
use crate::KernelError;
use rayon::prelude::*;

const PARAMS_LEN: usize = 1;

fn norm_pow(x: &[f64], x_prime: &[f64]) -> f64 {
    x.par_iter()
        .zip(x_prime.par_iter())
        .map(|(x_i, x_prime_i)| (x_i - x_prime_i).powi(2))
        .sum()
}

pub fn rbf() -> Kernel<[f64]> {
    Kernel::<[f64]>::new(
        vec![1000.0; PARAMS_LEN],
        Box::new(
            |x: &[f64], x_prime: &[f64], with_grad: bool, params: &[f64]| {
                if x.len() != x_prime.len() {
                    return Err(KernelError::InvalidArgument.into());
                }

                let norm_pow = norm_pow(x, x_prime);

                let func = (-norm_pow / params[0]).exp();

                let grad = if !with_grad {
                    None
                } else {
                    let mut grad = vec![f64::default(); PARAMS_LEN];

                    grad[0] = (-norm_pow / params[0]).exp() * (norm_pow / params[0].powi(2));

                    Some(grad)
                };

                Ok((func, grad))
            },
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let kernel = rbf();

        let (func, grad) = kernel
            .func(&vec![1.0, 2.0, 3.0], &vec![10.0, 20.0, 30.0], true, None)
            .unwrap();

        println!("{}", func);
        println!("{:#?}", grad);
    }
}
