use super::Kernel;

const PARAMS_LEN: usize = 1;

pub fn bias<T>() -> Kernel<T>
where
    T: ?Sized,
{
    Kernel::<T>::new(
        vec![1.0; PARAMS_LEN],
        Box::new(|_: &T, _: &T, with_grad: bool, params: &[f64]| {
            let func = params[0];

            let grad = if !with_grad { None } else { Some(vec![1.0]) };

            Ok((func, grad))
        }),
    )
}
