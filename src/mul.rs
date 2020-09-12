use crate::{Kernel, KernelAdd};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::{error::Error, ops::Add};
use std::{fmt::Debug, ops::Mul};

#[derive(Clone, Debug)]
pub struct KernelMul<L, R, T>
where
    L: Kernel<T>,
    R: Kernel<T>,
    T: Clone + Debug,
{
    lhs: L,
    rhs: R,
    phantom: PhantomData<T>,
}

impl<L, R, T> KernelMul<L, R, T>
where
    L: Kernel<T>,
    R: Kernel<T>,
    T: Clone + Debug,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self {
            lhs,
            rhs,
            phantom: PhantomData,
        }
    }
}

impl<L, R, T> Kernel<T> for KernelMul<L, R, T>
where
    L: Kernel<T>,
    R: Kernel<T>,
    T: Clone + Debug,
{
    fn params_len(&self) -> usize {
        self.lhs.params_len() + self.rhs.params_len()
    }

    fn value(
        &self,
        params: &[f64],
        x: &T,
        xprime: &T,
        with_grad: bool,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        let lhs_params_len = self.lhs.params_len();
        let (fx, dfx) = self
            .lhs
            .value(&params[..lhs_params_len], x, xprime, with_grad)?;
        let (gx, dgx) = self
            .rhs
            .value(&params[lhs_params_len..], x, xprime, with_grad)?;

        let hx = fx * gx;

        let ghx = if !with_grad {
            vec![]
        } else {
            let ghx = dfx
                .par_iter()
                .map(|dfxi| dfxi * gx)
                .chain(dgx.par_iter().map(|dgxi| fx * dgxi))
                .collect::<Vec<_>>();

            ghx
        };

        Ok((hx, ghx))
    }
}

impl<Rhs, L, R, T> Add<Rhs> for KernelMul<L, R, T>
where
    Rhs: Kernel<T>,
    L: Kernel<T>,
    R: Kernel<T>,
    T: Clone + Debug,
{
    type Output = KernelAdd<Self, Rhs, T>;

    fn add(self, rhs: Rhs) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<Rhs, L, R, T> Mul<Rhs> for KernelMul<L, R, T>
where
    Rhs: Kernel<T>,
    L: Kernel<T>,
    R: Kernel<T>,
    T: Clone + Debug,
{
    type Output = KernelMul<Self, Rhs, T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}
