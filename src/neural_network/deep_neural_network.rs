use super::ActivationFunction;
use crate::{Bias, Kernel, KernelAdd, KernelError, KernelMul, Linear};
use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

/// https://arxiv.org/abs/1711.00165
#[derive(Clone, Debug)]
pub struct DeepNeuralNetwork<'a> {
    layers: Vec<&'a dyn ActivationFunction>,
}

impl<'a> Kernel<Vec<f64>> for DeepNeuralNetwork<'a> {
    fn params_len(&self) -> usize {
        2 * (1 + self.layers.len())
    }

    fn value(&self, params: &[f64], x: &Vec<f64>, xprime: &Vec<f64>) -> Result<f64, KernelError> {
        if params.len() != self.params_len() {
            return Err(KernelError::ParametersLengthMismatch.into());
        }
        if x.len() != xprime.len() {
            return Err(KernelError::InvalidArgument.into());
        }

        let layer0 = Bias + Bias * Linear;
        let mut previous_layer_kernel = (
            layer0.value(&params[0..2], x, xprime)?,
            layer0.value(&params[0..2], x, x)?,
            layer0.value(&params[0..2], xprime, xprime)?,
        );
        let params = &params[2..];

        for (i, &layer) in self.layers.iter().enumerate() {
            let sigma_b = params[(i + 1) * 2];
            let sigma_w = params[(i + 1) * 2 + 1];
            let f = layer.f(previous_layer_kernel);
            let fxx = layer.f((
                previous_layer_kernel.1,
                previous_layer_kernel.1,
                previous_layer_kernel.1,
            ));
            let fxpxp = layer.f((
                previous_layer_kernel.2,
                previous_layer_kernel.2,
                previous_layer_kernel.2,
            ));

            previous_layer_kernel = (
                sigma_b + sigma_w * f,
                sigma_b + sigma_w * fxx,
                sigma_b + sigma_w * fxpxp,
            );
        }

        Ok(previous_layer_kernel.0)
    }

    fn value_with_grad(
        &self,
        params: &[f64],
        x: &Vec<f64>,
        xprime: &Vec<f64>,
    ) -> Result<(f64, Vec<f64>), KernelError> {
        todo!("{:?}{:?}{:?}", params, x, xprime)
    }
}

impl<'a, R> Add<R> for DeepNeuralNetwork<'a>
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelAdd<Self, R, Vec<f64>>;
    fn add(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}

impl<'a, R> Mul<R> for DeepNeuralNetwork<'a>
where
    R: Kernel<Vec<f64>>,
{
    type Output = KernelMul<Self, R, Vec<f64>>;

    fn mul(self, rhs: R) -> Self::Output {
        Self::Output::new(self, rhs)
    }
}
