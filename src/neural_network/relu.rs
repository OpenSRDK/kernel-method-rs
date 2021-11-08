use super::ActivationFunction;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct ReLU;

impl ActivationFunction for ReLU {
    fn f(&self, previous_layer_kernel: (f64, f64, f64)) -> f64 {
        let sqrt = (previous_layer_kernel.1 * previous_layer_kernel.2).sqrt();
        let theta = (previous_layer_kernel.0 / sqrt).acos();

        sqrt * (theta.sin() + (PI - theta) * theta.cos()) / 2.0 * PI
    }
}
