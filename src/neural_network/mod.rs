use std::fmt::Debug;

pub mod deep_neural_network;
pub mod relu;

pub trait ActivationFunction: Debug + Send + Sync {
    fn f(&self, previous_layer_kernel: (f64, f64, f64)) -> f64;
}
