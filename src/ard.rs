use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct ARD {
    data_len: usize,
    params: Vec<f64>,
}
