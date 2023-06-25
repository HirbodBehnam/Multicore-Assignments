pub mod dense;
pub mod sparse;
pub mod utils;

type Matrix<T> = Vec<Vec<T>>;
pub trait SpMV: From<Matrix<f32>> {
    fn mul(&self, vector: &Vec<f32>) -> Vec<f32>;
}