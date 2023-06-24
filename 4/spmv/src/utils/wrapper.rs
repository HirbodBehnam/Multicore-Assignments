use std::{
    mem::take,
    time::{Duration, Instant},
};

use crate::{Matrix, SpMV};

pub struct Wrapper {
    matrix: Matrix<f32>,
}

impl Wrapper {
    pub fn wrap(matrix: Matrix<f32>) -> Self {
        Self { matrix }
    }

    pub fn time_mul<W: SpMV>(&mut self, vector: Vec<f32>) -> Result<Duration, ()> {
        let matrix = take(&mut self.matrix);
        let valid_out = mul(&matrix, &vector);
        let spmv = W::from(matrix);

        let start = Instant::now();
        let spmv_out = spmv.mul(&vector);
        let duration = start.elapsed();

        match valid_out == spmv_out {
            true => Ok(duration),
            false => Err(()),
        }
    }

    pub fn check_mul<W: SpMV>(&mut self, vector: Vec<f32>) -> bool {
        let matrix = take(&mut self.matrix);
        let valid_out = mul(&matrix, &vector);
        let spmv_out = W::from(matrix).mul(&vector);

        valid_out == spmv_out
    }
}

fn mul(matrix: &Matrix<f32>, vector: &Vec<f32>) -> Vec<f32> {
    matrix
        .iter() // for each row...
        .map(
            |row| {
                row // map to the float in result vector
                    .iter()
                    .zip(vector.iter()) // combine each elem of row with vector
                    .map(|(x, y)| x * y) // multiply them
                    .sum()
            }, // and finally sum them
        )
        .collect()
}
