pub mod dense;
pub mod sparse;
pub mod utils;

type Matrix<T> = Vec<Vec<T>>;
pub trait SpMV: From<Matrix<f32>> {
    fn mul(&self, vector: &Vec<f32>) -> Vec<f32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_check() {
        struct Dummy {
            inner: Matrix<f32>,
        }

        impl From<Matrix<f32>> for Dummy {
            fn from(inner: Matrix<f32>) -> Self {
                Self { inner }
            }
        }
        impl SpMV for Dummy {
            fn mul(&self, vector: &Vec<f32>) -> Vec<f32> {
                self.inner
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
        }

        let matrix: Matrix<f32> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let vector = vec![3.0, 2.0, 1.0];
        assert_eq!(Dummy::from(matrix).mul(&vector), vec![18.0, 30.0, 24.0]);

        let matrix: Matrix<f32> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let vector = vec![1.0, 0.0, 0.0];
        assert_eq!(Dummy::from(matrix).mul(&vector), vec![6.0, 0.0, 0.0]);
    }
}
