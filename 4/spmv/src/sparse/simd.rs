use crate::{Matrix, SpMV};

use super::RowElement;

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    inner: Vec<Vec<RowElement<f32>>>,
}

impl SparseMatrix {
    unsafe fn mul_avx2(&self, vector: &Vec<f32>) -> Vec<f32> {
        use std::arch::x86_64::*;
        let mut result: Vec<f32> = Vec::with_capacity(self.inner.len());
        for row in &self.inner {
            // for each row...
            let mut acc = 0f32;
            for chunk in row.chunks(8) {
                // chunk them into 8 elem data
                let mut buffer = chunk.iter().map(|c| c.value).collect::<Vec<f32>>();
                buffer.resize(8, 0f32);
                let elems = _mm256_set_ps(buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7]);
                // extract the pairwise elements from vector
                buffer = chunk.iter().map(|c| vector[c.index]).collect::<Vec<f32>>();
                buffer.resize(8, 0f32);
                let vec_elems = _mm256_set_ps(buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7]);
                // mult them together. This register has 8 float values
                let result = _mm256_mul_ps(elems, vec_elems);
                // mostly from https://stackoverflow.com/a/9776522/4213397
                /*
                 * sum[0] = x[0] + x[1]
                 * sum[1] = x[2] + x[3]
                 * sum[2] = x[0] + x[1]
                 * sum[3] = x[2] + x[3]
                 * sum[4] = x[4] + x[5]
                 * sum[5] = x[6] + x[7]
                 * sum[6] = x[4] + x[5]
                 * sum[7] = x[6] + x[7]
                 */
                let mut sum = _mm256_hadd_ps(result, result);
                /*
                 * sum[0] = x[0] + x[1] + x[2] + x[3]
                 * sum[1] = x[0] + x[1] + x[2] + x[3]
                 * sum[2] = x[0] + x[1] + x[2] + x[3]
                 * sum[3] = x[0] + x[1] + x[2] + x[3]
                 * sum[4] = x[4] + x[5] + x[6] + x[7]
                 * sum[5] = x[4] + x[5] + x[6] + x[7]
                 * sum[6] = x[4] + x[5] + x[6] + x[7]
                 * sum[7] = x[4] + x[5] + x[6] + x[7]
                 */
                sum = _mm256_hadd_ps(sum, sum);
                let sum_high = _mm256_extractf128_ps(sum, 1);
                let final_sum = _mm_add_ps(sum_high, _mm256_castps256_ps128(sum));
                // accumulate
                acc += _mm_cvtss_f32(final_sum);
            }
            result.push(acc);
        }
        return result;
    }
}

impl SpMV for SparseMatrix {
    fn mul(&self, vector: &Vec<f32>) -> Vec<f32> {
        unsafe { self.mul_avx2(vector) }
    }
}

impl From<Matrix<f32>> for SparseMatrix {
    fn from(matrix: Matrix<f32>) -> Self {
        SparseMatrix {
            inner: matrix
                .iter() // for each row...
                .map(|row| {
                    // map to sparse matrix rows
                    row.iter()
                        .enumerate() // get the index and value together
                        .filter_map(|(index, cell)| {
                            // filter out the ones which have zero value
                            if *cell == 0f32 {
                                return None;
                            } else {
                                return Some(RowElement {
                                    index,
                                    value: *cell,
                                });
                            }
                        })
                        .collect::<Vec<RowElement<f32>>>() // convert to row
                })
                .collect::<Vec<Vec<RowElement<f32>>>>(), // convert to matrix
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;

    const DEBUG_TEST_SIZE: usize = 10;
    const DEBUG_MATRIX_SIZE: usize = 10;
    const STRESS_TEST_SIZE: usize = 100;
    const STRESS_MATRIX_SIZE: usize = 1000;

    type SpmvMatrix = SparseMatrix;

    #[test]
    fn smoke() {
        let matrix: Matrix<f32> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let spmv = SpmvMatrix::from(matrix);

        let mut testcases: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        testcases.push((vec![1.0, 1.0, 1.0], vec![6.0, 15.0, 24.0]));
        testcases.push((vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]));
        testcases.push((vec![1.0, 0.0, 0.0], vec![1.0, 4.0, 7.0]));
        testcases.push((vec![3.0, 2.0, 1.0], vec![10.0, 28.0, 46.0]));
        testcases.push((vec![1.0, -1.0, 1.0], vec![2.0, 5.0, 8.0]));

        for testcase in testcases {
            assert_eq!(spmv.mul(&testcase.0), testcase.1);
        }
    }

    #[test]
    fn random_debug() {
        let gen = utils::Generator::with_sparsity(0.3);
        for _ in 0..DEBUG_TEST_SIZE {
            let matrix = gen.debug_matrix(DEBUG_MATRIX_SIZE);
            let vector = gen.debug_vector(DEBUG_MATRIX_SIZE);
            let mut wrapper = utils::Wrapper::wrap(matrix);
            wrapper.check_mul::<SpmvMatrix>(vector);
        }
    }

    #[test]
    fn random() {
        let gen = utils::Generator::with_sparsity(0.3);
        for _ in 0..STRESS_TEST_SIZE {
            let matrix = gen.matrix(STRESS_MATRIX_SIZE);
            let vector = gen.vector(STRESS_MATRIX_SIZE);
            let mut wrapper = utils::Wrapper::wrap(matrix);
            wrapper.check_mul::<SpmvMatrix>(vector);
        }
    }
}
