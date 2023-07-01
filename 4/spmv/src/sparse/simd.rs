use crate::{Matrix, SpMV};

#[repr(align(32))]
#[derive(Default)]
struct YMMBuffer([f32; 8]);

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    inner: Vec<(Vec<f32>, Vec<usize>)>,
}

impl SparseMatrix {
    unsafe fn mul_avx2(&self, vector: &Vec<f32>) -> Vec<f32> {
        use std::arch::x86_64::*;
        let mut result: Vec<f32> = Vec::with_capacity(self.inner.len());
        let mut vec_buffer = YMMBuffer::default();
        for row in &self.inner {
            assert!(row.0.len() == row.1.len());
            let (value_before, value_aligned, value_after) = row.0.align_to::<YMMBuffer>();
            let (index_before, index_aligned, index_after) = row.1.align_to::<[usize; 8]>();
            // for each row...
            let mut accumulator = _mm256_setzero_ps();
            for (values, indexes) in value_aligned.iter().zip(index_aligned.iter()) {
                // Load values into YMM registers
                let current_matrix = _mm256_load_ps(values.0.as_ptr());
                for (buf_index, vec_index) in indexes.iter().enumerate() {
                    vec_buffer.0[buf_index] = vector[*vec_index];
                }
                let current_vector = _mm256_load_ps(vec_buffer.0.as_ptr());
                // Fuse mult add them
                accumulator = _mm256_fmadd_ps(current_matrix, current_vector, accumulator);
            }
            // Now sum the remanding by hand
            let mut final_accumulator = 0f32;
            for (value, index) in value_before.iter().zip(index_before.iter()) {
                final_accumulator += vector[*index] * (*value);
            }
            for (value, index) in value_after.iter().zip(index_after.iter()) {
                final_accumulator += vector[*index] * (*value);
            }
            _mm256_store_ps(vec_buffer.0.as_mut_ptr(), accumulator);
            for result in vec_buffer.0 {
                final_accumulator += result;
            }
            result.push(final_accumulator);
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
        let mut inner = Vec::<(Vec<f32>, Vec<usize>)>::with_capacity(matrix.len());
        for row in matrix {
            let mut row_values = Vec::<f32>::new();
            let mut row_indexes = Vec::<usize>::new();
            for (index, elem) in row.iter().enumerate() {
                if *elem != 0.0 {
                    row_values.push(*elem);
                    row_indexes.push(index);
                }
            }
            inner.push((row_values, row_indexes));
        }
        SparseMatrix { inner }
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
