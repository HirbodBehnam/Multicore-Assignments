use crate::{Matrix, SpMV};

use super::RowElement;

#[repr(align(32))]
#[derive(Debug, Default, Clone)]
struct YMMBuffer([f32; 8]);

#[repr(align(32))]
#[derive(Debug, Default, Clone)]
struct IndexBuffer([i32; 8]);

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    values: Vec<Vec<YMMBuffer>>,
    indexes: Vec<Vec<IndexBuffer>>,
    real_rows: usize,
}

impl SparseMatrix {
    // From https://sci-hub.ru/10.1145/3225058.3225100
    unsafe fn mul_avx2(&self, vector: &Vec<f32>) -> Vec<f32> {
        use std::arch::x86_64::*;
        assert_eq!(self.values.len(), self.indexes.len());
        let mut result: Vec<f32> = Vec::with_capacity(self.values.len() * 8);
        let mut vec_buffer = YMMBuffer::default();
        for (values, indexes) in self.values.iter().zip(&self.indexes) {
            assert_eq!(values.len(), indexes.len());
            // for each 4 columns...
            let mut accumulator = _mm256_setzero_ps();
            for (value, index) in values.iter().zip(indexes.iter()) {
                // Load values into YMM registers
                let current_matrix = _mm256_load_ps(value.0.as_ptr());
                for (buf_index, vec_index) in index.0.iter().enumerate() {
                    vec_buffer.0[buf_index] = vector[*vec_index as usize];
                }
                let current_vector = _mm256_load_ps(vec_buffer.0.as_ptr());
                // Fuse mult add them
                accumulator = _mm256_fmadd_ps(current_matrix, current_vector, accumulator);
            }
            _mm256_store_ps(vec_buffer.0.as_mut_ptr(), accumulator);
            result.extend_from_slice(&vec_buffer.0);
        }
        result.resize(self.real_rows, 0f32);
        return result;
    }

    fn to_semi_sparse(matrix: Matrix<f32>) -> Vec<Vec<RowElement<f32>>> {
        matrix
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
        .collect::<Vec<Vec<RowElement<f32>>>>()
    }
}

impl SpMV for SparseMatrix {
    fn mul(&self, vector: &Vec<f32>) -> Vec<f32> {
        unsafe { self.mul_avx2(vector) }
    }
}

impl From<Matrix<f32>> for SparseMatrix {
    fn from(matrix: Matrix<f32>) -> Self {
        let rows = matrix.len();
        let semi_sparse = SparseMatrix::to_semi_sparse(matrix);
        let mut values_inner = Vec::<Vec<YMMBuffer>>::with_capacity(semi_sparse.len() / 8);
        let mut indexes_inner = Vec::<Vec<IndexBuffer>>::with_capacity(semi_sparse.len() / 8);
        for rows_batch in semi_sparse.chunks(8) {
            let mut iterators: Vec<std::slice::Iter<'_, RowElement<f32>>> = rows_batch.iter().map(|row| row.iter()).collect();
            let mut current_values = Vec::<YMMBuffer>::new();
            let mut current_indexes = Vec::<IndexBuffer>::new();
            loop {
                let mut current_value = YMMBuffer::default();
                let mut current_index = IndexBuffer::default();
                let mut saw_non_none = false;
                for (index, row_iterator) in iterators.iter_mut().enumerate() {
                    if let Some(elem) = row_iterator.next() {
                        saw_non_none = true;
                        current_value.0[index] = elem.value;
                        current_index.0[index] = elem.index as i32;
                    }
                }
                if !saw_non_none {
                    break;
                }
                current_values.push(current_value);
                current_indexes.push(current_index);
            }
            values_inner.push(current_values);
            indexes_inner.push(current_indexes);
        }
        SparseMatrix { values: values_inner, indexes: indexes_inner, real_rows: rows }
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
    fn convert() {
        let matrix: Matrix<f32> = vec![
            vec![0.0, 1.0, 0.0],
            vec![2.0, 0.0, 3.0],
            vec![0.0, 0.0, 4.0],
        ];
        let spmv = SpmvMatrix::from(matrix);
        assert_eq!(spmv.indexes.len(), 1);
        assert_eq!(spmv.values.len(), 1);
        assert_eq!(spmv.indexes[0].len(), 2);
        assert_eq!(spmv.values[0].len(), 2);
        assert_eq!(&spmv.indexes[0][0].0, &[1, 0, 2, 0, 0, 0, 0, 0]);
        assert_eq!(&spmv.values[0][0].0, &[1f32, 2f32, 4f32, 0f32, 0f32, 0f32, 0f32, 0f32]);
        assert_eq!(&spmv.indexes[0][1].0, &[0, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&spmv.values[0][1].0, &[0f32, 3f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]);
    }

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
