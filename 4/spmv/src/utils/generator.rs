use crate::Matrix;

#[derive(Debug, Clone)]
pub struct Generator {
    sparsity: f32,
}

impl Generator {
    pub fn with_sparsity(sparsity: f32) -> Self {
        assert!(sparsity < 1f32);
        assert!(0f32 <= sparsity);
        Self { sparsity }
    }

    // generates new matrix of `size` row and cols with the given sparsity
    pub fn matrix(&self, size: usize) -> Matrix<f32> {
        let mut matrix = Vec::with_capacity(size);
        for _ in 0..size {
            let mut row = vec![0f32; size];
            let nnz = ((size as f32) * self.sparsity).round() as usize + 1;
            for _ in 0..nnz {
                row[(rand::random::<usize>() % size) as usize] = rand::random::<f32>();
            }
            matrix.push(row);
        }
        matrix
    }

    // generates new matrix of `size` row and cols with the given sparsity suited for debugging
    // the functionality is the same as the matrix function. the only difference
    // is that the random numbers are small and decimal (i.e. 5.0) which makes the
    // resulting matrix suited for debugging
    pub fn debug_matrix(&self, size: usize) -> Matrix<f32> {
        let mut matrix = Vec::with_capacity(size);
        for _ in 0..size {
            let mut row = vec![0f32; size];
            let nnz = ((size as f32) * self.sparsity).round() as usize + 1;
            for _ in 0..nnz {
                row[(rand::random::<usize>() % size) as usize] = (rand::random::<u32>() % 5) as f32;
            }
            matrix.push(row);
        }
        matrix
    }

    // generates a new vector of `size` cells
    pub fn vector(&self, size: usize) -> Vec<f32> {
        let mut vector = Vec::with_capacity(size);
        for _ in 0..size {
            vector.push(rand::random::<f32>());
        }
        vector
    }

    // generates a new vector of `size` cells suited for debugging
    // the functionality is the same as the `vector` function. the onlty difference
    // is that the random numbers are 0s and 1s which makes the resulting vector
    // suited for debugging
    pub fn debug_vector(&self, size: usize) -> Vec<f32> {
        let mut vector = Vec::with_capacity(size);
        for _ in 0..size {
            vector.push((rand::random::<u32>() % 2) as f32);
        }
        vector
    }
}
