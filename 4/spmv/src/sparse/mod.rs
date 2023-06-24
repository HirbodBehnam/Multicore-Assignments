mod single_thread;
mod multi_thread;
mod simd;


pub use single_thread::SparseMatrix as SingleThreadOptimized;
pub use multi_thread::SparseMatrix as MultiThreadOptimized;
pub use simd::SparseMatrix as SimdOptimized;

#[derive(Debug, Clone)]
pub(crate) struct RowElement<T> {
    index: usize, // index in row
    value: T,   // the value of element
}