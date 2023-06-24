use criterion::{criterion_group, criterion_main, Criterion};
use spmv::utils;
use spmv::dense::DenseMatrix as SpmvMatrix;
use spmv::SpMV;

fn sparsity_10_percent(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.1);
    let matrix = gen.matrix(10000);
    let vector = gen.vector(10000);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("sparsity 10", |b| b.iter(|| spmv.mul(&vector)));
}

fn sparsity_20_percent(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.2);
    let matrix = gen.matrix(10000);
    let vector = gen.vector(10000);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("sparsity 20", |b| b.iter(|| spmv.mul(&vector)));
}

fn sparsity_30_percent(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.3);
    let matrix = gen.matrix(10000);
    let vector = gen.vector(10000);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("sparsity 30", |b| b.iter(|| spmv.mul(&vector)));
}

fn size_100(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.3);
    let matrix = gen.matrix(100);
    let vector = gen.vector(100);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("size 100", |b| b.iter(|| spmv.mul(&vector)));
}

fn size_1000(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.3);
    let matrix = gen.matrix(1000);
    let vector = gen.vector(1000);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("size 1000", |b| b.iter(|| spmv.mul(&vector)));
}
fn size_10000(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.3);
    let matrix = gen.matrix(10000);
    let vector = gen.vector(10000);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("size 10000", |b| b.iter(|| spmv.mul(&vector)));
}
fn size_20000(c: &mut Criterion) {
    let gen = utils::Generator::with_sparsity(0.3);
    let matrix = gen.matrix(20000);
    let vector = gen.vector(20000);
    let spmv = SpmvMatrix::from(matrix);
    c.bench_function("size 20000", |b| b.iter(|| spmv.mul(&vector)));
}



criterion_group!(dense, size_100, size_1000, size_10000,size_20000, sparsity_10_percent, sparsity_20_percent, sparsity_30_percent);
criterion_main!(dense);
