# Implementation of Sparse Matrix
## Data Storage
For each row of matrix, we just store the index + value of each column that has an non zero value. To convert a dense matrix to a sparse one, we just iterate over each row and remove the ones which does have zero value. This is done with `filter_map` filter in Rust.
## Single Thread
In single thread, we just iterate over each row and multiply the values of each row respectfully with the vector elements, element-vise and accumulate them.

## Multi Thread
In multi-thread mode, we use `thread::scope` to create a scope to avoid using `Arc`. Then, based on `THREAD_NUMBER` value, we create threads in the scope and assign the rows of matrix to it. Each thread computes the result of the passed rows to it.

### Optimizations
* False sharing avoidance: To avoid false sharing, I used channels to do message passing instead of memory sharing to avoid accessing mutably to result array.
* No mutex: Because we use message passing, we don't use mutex.

## SIMD
### AVX2
For AVX2 implementation, at first for each row, we chunk the available indexes to 8 float values so we can store them in YMM registers. For each chunk, we create two YMM registers; One for matrix row elements and other for corresponding vector elements. Next we multiply these two YMM registers element wise. After that we shall sum each element in array. To do so, we use horizontal add instructions.

Have a look at this explanation:
```rust
let start = ...
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
let mut sum = _mm256_hadd_ps(start, start);
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
// sum_high elements are all x[4] + x[5] + x[6] + x[7]
let sum_high = _mm256_extractf128_ps(sum, 1);
// This will add element wise. So the result is in the first index of result
let final_sum = _mm_add_ps(sum_high, _mm256_castps256_ps128(sum));
let result = _mm_cvtss_f32(final_sum);
```
After that, we just accumulate these values to find the result. However, because float is inaccurate, this will create some problems on big and small numbers because order of operations matter in floating point numbers. Because of this, the big test fails.

### AVX512
My CPU does not support AVX512 but from what I read we can do two things:
1. Just use bigger registers to add 16 float values together. This speed things up a little less than 2 times.
2. Use [VGATHERDPS](https://www.felixcloutier.com/x86/vgatherdps:vgatherdpd) instruction to gather all vector elements without looping over them. This combined with 1 will probably speed things up more than 2 times faster.

## Benchmarks
My CPU is Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz. First gen to support AVX2!
### Dense
```
size 100                time:   [5.2145 µs 5.2178 µs 5.2215 µs]
size 1000               time:   [672.80 µs 673.63 µs 674.75 µs]
size 10000              time:   [75.366 ms 75.406 ms 75.449 ms]
size 20000              time:   [305.73 ms 305.97 ms 306.26 ms]
sparsity 10             time:   [75.932 ms 75.999 ms 76.068 ms]
sparsity 20             time:   [75.855 ms 75.917 ms 75.983 ms]
sparsity 30             time:   [75.334 ms 75.379 ms 75.428 ms]
```
### Sparse single thread
```
size 100                time:   [1.9175 µs 1.9194 µs 1.9218 µs]
size 1000               time:   [190.33 µs 190.57 µs 190.84 µs]
size 10000              time:   [27.427 ms 27.465 ms 27.503 ms]
size 20000              time:   [109.45 ms 109.58 ms 109.74 ms]
sparsity 10             time:   [10.902 ms 10.920 ms 10.939 ms]
sparsity 20             time:   [19.442 ms 19.470 ms 19.499 ms]
sparsity 30             time:   [27.077 ms 27.126 ms 27.190 ms]
```
### SIMD
```
size 100                time:   [5.5704 µs 5.5789 µs 5.5899 µs]
size 1000               time:   [480.99 µs 481.58 µs 482.28 µs]
size 10000              time:   [48.910 ms 48.971 ms 49.053 ms]
size 20000              time:   [194.12 ms 194.50 ms 194.99 ms]
sparsity 10             time:   [18.676 ms 18.707 ms 18.742 ms]
sparsity 20             time:   [34.790 ms 34.869 ms 34.959 ms]
sparsity 30             time:   [48.887 ms 48.930 ms 48.981 ms]
```
SIMD performs worse because I used heavy operations such as vector initialization and such. This could probably be improved if I could store the data in a single stack allocated array instead of heap allocated vector.
### Multithread
#### Thread Count = 2
```
size 100                time:   [30.326 µs 30.454 µs 30.581 µs]
size 1000               time:   [351.44 µs 352.96 µs 354.70 µs]
size 10000              time:   [23.854 ms 23.934 ms 24.016 ms]
size 20000              time:   [78.106 ms 78.381 ms 78.673 ms]
sparsity 10             time:   [14.578 ms 14.673 ms 14.771 ms]
sparsity 20             time:   [20.772 ms 20.856 ms 20.944 ms]
sparsity 30             time:   [23.899 ms 24.047 ms 24.218 ms]
```
#### Thread Count = 4
```
size 100                time:   [47.589 µs 47.843 µs 48.112 µs]
size 1000               time:   [247.23 µs 248.09 µs 248.97 µs]
size 10000              time:   [17.079 ms 17.122 ms 17.168 ms]
size 20000              time:   [65.647 ms 65.849 ms 66.081 ms]
sparsity 10             time:   [8.7933 ms 8.8305 ms 8.8692 ms]
sparsity 20             time:   [12.761 ms 12.804 ms 12.851 ms]
sparsity 30             time:   [17.002 ms 17.028 ms 17.059 ms]
```
#### Thread Count = 6
```
size 100                time:   [70.190 µs 71.861 µs 73.962 µs]
size 1000               time:   [216.81 µs 222.86 µs 229.05 µs]
size 10000              time:   [17.160 ms 17.240 ms 17.340 ms]
size 20000              time:   [66.532 ms 66.587 ms 66.649 ms]
sparsity 10             time:   [7.9562 ms 7.9717 ms 7.9876 ms]
sparsity 20             time:   [12.326 ms 12.360 ms 12.413 ms]
sparsity 30             time:   [17.021 ms 17.033 ms 17.047 ms]
```
#### Thread Count = 8
```
size 100                time:   [84.016 µs 84.375 µs 84.729 µs]
size 1000               time:   [196.97 µs 197.78 µs 198.57 µs]
size 10000              time:   [17.099 ms 17.130 ms 17.172 ms]
size 20000              time:   [66.639 ms 66.787 ms 66.971 ms]
sparsity 10             time:   [7.1850 ms 7.2145 ms 7.2441 ms]
sparsity 20             time:   [12.180 ms 12.201 ms 12.223 ms]
sparsity 30             time:   [17.043 ms 17.070 ms 17.102 ms]
```