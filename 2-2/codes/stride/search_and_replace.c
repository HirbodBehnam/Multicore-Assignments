#include <omp.h>
#include <stdio.h>

#define MAX_SIZE 1000 * 1000 * 1000
#define TEST_COUNT 10
#define NUM_THREADS 8
typedef void(bench_func)(int *, int, int, int);

void replace_continues(int *arr, int n, int search, int replace) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    if (arr[i] == search)
      arr[i] = replace;
}

void replace_strided(int *arr, int n, int search, int replace) {
#pragma omp parallel num_threads(NUM_THREADS)
  {
    int thread_id = omp_get_thread_num();
#pragma omp stride(NUM_THREADS)
    for (int i = thread_id; i < n; i += NUM_THREADS)
      if (arr[i] == search)
        arr[i] = replace;
  }
}

int count_zeros(int *arr, int n) {
  int count = 0;
  for (int i = 0; i < n; i++)
    count += arr[i] == 0;
  return count;
}

void bench_function(bench_func func, int *arr, int n, int search, int replace) {
  double start, end, avg = 0;
  volatile int num_zeros = 0;
  for (int i = 0; i < TEST_COUNT; i++) {
    start = omp_get_wtime();
    func(arr, MAX_SIZE, 0, 1);
    end = omp_get_wtime();

    num_zeros = count_zeros(arr, MAX_SIZE);
    avg += (end - start);
  }
  avg /= TEST_COUNT;
  printf("function took %f in average\n", avg * 1000);
}

int arr[MAX_SIZE];
int main() {
  bench_function(replace_continues, arr, MAX_SIZE, 0, 1);
  bench_function(replace_strided, arr, MAX_SIZE, 1, 0);
}
