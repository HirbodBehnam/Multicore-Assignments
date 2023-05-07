#include <omp.h>

#define N 10

void process_a(int *a) {
    int temp[N];
    omp_set_num_threads(8); // my cpu cores
    #pragma omp for schedule(static)
    for (int i = 1; i < N - 1; i++)
        temp[i] = a[i - 1] + a[i] - a[i + 1];
    #pragma omp for schedule(static)
    for (int i = 1; i < N - 1; i++)
        a[i] = temp[i];
}

void process_b(int *b) {
    omp_set_num_threads(8); // my cpu cores
    #pragma omp for schedule(static)
    for (int i = 0; i < N; i++)
        b[i] = 2^b[i] + 2*b[i];
}

int main() {
    int B[N], A[N];

    omp_set_num_threads(2);
    omp_set_nested(1);
    #pragma omp parallel sections
    {
        #pragma omp section
        process_a(A);
        #pragma omp section
        process_b(B);
    }
}