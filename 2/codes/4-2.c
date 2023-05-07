int A[M], indices[N];
#pragma omp for schedule(static)
for (int i = 0; i < N; i++) {
    int result = exp(i) ˆ 2;
    int index = indices[i];
    #pragma omp atomic
    A[index] += result;
}