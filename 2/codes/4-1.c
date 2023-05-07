int A[N], temp[N];
#pragma omp parallel for schedule(static)
for (int i = 0; i < N - 1; i++)
    temp[i] = A[i + 1] + i;
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++)
    A[i] = temp[i];