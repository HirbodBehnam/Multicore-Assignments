int B[N];
#pragma omp for schedule(static)
for (int i = 0; i < N; i++)
    B[i] = 0;