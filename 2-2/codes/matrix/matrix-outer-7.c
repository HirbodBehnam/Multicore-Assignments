#include<omp.h> 
#include<stdio.h> 
#include<stdlib.h> 

#define N 800

int main() { 
    int A[N][N], B[N][N], C[N][N];
    // initialize matrices with numbers in the range [-5, 5]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { 
            A[i][j] = rand() % 11 - 5; 
            B[i][j] = rand() % 11 - 5; 
        }
    }

    omp_set_num_threads(7);
    double start, end; 
    start = omp_get_wtime();    // start time measurement

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) { 
            int sum = 0;
            for (int k = 0; k < N; k++) { 
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum; 
        }
    }

    end = omp_get_wtime();      // end time measurement

    printf("Time: %f (ms)\n", (end - start) * 1000);
}
