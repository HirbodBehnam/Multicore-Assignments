#include<omp.h> 
#include<stdio.h> 

int thread_work(int i) { 
    return i; 
}

int main() { 
    int A[256];

    double start, end; 
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)       // set different scheduling ways here
    for (int i = 0; i < 32; i++) { 
        A[i] = thread_work(i);
    }
    end = omp_get_wtime();

    printf("Time: %f (ms)\n", (end - start) * 1000);
}
