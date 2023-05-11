#include<omp.h> 
#include<stdio.h> 
#include<stdio.h> 

int thread_work(int i) { 
    int sum = 0; 
    for (int j = 0; j < i; j++) { 
        sum += i;
    }
    return sum; 
}

int main() { 
    int A[32];

    double start, end; 
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static)       // set different schedulings here
    for (int i = 0; i < 32; i++) { 
        A[i] = thread_work((rand() % 32) * 10000000l);
    }
    end = omp_get_wtime();

    printf("Time: %f (ms)\n", (end - start) * 1000);
}
