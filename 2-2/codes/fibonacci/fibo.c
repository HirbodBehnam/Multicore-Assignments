#include <stdio.h>
#include <omp.h>

#define N 20

int fibonacci(int n)
{
    int x, y;

    if (n < 2) 
        return 1;

    #pragma omp parallel sections
    {
        #pragma omp section
        x = fibonacci(n-1);

        #pragma omp section
        y = fibonacci(n-2);
    }

    return x + y;
}

int main ( void )  
{
  int i, fib[N];

  #pragma omp parallel for schedule(dynamic)
  for(i = 0 ; i < N; i++)
      fib[i] = fibonacci(i);

  for(i = 0 ; i < N; i++)
      printf("%d ", fib[i]);
  printf("\n");

  return 0;
}


