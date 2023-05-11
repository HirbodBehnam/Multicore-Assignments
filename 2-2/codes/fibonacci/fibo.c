#include <stdio.h>
#include <omp.h>

#define N 10

int fibonacci(int n)
{
    int x, y;

    if (n < 2) 
        return 1;
    x = fibonacci(n-1);
    y = fibonacci(n-2);

    return x + y;
}

int main ( void )  
{
  int i, fib[N];

  // TODO: parallelize 
  for(i = 0 ; i < N; i++)
      fib[i] = fibonacci(i);

  for(i = 0 ; i < N; i++)
      printf("%d ", fib[i]);
  printf("\n");

  return 0;
}


