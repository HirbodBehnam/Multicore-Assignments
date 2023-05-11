#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE (1000 * 1000)
#define TESTS 10

// Code from https://www.programiz.com/dsa/merge-sort

// Merge two subarrays L and M into arr
void merge(int arr[], int p, int q, int r)
{
    // Create L ← A[p..q] and M ← A[q+1..r]
    int n1 = q - p + 1;
    int n2 = r - q;

    int L[n1], M[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[p + i];
    for (int j = 0; j < n2; j++)
        M[j] = arr[q + 1 + j];

    // Maintain current index of sub-arrays and main array
    int i, j, k;
    i = 0;
    j = 0;
    k = p;

    // Until we reach either end of either L or M, pick larger among
    // elements L and M and place them in the correct position at A[p..r]
    while (i < n1 && j < n2)
    {
        if (L[i] <= M[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = M[j];
            j++;
        }
        k++;
    }

    // When we run out of elements in either L or M,
    // pick up the remaining elements and put in A[p..r]
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = M[j];
        j++;
        k++;
    }
}

// Divide the array into two subarrays, sort them and merge them
void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        // m is the point where the array is divided into two subarrays
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Merge the sorted subarrays
        merge(arr, l, m, r);
    }
}

int main()
{
    srand(1); // make deterministic array
    int main_array[ARRAY_SIZE];
    double elapsed_sum = 0;
    for (int test_number = 0; test_number < TESTS; test_number++) {
        for (int i = 0; i < ARRAY_SIZE; i++)
            main_array[i] = rand();
        double start = omp_get_wtime();
        mergeSort(main_array, 0, ARRAY_SIZE - 1);
        elapsed_sum += omp_get_wtime() - start;
    }
    printf("Avg %lf (ms)\n", elapsed_sum / TESTS);
}