#include "stdio.h"
#include "stdlib.h"
#include "cuda_runtime.h"

#define ARRAY_SIZE 100

__global__ void addVectors(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        c[i] = (a[i] + b[i]) * (a[i] + b[i]);
    }
}

int main()
{
    int *a;
    int *b;
    int *c;
    cudaMallocManaged(&a, ARRAY_SIZE * sizeof(int));
    cudaMallocManaged(&b, ARRAY_SIZE * sizeof(int));
    cudaMallocManaged(&c, ARRAY_SIZE * sizeof(int));

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = i;
        b[i] = i + 1;
    }
    addVectors<<<1, ARRAY_SIZE>>>(a, b, c, ARRAY_SIZE);

    cudaDeviceSynchronize();

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%d\n", c[i]);
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}