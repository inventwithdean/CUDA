#include "cuda_runtime.h"
#include "stdio.h"

__global__ void mat_add(int *a, int *b, int *c, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i * n + j;
    c[idx] = a[idx] + b[idx];
}

int main()
{
    int *a;
    int *b;
    int *c;
    // SIZE OF ARRAY: 4, 3
    int M = 50304, N = 50304;
    int size = M * N * sizeof(int);
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    srand(1337);
    for (int i = 0; i < M * N; i++)
    {
        a[i] = rand() % 101;
        b[i] = rand() % 101;
    }

    dim3 threadPerBlock(16, 16);
    dim3 numBlocks(M / threadPerBlock.x, N / threadPerBlock.y);
    mat_add<<<numBlocks, threadPerBlock>>>(a, b, c, M, N);
    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("Done!\n");

    return 0;
}