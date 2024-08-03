#include "cuda_runtime.h"
#include "stdio.h"
#include "time.h"

__global__ void matmul(float *a, float *b, float *c, int M, int N, int P)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float tmp_sum = 0;
    if (row < M && col < P)
    {
        for (int k = 0; k < N; k++)
        {
            tmp_sum += a[(row * N + k)] * b[(k * P + col)];
        }
        c[row * P + col] = tmp_sum;
    }
}

int main()
{
    int M = 8192, N = 4096;
    size_t size_first = M * N * sizeof(float);
    int P = 8192;
    size_t size_second = N * P * sizeof(float);
    size_t size_third = M * P * sizeof(float);
    float *a;
    float *b;
    float *c;
    cudaError_t err;
    err = cudaMallocManaged(&a, size_first);
    if (err != cudaSuccess)
    {
        printf("CudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(&b, size_second);
    if (err != cudaSuccess)
    {
        printf("CudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(&c, size_third);
    if (err != cudaSuccess)
    {
        printf("CudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Initializing Arrays...\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = rand() % 101;
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < P; j++)
        {
            b[i * P + j] = rand() % 101;
        }
    }
    printf("Successfuly Initialized Arrays!\n");

    // MATRIX MULTIPLY WITH GPU
    dim3 threadSize(16, 16);
    dim3 numBlocks(P / 16, M / 16);
    printf("Number of Blocks: %d by %d\n", numBlocks.x, numBlocks.y);
    clock_t time_gpu = clock();
    matmul<<<numBlocks, threadSize>>>(a, b, c, M, N, P); // KERNEL
    cudaDeviceSynchronize();
    time_gpu = clock() - time_gpu;

    printf("Took in %.2fms!\n", (float)time_gpu);
    // FREE UP GPU MEMORY
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}