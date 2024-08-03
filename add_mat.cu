#include "stdio.h"
#include "cuda_runtime.h"

#define N 5

__global__ void matadd(float *a, float *b, float *c)
{

    int i = threadIdx.x;
    int j = threadIdx.y;
    int idx = i + (j * N);
    c[idx] = a[idx] + b[idx];
}

int main()
{
    const int size = N * N * sizeof(int);
    float cpu_a[N][N] = {
        {12, 43, 41, 53, 22},
        {34, 42, 23, 32, 32},
        {23, 98, 32, 43, 65},
        {54, 54, 34, 43, 53},
        {23, 83, 48, 43, 84}};
    float cpu_b[N][N] = {
        {12, 43, 41, 53, 22},
        {34, 42, 23, 32, 32},
        {23, 98, 32, 43, 65},
        {54, 54, 34, 43, 53},
        {23, 83, 48, 43, 84}};

    float *a;
    float *b;
    float *c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    cudaMemcpy(a, cpu_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, cpu_b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    matadd<<<1, threadsPerBlock>>>(a, b, c);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", *(c + j + (i * N)));
        }
        printf("\n");
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}