#include <hip/hip_runtime.h>

__global__ void transpose_kernel(const float* A, float* B, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) B[col * M + row] = A[row * N + col];
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    transpose_kernel<<<grid, block>>>(A, B, M, N);
    hipDeviceSynchronize();
}
