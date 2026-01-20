#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "ck/utility/dtype_vector.hpp"

#define HIP_CHECK(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// HIP kernel to copy data from source buffer to destination buffer
__global__ void copyKernel(const float* src, float* dst, long long n, const int data_per_thread) {
    long long idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < data_per_thread / 8; i++) {
        *((ck::float8_t*)(dst + idx)) = *((ck::float8_t*)(src + idx));
        idx += blockDim.x * gridDim.x * 8;
    }
}

int main() {
    const int device0_id = 30;
    const int device1_id = 31;
    printf("device0_id: %d, device1_id: %d\n", device0_id, device1_id);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = 65536;
    const int data_per_thread = 4;
    const long long N = threadsPerBlock * blocksPerGrid * data_per_thread;
    const long long bytes = N * sizeof(float);
    printf("N: %lld, bytes: %lld\n", N, bytes);
    // Allocate host memory
    std::vector<float> h_src(N);
    std::vector<float> h_dst(N);
    
    // Allocate device memory on device1
    HIP_CHECK(hipSetDevice(device1_id));
    float *d_src;
    HIP_CHECK(hipMalloc(&d_src, bytes));

    // Copy data from host to device1
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), bytes, hipMemcpyHostToDevice));
    float *d_dst;
    HIP_CHECK(hipSetDevice(device0_id));
    HIP_CHECK(hipMalloc(&d_dst, bytes));
    // Switch to device0 and enable peer access to device1
    if (device0_id != device1_id) {
        HIP_CHECK(hipDeviceEnablePeerAccess(device1_id, 0));
        int can_access = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access, device0_id, device1_id));
        std::cout << "Can device0 access device1: " << can_access << std::endl;
    }
    std::cout << "Launching kernel on device0 with " << blocksPerGrid << " blocks and " 
              << threadsPerBlock << " threads per block" << std::endl;

    hipEvent_t start_total, stop_total;
    hipEventCreate(&start_total);
    hipEventCreate(&stop_total);
    hipEventRecord(start_total, 0);
    for (size_t i = 0; i< 1000; i++){
        hipLaunchKernelGGL(copyKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 
                    0, 0, d_src, d_dst, N, data_per_thread);
    }
    hipEventRecord(stop_kernel, 0);

    hipEventSynchronize(stop_total);
    float total_time_ms = 0.0f;
    hipEventElapsedTime(&total_time_ms, start_total, stop_total);
    total_time_ms = total_time_ms / 1000;
    printf("核函数执行耗时: %.3f ms (%.3f μs)\n", total_time_ms, total_time_ms * 1000);
    HIP_CHECK(hipGetLastError());
    
    // Wait for kernel to finish
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, bytes, hipMemcpyDeviceToHost));
    
    // Verify results
    bool success = true;
    
    if (success) {
        std::cout << "Copy kernel executed successfully!" << std::endl;
        std::cout << "Verified " << N << " elements" << std::endl;
    }

    HIP_CHECK(hipFree(d_src));
    if (device0_id != device1_id) {
        HIP_CHECK(hipFree(d_dst));
    }
    hipEventDestroy(start_total);
    hipEventDestroy(stop_total);
    return 0;
}
 