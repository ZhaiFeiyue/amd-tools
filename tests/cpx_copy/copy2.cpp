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
    const int device0_id = 31;
    const int device1_id = 31;
    printf("device0_id: %d, device1_id: %d\n", device0_id, device1_id);

    const int threadsPerBlock = 64;
    const int blocksPerGrid = 160;
    const int data_per_thread = 8 * 100;
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

    hipLaunchKernelGGL(copyKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 
                       0, 0, d_src, d_dst, N, data_per_thread);
    // std::cout << "Sleeping ..." << std::endl;
    // sleep(10000);
    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    
    // Wait for kernel to finish
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, bytes, hipMemcpyDeviceToHost));
    
    // Verify results
    bool success = true;
    // for (long long i = 0; i < N; i++) {
    //     if (h_dst[i] != h_src[i]) {
    //         std::cerr << "Verification failed at index " << i << std::endl;
    //         success = false;
    //         break;
    //     }
    // }
    
    if (success) {
        std::cout << "Copy kernel executed successfully!" << std::endl;
        std::cout << "Verified " << N << " elements" << std::endl;
    }
    // std::cout << "Sleeping ..." << std::endl;
    // sleep(10);
    // Free device memory
    HIP_CHECK(hipFree(d_src));
    if (device0_id != device1_id) {
        HIP_CHECK(hipFree(d_dst));
    }

    return 0;
}
 