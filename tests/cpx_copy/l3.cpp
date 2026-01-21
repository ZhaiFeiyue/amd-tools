#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

// 定义 L3 缓存大小（根据你的 GPU 调整，比如 MI250X 是 32MB/卡）
#define L3_CACHE_SIZE (32 * 1024 * 1024)
// 数据类型（可换 double/int 等）
using DataType = float;

// 核心测试函数：循环读写 L3 缓存数据
__global__ void l3_bandwidth_kernel(DataType* data, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= L3_CACHE_SIZE / sizeof(DataType)) return;

    // 循环读写，确保数据在 L3 缓存中（不溢出到全局内存）
    for (int i = 0; i < iterations; i++) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    // 1. 获取 GPU 设备信息
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No AMD GPU found!" << std::endl;
        return -1;
    }
    hipSetDevice(0);

    // 2. 分配设备内存（刚好填满 L3 缓存）
    DataType* d_data;
    size_t data_size = L3_CACHE_SIZE;
    hipMalloc(&d_data, data_size);
    hipMemset(d_data, 0, data_size); // 初始化数据

    // 3. 配置核函数参数
    int block_size = 256;
    int grid_size = (data_size / sizeof(DataType) + block_size - 1) / block_size;
    int iterations = 10000; // 循环次数（保证测试时间足够长）

    // 4. 预热（避免首次运行的初始化开销）
    l3_bandwidth_kernel<<<grid_size, block_size>>>(d_data, 1);
    hipDeviceSynchronize();

    // 5. 计时运行核函数
    auto start = std::chrono::high_resolution_clock::now();
    l3_bandwidth_kernel<<<grid_size, block_size>>>(d_data, iterations);
    hipDeviceSynchronize(); // 等待核函数完成
    auto end = std::chrono::high_resolution_clock::now();

    // 6. 计算耗时和带宽
    std::chrono::duration<double> elapsed = end - start;
    double total_bytes = (double)data_size * iterations * 2; // 读+写，2次访问
    double bandwidth_gb_s = (total_bytes / 1e9) / elapsed.count();

    // 输出结果
    std::cout << "=== L3 Cache Bandwidth Test ===" << std::endl;
    std::cout << "L3 Cache Size: " << L3_CACHE_SIZE / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Elapsed Time: " << elapsed.count() << " s" << std::endl;
    std::cout << "L3 Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

    // 7. 释放内存
    hipFree(d_data);
    return 0;
}