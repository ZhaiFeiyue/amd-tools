#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
#include <algorithm>

// 适配AMD GPU的核心配置
#define WAVEFRONT_SIZE 64        // AMD GPU Wavefront大小（固定64）
#define BLOCK_SIZE     256       // 线程块大小（必须是64的倍数）
#define VECTOR_SIZE    4         // 向量加载/存储大小（int4/float4）
#define ALIGNMENT      16        // 内存对齐字节数（匹配VECTOR_SIZE*sizeof(int)）

// 高性能合并访存Memcpy Kernel
template <typename T, int VecSize>
__global__ void coalesced_memcpy_kernel(T* __restrict__ dst, 
                                       const T* __restrict__ src, 
                                       size_t num_elements) {
    const size_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vec_idx = global_thread_id;
    const size_t element_idx = vec_idx * VecSize;

    using VecType = typename std::conditional<VecSize == 4, int4, 
                    typename std::conditional<VecSize == 2, int2, T>::type>::type;

    VecType* dst_vec = reinterpret_cast<VecType*>(dst);
    const VecType* src_vec = reinterpret_cast<const VecType*>(src);

    const size_t total_vecs = num_elements / VecSize;

    if (vec_idx < total_vecs) {
        dst_vec[vec_idx] = src_vec[vec_idx];
    }

    const size_t remaining = num_elements % VecSize;
    if (remaining > 0 && global_thread_id == 0) {
        const size_t start = total_vecs * VecSize;
        for (size_t i = start; i < num_elements; i++) {
            dst[i] = src[i];
        }
    }
}

// 批量处理版本（针对超大内存拷贝）
template <typename T, int VecSize>
__global__ void batched_coalesced_memcpy_kernel(T* __restrict__ dst, 
                                               const T* __restrict__ src, 
                                               size_t num_elements) {
    const size_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid_size = gridDim.x * blockDim.x;

    using VecType = typename std::conditional<VecSize == 4, int4, 
                    typename std::conditional<VecSize == 2, int2, T>::type>::type;

    VecType* dst_vec = reinterpret_cast<VecType*>(dst);
    const VecType* src_vec = reinterpret_cast<const VecType*>(src);

    const size_t total_vecs = num_elements / VecSize;

    for (size_t vec_idx = global_thread_id; vec_idx < total_vecs; vec_idx += grid_size) {
        dst_vec[vec_idx] = src_vec[vec_idx];
    }

    const size_t remaining = num_elements % VecSize;
    if (remaining > 0 && global_thread_id == 0) {
        const size_t start = total_vecs * VecSize;
        for (size_t i = start; i < num_elements; i++) {
            dst[i] = src[i];
        }
    }
}

// 主机端封装函数（自动选择合适的Kernel）
template <typename T>
hipError_t hip_coalesced_memcpy(T* dst, const T* src, size_t num_elements, hipStream_t stream = 0) {
    if (reinterpret_cast<uintptr_t>(dst) % ALIGNMENT != 0 || 
        reinterpret_cast<uintptr_t>(src) % ALIGNMENT != 0) {
        fprintf(stderr, "Warning: Data not aligned to %d bytes! Performance will drop!\n", ALIGNMENT);
    }

    const dim3 block_dim(BLOCK_SIZE);
    size_t grid_dim_x = min((num_elements / (BLOCK_SIZE * VECTOR_SIZE)) + 1, 65535UL);
    const dim3 grid_dim(grid_dim_x);

    hipError_t err;
    if (num_elements <= BLOCK_SIZE * grid_dim_x * VECTOR_SIZE) {
        coalesced_memcpy_kernel<T, VECTOR_SIZE><<<grid_dim, block_dim, 0, stream>>>(dst, src, num_elements);
    } else {
        batched_coalesced_memcpy_kernel<T, VECTOR_SIZE><<<grid_dim, block_dim, 0, stream>>>(dst, src, num_elements);
    }
    err = hipGetLastError();

    return err;
}

// 性能测试函数（带计时和带宽计算）
template <typename T>
void benchmark_memcpy(size_t num_elements, int num_runs = 10, int warmup_runs = 3) {
    // 1. 计算数据大小
    const size_t element_size = sizeof(T);
    const size_t data_size_bytes = num_elements * element_size;
    const double data_size_gb = data_size_bytes / (1024.0 * 1024.0 * 1024.0);

    // 2. 分配对齐的主机内存
    T* h_src = (T*)aligned_alloc(ALIGNMENT, data_size_bytes);
    T* h_dst = (T*)aligned_alloc(ALIGNMENT, data_size_bytes);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        return;
    }

    // 3. 初始化数据
    for (size_t i = 0; i < num_elements; i++) {
        h_src[i] = static_cast<T>(i % 1000); // 填充随机范围的测试数据
    }


    const int device0_id = 0;
    const int device1_id = 4;

    // 4. 分配设备内存
    T* d_src = nullptr;
    HIP_CHECK(hipSetDevice(device1_id));
    hipMalloc(&d_dst, data_size_bytes);
    


    T* d_dst = nullptr;
    HIP_CHECK(hipSetDevice(device0_id));
    hipMalloc(&d_src, data_size_bytes);
    // 5. 拷贝数据到设备（预加载）
    hipMemcpy(d_src, h_src, data_size_bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    if (device0_id != device1_id) {
        HIP_CHECK(hipDeviceEnablePeerAccess(device1_id, 0));
        int can_access = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access, device0_id, device1_id));
        std::cout << "Can device0 access device1: " << can_access << std::endl;
    }

    // 6. 创建计时事件
    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);

    // 存储每轮耗时（毫秒）
    std::vector<float> run_times;

    // 7. 热身运行（消除初始化开销）
    for (int i = 0; i < warmup_runs; i++) {
        hip_coalesced_memcpy(d_dst, d_src, num_elements);
    }
    hipDeviceSynchronize();

    // 8. 正式测试
    for (int i = 0; i < num_runs; i++) {
        // 重置目标内存（避免缓存影响）
        hipMemset(d_dst, 0, data_size_bytes);
        hipDeviceSynchronize();

        // 开始计时
        hipEventRecord(start_event, 0);

        // 执行memcpy kernel
        hip_coalesced_memcpy(d_dst, d_src, num_elements);

        // 停止计时
        hipEventRecord(stop_event, 0);
        hipEventSynchronize(stop_event);

        // 获取耗时（毫秒）
        float elapsed_ms = 0.0f;
        hipEventElapsedTime(&elapsed_ms, start_event, stop_event);
        run_times.push_back(elapsed_ms);

        // 验证数据正确性（仅第一轮验证，避免影响性能）
        if (i == 0) {
            hipMemcpy(h_dst, d_dst, data_size_bytes, hipMemcpyDeviceToHost);
            bool success = true;
            for (size_t j = 0; j < num_elements; j++) {
                if (h_dst[j] != h_src[j]) {
                    fprintf(stderr, "Error: Mismatch at index %zu: %d vs %d\n", j, h_dst[j], h_src[j]);
                    success = false;
                    break;
                }
            }
            if (!success) {
                fprintf(stderr, "Data validation failed!\n");
                goto cleanup;
            }
        }
    }

    // 9. 计算性能指标
    // 移除最大值和最小值，计算平均耗时
    std::sort(run_times.begin(), run_times.end());
    float avg_time_ms = 0.0f;
    for (int i = 1; i < num_runs - 1; i++) { // 去掉首尾
        avg_time_ms += run_times[i];
    }
    avg_time_ms /= (num_runs - 2);

    // 转换为秒
    const double avg_time_s = avg_time_ms / 1000.0;
    
    // 计算带宽：总数据量(GB) × 2（读+写） / 耗时(s)
    const double bandwidth_gbs = (data_size_gb * 2) / avg_time_s;
    
    // 计算吞吐量：数据量(GB) / 耗时(s)
    const double throughput_gbs = data_size_gb / avg_time_s;

    // 10. 输出结果
    printf("=======================================\n");
    printf("Memcpy Benchmark Results\n");
    printf("=======================================\n");
    printf("Data size        : %.2f GB\n", data_size_gb);
    printf("Number of runs   : %d (warmup: %d)\n", num_runs, warmup_runs);
    printf("Average time     : %.4f ms\n", avg_time_ms);
    printf("Throughput       : %.2f GB/s (read)\n", throughput_gbs);
    printf("Memory bandwidth : %.2f GB/s (read+write)\n", bandwidth_gbs);
    printf("=======================================\n");
    printf("Note: Typical AMD RDNA3 GPU peak bandwidth ~ 500-600 GB/s\n");
    printf("      Typical AMD CDNA2 GPU peak bandwidth ~ 1.2-1.7 TB/s\n");

cleanup:
    // 11. 释放资源
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
    free(h_src);
    free(h_dst);
    hipFree(d_src);
    hipFree(d_dst);
}

// 获取GPU设备信息（可选）
void print_gpu_info() {
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No ROCm compatible GPU found!\n");
        return;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("=======================================\n");
    printf("GPU Device Information\n");
    printf("=======================================\n");
    printf("Device name      : %s\n", prop.name);
    printf("GPU arch         : gfx%s\n", prop.gcnArchName);
    printf("Global memory    : %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Clock rate       : %.2f GHz\n", prop.clockRate / 1000000.0);
    printf("Memory clock     : %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Memory bus width : %d bits\n", prop.memoryBusWidth);
    // 计算理论峰值带宽 = 内存时钟(MHz) × 位宽(bits) / 8 / 1024
    double peak_bandwidth = (prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1024.0 * 1024.0);
    printf("Theoretical peak bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("=======================================\n\n");
}

int main() {
    // 打印GPU信息
    print_gpu_info();

    // 测试不同数据量（可根据GPU内存调整）
    const size_t small_data = 16 * 1024 * 1024;    // 16M int (64MB)
    const size_t medium_data = 256 * 1024 * 1024;  // 256M int (1GB)
    const size_t large_data = 1024 * 1024 * 1024;  // 1024M int (4GB)

    // 运行基准测试
    printf("\n--- Testing Small Data (64MB) ---\n");
    benchmark_memcpy<int>(small_data, 10, 3);

    printf("\n--- Testing Medium Data (1GB) ---\n");
    benchmark_memcpy<int>(medium_data, 10, 3);

    printf("\n--- Testing Large Data (4GB) ---\n");
    benchmark_memcpy<int>(large_data, 10, 3);

    return 0;
}