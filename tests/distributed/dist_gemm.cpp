#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

// GEMM配置（可根据GPU显存调整）
#define M 2048    // 矩阵A的行
#define N 2048    // 矩阵B的列/矩阵C的列
#define K 2048    // 矩阵A的列/矩阵B的行
#define BLOCK_SIZE 256  // 线程块大小（64的倍数，适配AMD Wavefront）

// 优化的GEMM Kernel（列主序，float类型）
__global__ void gemm_kernel(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C, 
                            int M, int N, int K) {
    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 每个线程块负责的子矩阵
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // 共享内存缓存（减少全局内存访问）
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // 分块计算GEMM
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // 加载A的子块到共享内存
        if (row < M && (t * BLOCK_SIZE + tx) < K) {
            sA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // 加载B的子块到共享内存
        if (col < N && (t * BLOCK_SIZE + ty) < K) {
            sB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 乘加计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // 写入结果到C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 单个GPU的GEMM测试函数（返回TOPS）
float run_gemm_on_gpu(int dev_id) {
    // 1. 切换到目标GPU
    hipSetDevice(dev_id);

    // 2. 计算矩阵大小
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 3. 分配设备内存
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    // 4. 初始化数据（主机端生成，拷贝到设备）
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice);

    // 5. 配置Kernel参数
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 6. 创建流和计时事件
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // 7. 热身（消除初始化开销）
    for (int i = 0; i < 3; i++) {
        gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    }
    hipStreamSynchronize(stream);

    // 8. 正式计时
    hipEventRecord(start, stream);
    gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    hipEventRecord(stop, stream);
    hipStreamSynchronize(stream);

    // 9. 计算耗时（秒）
    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_s = elapsed_ms / 1000.0;

    // 10. 计算TOPS
    // GEMM总运算次数：2 * M * N * K（每个元素需要K次乘加，每次乘加2次操作）
    double total_ops = 2.0 * (double)M * (double)N * (double)K;
    // TOPS = 总运算次数 / (耗时 * 1e12)
    double tops = total_ops / (elapsed_s * 1e12);

    // 11. 验证结果（可选，确保计算正确）
    std::vector<float> h_C(M * N);
    hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost);
    bool valid = true;
    float expected = (float)K; // A/B全为1，每个C元素应为K
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - expected) > 1e-3) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        std::cerr << "GPU " << dev_id << " GEMM计算结果验证失败！" << std::endl;
        tops = 0.0f;
    }

    // 12. 清理资源
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return (float)tops;
}

int main() {
    // 1. 检查GPU数量
    int dev_count = 0;
    hipGetDeviceCount(&dev_count);
    if (dev_count < 8) {
        std::cerr << "错误：当前系统只有" << dev_count << "个GPU，需要至少8个！" << std::endl;
        return 1;
    }
    std::cout << "检测到" << dev_count << "个AMD GPU，将使用前8个进行GEMM测试..." << std::endl;

    // 2. 存储每个GPU的TOPS
    std::vector<float> gpu_tops(8, 0.0f);

    // 3. 并行运行8个GPU的GEMM（这里用串行执行，如需真正并行可使用多线程）
    std::cout << "\n开始测试每个GPU的GEMM性能（M=" << M << ", N=" << N << ", K=" << K << "）..." << std::endl;
    std::cout << "========================================" << std::endl;
    for (int i = 0; i < 8; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        
        // 运行GEMM并计算TOPS
        gpu_tops[i] = run_gemm_on_gpu(i);
        
        std::cout << "GPU " << i << " 算力：" << std::fixed << std::setprecision(2) 
                  << gpu_tops[i] << " TOPS" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // 4. 计算总计算力
    float total_tops = 0.0f;
    for (float t : gpu_tops) {
        total_tops += t;
    }

    // 5. 输出汇总结果
    std::cout << "\n========================================" << std::endl;
    std::cout << "8个GPU GEMM算力汇总" << std::endl;
    std::cout << "========================================" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "GPU " << i << ": " << std::fixed << std::setprecision(2) << gpu_tops[i] << " TOPS" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "8卡总计算力：" << std::fixed << std::setprecision(2) << total_tops << " TOPS" << std::endl;
    std::cout << "8卡平均算力：" << std::fixed << std::setprecision(2) << (total_tops / 8) << " TOPS/卡" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}