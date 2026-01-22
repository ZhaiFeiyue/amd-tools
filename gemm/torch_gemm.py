import torch
import time
import numpy as np

def rocm_gemm_with_tops():
    """
    在ROCm环境下执行GEMM并计算TOPS
    """
    # 1. 检查ROCm和GPU设备
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA/ROCm设备，请确认ROCm环境配置正确！")
        return
    
    # 设置ROCm设备
    device = torch.device("cuda:0")
    print(f"使用设备: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch ROCm版本信息: {torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'}")
    
    # 2. 定义矩阵尺寸 (M=行数, N=列数, K=中间维度)
    # 注意：尺寸要根据GPU显存合理设置，避免OOM
    M = 4096
    N = 4096
    K = 4096
    print(f"\n矩阵尺寸: A({M}x{K}), B({K}x{N}), C({M}x{N})")
    
    # 3. 创建随机矩阵并移到GPU
    dtype = torch.float32  # 使用float32计算，也可以尝试float16/bfloat16
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    
    # 4. 预热运行 - 避免首次编译开销
    print("\n开始预热运行...")
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()  # 等待GPU操作完成
    
    # 5. 性能测试
    print("开始性能测试...")
    num_runs = 100  # 多次运行取平均
    start_time = time.time()
    
    for _ in range(num_runs):
        C = torch.matmul(A, B)
    
    torch.cuda.synchronize()  # 等待所有GPU操作完成
    end_time = time.time()
    
    # 6. 计算性能指标
    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs
    
    # GEMM运算量计算: 2 * M * N * K (每个元素需要K次乘法和K-1次加法，约等于2*K)
    flops_per_gemm = 2 * M * N * K
    total_flops = flops_per_gemm * num_runs
    
    # 转换为TOPS (1 TOPS = 1e12 次运算/秒)
    tops = (flops_per_gemm / avg_time_per_run) / 1e12
    gflops = (flops_per_gemm / avg_time_per_run) / 1e9
    
    # 7. 打印结果
    print("\n===== 性能测试结果 =====")
    print(f"总运行时间: {total_time:.4f} 秒")
    print(f"单次运行平均时间: {avg_time_per_run:.6f} 秒")
    print(f"单次GEMM运算量: {flops_per_gemm / 1e9:.2f} GFLOPs")
    print(f"平均计算性能: {gflops:.2f} GFLOPS")
    print(f"平均计算性能: {tops:.4f} TOPS")
    print(f"矩阵乘法结果维度: {C.shape}")

if __name__ == "__main__":
    # 设置PyTorch ROCm相关配置
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
    torch.backends.cudnn.allow_tf32 = True
    
    # 执行GEMM并计算TOPS
    rocm_gemm_with_tops()