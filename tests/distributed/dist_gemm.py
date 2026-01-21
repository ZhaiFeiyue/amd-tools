import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    """初始化DDP分布式环境"""
    # 初始化进程组，使用nccl后端（GPU推荐）
    dist.init_process_group(backend='nccl')
    # 设置当前GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def calculate_gemm_tops(
    matrix_size: int = 4096, 
    warmup_steps: int = 10, 
    test_steps: int = 50
) -> float:
    """
    计算GEMM操作的TOPS
    
    Args:
        matrix_size: 方阵的维度大小 (A: M×K, B: K×N, 这里M=K=N=matrix_size)
        warmup_steps: 预热迭代次数
        test_steps: 测试迭代次数
    
    Returns:
        单卡TOPS值
    """
    # 获取当前设备
    device = torch.cuda.current_device()
    
    # 创建随机矩阵 (float32)
    A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    B = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    
    # 预热：让GPU达到稳定状态
    for _ in range(warmup_steps):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()  # 等待GPU计算完成
    
    # 测试阶段：记录时间
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(test_steps):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 计算总耗时
    total_time = end_time - start_time
    avg_time_per_step = total_time / test_steps
    
    # GEMM运算量计算：2*M*K*N (M=K=N=matrix_size)
    # 对于A(M×K) × B(K×N) = C(M×N)，需要2*M*K*N次浮点运算
    flops_per_gemm = 2 * (matrix_size ** 3)
    
    # 转换为TOPS (1 TOPS = 10^12 次运算/秒)
    tops = (flops_per_gemm / avg_time_per_step) / (10 ** 12)
    
    return tops

def main():
    # 初始化DDP
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 配置参数
    MATRIX_SIZE = 8192  # 可根据GPU显存调整，建议从4096开始测试
    WARMUP_STEPS = 10
    TEST_STEPS = 50
    
    try:
        # 计算单卡TOPS
        single_gpu_tops = calculate_gemm_tops(MATRIX_SIZE, WARMUP_STEPS, TEST_STEPS)
        
        # 收集所有进程的结果
        all_tops = [torch.tensor(0.0, device=local_rank) for _ in range(world_size)]
        dist.all_gather(all_tops, torch.tensor(single_gpu_tops, device=local_rank))
        
        # 主进程输出结果
        if rank == 0:
            print("=" * 60)
            print(f"分布式GEMM性能测试结果 (DDP, {world_size} GPUs)")
            print("=" * 60)
            print(f"矩阵尺寸: {MATRIX_SIZE} × {MATRIX_SIZE}")
            print(f"预热步数: {WARMUP_STEPS}, 测试步数: {TEST_STEPS}")
            print("-" * 60)
            for i, tops in enumerate(all_tops):
                print(f"GPU {i} TOPS: {tops.item():.2f}")
            print("-" * 60)
            avg_tops = sum([t.item() for t in all_tops]) / world_size
            total_tops = sum([t.item() for t in all_tops])
            print(f"平均单卡TOPS: {avg_tops:.2f}")
            print(f"总TOPS (所有GPU合计): {total_tops:.2f}")
            print("=" * 60)
    
    finally:
        # 清理分布式环境
        cleanup_ddp()

if __name__ == "__main__":
    # 确保在分布式环境下运行
    # 运行命令示例: torchrun --nproc_per_node=4 gemm_ddp_tops.py
    main()