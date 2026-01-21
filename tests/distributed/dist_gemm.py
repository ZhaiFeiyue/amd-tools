import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    """初始化DDP分布式环境 (适配spawn方式)"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # 固定端口，避免冲突
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前GPU
    torch.cuda.set_device(rank)
    return rank

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
    flops_per_gemm = 2 * (matrix_size ** 3)
    
    # 转换为TOPS (1 TOPS = 10^12 次运算/秒)
    tops = (flops_per_gemm / avg_time_per_step) / (10 ** 12)
    
    return tops

def worker(rank, world_size, matrix_size, warmup_steps, test_steps):
    """每个进程的工作函数 (由spawn启动)"""
    try:
        # 初始化DDP
        setup_ddp(rank, world_size)
        
        # 计算单卡TOPS
        single_gpu_tops = calculate_gemm_tops(matrix_size, warmup_steps, test_steps)
        
        # 收集所有进程的结果
        all_tops = [torch.tensor(0.0, device=rank) for _ in range(world_size)]
        dist.all_gather(all_tops, torch.tensor(single_gpu_tops, device=rank))
        
        # 主进程输出结果
        if rank == 0:
            print("=" * 60)
            print(f"分布式GEMM性能测试结果 (Spawn + DDP, {world_size} GPUs)")
            print("=" * 60)
            print(f"矩阵尺寸: {matrix_size} × {matrix_size}")
            print(f"预热步数: {warmup_steps}, 测试步数: {test_steps}")
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

def main():
    # 配置参数
    WORLD_SIZE = torch.cuda.device_count()  # 自动获取可用GPU数量
    MATRIX_SIZE = 4096  # 根据GPU显存调整（如2048/4096/8192）
    WARMUP_STEPS = 10
    TEST_STEPS = 50
    
    # 检查GPU数量
    if WORLD_SIZE < 1:
        print("错误：未检测到GPU，请确保使用支持CUDA的环境")
        return
    
    print(f"检测到 {WORLD_SIZE} 个GPU，开始分布式GEMM测试...")
    
    # 使用spawn启动多进程
    mp.spawn(
        worker,
        args=(WORLD_SIZE, MATRIX_SIZE, WARMUP_STEPS, TEST_STEPS),
        nprocs=WORLD_SIZE,
        join=True
    )

if __name__ == "__main__":
    # 设置多进程启动方式
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()