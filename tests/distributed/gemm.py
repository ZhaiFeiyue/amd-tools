import torch
import torch.distributed as dist
import os
import argparse

def setup_ddp():
    """初始化DDP环境（单机8卡）"""
    # 解析命令行参数（由torchrun自动传入）
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 初始化分布式环境
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',  # 单机多卡推荐使用nccl后端
        init_method='env://',  # 从环境变量读取分布式信息
        world_size=8,  # 单机8卡
        rank=args.local_rank
    )
    
    return args.local_rank

def gemm_with_allreduce(rank, world_size=8):
    """执行GEMM计算并通过allreduce同步结果"""
    # 设置随机种子保证各卡初始张量可复现
    torch.manual_seed(42 + rank)
    
    # 1. 创建大矩阵用于GEMM计算（每个卡上的矩阵尺寸相同）
    # 注意：为了体现allreduce的作用，我们让每个卡的初始矩阵有微小差异
    batch_size = 1024
    in_dim = 512
    out_dim = 512
    
    # 主矩阵（各卡基础值相同）
    mat1 = torch.randn(batch_size, in_dim, device=f'cuda:{rank}')
    mat2 = torch.randn(in_dim, out_dim, device=f'cuda:{rank}')
    
    # 加入rank相关的微小偏移（模拟各卡计算差异）
    mat1 = mat1 + rank * 0.01
    mat2 = mat2 + rank * 0.01
    
    # 2. 执行GEMM计算（矩阵乘法）
    # torch.matmul 就是PyTorch中GEMM的实现
    gemm_result = torch.matmul(mat1, mat2)
    print(f"Rank {rank} - GEMM计算完成，结果形状: {gemm_result.shape}")
    
    # 3. 显式执行allreduce同步各卡的GEMM结果
    # allreduce模式：默认是SUM，即所有卡的结果求和后平均
    dist.all_reduce(gemm_result, op=dist.ReduceOp.SUM)
    gemm_result = gemm_result / world_size  # 求平均
    
    # 4. 验证同步结果（仅rank0打印）
    if rank == 0:
        print(f"\nRank {rank} - Allreduce完成")
        print(f"GEMM结果均值: {gemm_result.mean().item():.6f}")
        print(f"GEMM结果方差: {gemm_result.var().item():.6f}")
    
    # 5. 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    # 检查是否有8张GPU
    assert torch.cuda.device_count() >= 8, "需要至少8张GPU"
    
    # 初始化DDP并执行GEMM+allreduce
    local_rank = setup_ddp()
    gemm_with_allreduce(local_rank)