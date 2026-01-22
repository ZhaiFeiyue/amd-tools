import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP
from aiter.tuned_gemm import tgemm
import time

# 简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024).cuda()
    
    def forward(self, x):
        with record_function("model_forward"):
            return self.linear(x)

def setup_ddp(rank, world_size):
    """初始化分布式环境"""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=world_size,
        rank=rank
    )

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_profile(rank, world_size):
    # 初始化DDP
    setup_ddp(rank, world_size)
    
    # 创建模型并移到对应GPU
    model = SimpleModel().to(rank)
    model = DDP(model, device_ids=[rank])
    device = torch.device(f"cuda:{rank}")
    # print(f"使用设备: {rank} {torch.cuda.get_device_name(device)}")
    # print(f"PyTorch ROCm版本信息: {rank} {torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'}")

    M = 4096
    N = 4096
    K = 4096
    # print(f"\n矩阵尺寸: {rank} A({M}x{K}), B({K}x{N}), C({M}x{N})")
    # 生成测试数据
    dtype = torch.bfloat16  # 使用float32计算，也可以尝试float16/bfloat16
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    #torch.distributed.barrier()

    # print(f"\n开始预热运行 {rank}...")
    for _ in range(10):
        C = tgemm.mm(A, B, None, None, None, None)
    torch.cuda.synchronize()  # 等待GPU操作完成

    #torch.distributed.barrier()

    num_runs = 500
    start_time = time.time()
    for _ in range(num_runs):
        C = tgemm.mm(A, B, None, None, None, None)
    torch.cuda.synchronize()  # 等待所有GPU操作完成
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs
    
    # GEMM运算量计算: 2 * M * N * K (每个元素需要K次乘法和K-1次加法，约等于2*K)
    flops_per_gemm = 2 * M * N * K
    total_flops = flops_per_gemm * num_runs
    
    # 转换为TOPS (1 TOPS = 1e12 次运算/秒)
    tops = (flops_per_gemm / avg_time_per_run) / 1e12
    gflops = (flops_per_gemm / avg_time_per_run) / 1e9

   # tops = torch.tensor(tops, device=device)
    #dist.all_reduce(tops)
    print(f"{rank} TOPS= {tops:.4f} TOPS")
    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # print(f"Running on {world_size} GPUs")
    # 启动多进程
    mp.spawn(run_profile, args=(world_size,), nprocs=world_size, join=True)
