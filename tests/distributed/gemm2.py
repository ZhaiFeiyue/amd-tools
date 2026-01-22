import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP

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
    
    # 生成测试数据
    input_data = torch.randn(256, 1024).to(rank)
    
    # 关键：多卡Profiler配置
    # 1. 显式指定要跟踪的活动（CPU+CUDA）
    # 2. 设置record_shapes=True捕获张量形状
    # 3. 设置profile_memory=True捕获内存使用
    # 4. 多卡下建议每个进程生成独立的trace文件
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA  # 必须显式开启CUDA追踪
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # 可选：捕获调用栈
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./profile_logs/rank_{rank}"  # 每个rank生成独立日志
        )
    ) as prof:
        # 运行前向+反向传播
        for _ in range(10):
            output = model(input_data)
            prof.step()
    
    # 可选：打印当前rank的性能摘要（仅主进程打印即可）
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running on {world_size} GPUs")
    # 启动多进程
    mp.spawn(run_profile, args=(world_size,), nprocs=world_size, join=True)