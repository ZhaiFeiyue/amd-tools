import torch
import torch.distributed as dist
import os
import argparse

def run_profile(rank, world_size):
    # 初始化DDP
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=world_size,
        rank=rank
    )
    batch_size = 1024
    in_dim = 512
    out_dim = 512

    mat1 = torch.randn(batch_size, in_dim, device=f'cuda{rank}')
    mat2 = torch.randn(in_dim, out_dim, device=f'cuda{rank}')

    
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
            gemm_result = torch.matmul(mat1, mat2)
            torch.cuda.sychronize()
            prof.step()
    
    # 可选：打印当前rank的性能摘要（仅主进程打印即可）
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    dist.all_reduce(gemm_result, op=dist.ReduceOp.SUM)
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_profile, args=(world_size,), nprocs=world_size, join=True)
