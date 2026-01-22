import torch
import torch.distributed as dist
import os
import argparse
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

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

    mat1 = torch.randn(batch_size, in_dim, device=f'cuda:{rank}')
    mat2 = torch.randn(in_dim, out_dim, device=f'cuda:{rank}')

    
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./profile_logs/rank_{rank}"
        )
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                gemm_result = torch.matmul(mat1, mat2)
                dist.all_reduce(gemm_result, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                prof.step()
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_profile, args=(world_size,), nprocs=world_size, join=True)
