import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
import torch.distributed as dist
import torch.multiprocessing as mp

# ====================== 1. 定义简单的大模型（模拟LLM/CNN） ======================
class LargeModel(nn.Module):
    def __init__(self, hidden_dim=4096, num_layers=12):
        super().__init__()
        # 构建多层全连接网络（模拟大模型的多层结构）
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ====================== 2. 自定义数据集（模拟训练数据） ======================
class RandomDataset(Dataset):
    def __init__(self, size, length, hidden_dim):
        self.len = length
        self.data = torch.randn(length, hidden_dim)  # 输入数据
        self.labels = torch.randint(0, 10, (length,))  # 标签

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len

# ====================== 3. 分布式训练主函数 ======================
def main(rank, world_size):
    # ---------------------- 初始化分布式环境 ----------------------
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # ---------------------- 配置FSDP关键参数 ----------------------
    # 1. 混合精度配置（降低显存占用，加速训练）
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,  # 参数用FP16存储
        reduce_dtype=torch.float16, # 梯度规约用FP16
        buffer_dtype=torch.float16  # 缓冲区用FP16
    )

    # 2. CPU卸载配置（极端显存不足时使用，会牺牲速度）
    cpu_offload_policy = CPUOffload(offload_params=True)

    # 3. 分片策略（推荐SHARD_GRAD_OP，平衡显存和速度）
    sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

    # ---------------------- 准备数据和模型 ----------------------
    hidden_dim = 4096
    batch_size = 8
    dataset = RandomDataset(size=hidden_dim, length=1000, hidden_dim=hidden_dim)
    # 分布式采样器（保证多卡数据不重复）
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, pin_memory=True
    )

    # 初始化模型并移到GPU
    model = LargeModel(hidden_dim=hidden_dim).to(rank)

    # ---------------------- 用FSDP包装模型 ----------------------
    model.train()
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        # cpu_offload=cpu_offload_policy,  # 显存足够时注释掉，加速训练
        auto_wrap_policy=None,  # 手动包装（简单模型），复杂模型用size_based_auto_wrap_policy
        device_id=torch.cuda.current_device(),
        sync_module_states=True,  # 多卡同步模型初始化参数
        forward_prefetch=True,   # 预取数据，提升效率
    )


    # ---------------------- 训练循环 ----------------------
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,  # 同时分析CPU
            torch.profiler.ProfilerActivity.CUDA, # 分析GPU
        ],
        record_shapes=False,  # 记录张量形状
        profile_memory=False, # 记录显存占用
        with_stack=True,     # 记录调用栈（定位代码行）
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./fsdp")  # 输出TensorBoard日志
    ) as prof:
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(rank, non_blocking=True)
            outputs = model(data)
            if rank == 0:
                print(f" Batch {batch_idx}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 自动获取GPU数量
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)