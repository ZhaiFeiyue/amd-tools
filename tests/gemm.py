import torch
import numpy as np
import time

#[1,8192] x[8192,8192] = [1,8192]
m, k, n = 2**0, 8192, 8192
m, k, n = 2**2, 4096, 8192
it = 10
AA = []
BB = []
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
A = torch.randn(m, k, device='cuda').to(torch.bfloat16)  # 随机初始化矩阵A，直接放到指定设备
B = torch.randn(k, n, device='cuda').to(torch.bfloat16)  # 随机初始化矩阵B

for i in range(it):
    A = torch.randn(m, k, device='cuda').to(torch.bfloat16)  # 随机初始化矩阵A，直接放到指定设备
    B = torch.randn(k, n, device='cuda').to(torch.bfloat16)  # 随机初始化矩阵B
    AA.append(A)
    BB.append(B)

torch.cuda.empty_cache()
C1 = torch.matmul(A, B)

torch.cuda.synchronize()
latencies = []
start_event.record()
for A, B in zip(AA, BB):
    C1 = torch.matmul(A, B)
end_event.record()
end_event.synchronize()
avg = start_event.elapsed_time(end_event)/it * 1000

print(avg)
print("tops:", m* k * n * 2 / (avg/1e6)/1e12)
print("bw:", (m* k + k * n + m * n) * 2 / (avg/1e6)/1e12)

ops = m * k * n * 2
Bytes = (m * k + k * n + m * n) * 2

padding_m = ((m - 1) // 32 + 1) * 32
padding_ops = padding_m * k * n * 2
padding_Bytes = (padding_m * k + k * n + padding_m * n) * 2
print(f'shape = {m},{k},{n} AI = {ops/Bytes}, padding AI = {padding_ops/padding_Bytes}')
print(f'weights = {Bytes} bytes = {Bytes/1e9}')


with torch.profiler.profile(
    activities=[
        # torch.profiler.ProfilerActivity.CPU,  # 同时分析CPU
        torch.profiler.ProfilerActivity.CUDA, # 分析GPU
    ],
    record_shapes=False,  # 记录张量形状
    profile_memory=False, # 记录显存占用
    with_stack=False,     # 记录调用栈（定位代码行）
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./gpu_profile")  # 输出TensorBoard日志
) as prof:
    for A, B in zip(AA, BB):
        torch.cuda.empty_cache()
        C1 = torch.matmul(A, B)
        torch.cuda.synchronize()  # 等待GPU任务完成，确保计时准确
        prof.step()  # 标记步骤（可选，用于区分迭代）
print(C1)

