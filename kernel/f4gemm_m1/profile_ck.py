#!/usr/bin/env python3
"""Profile CK kernel (id=14, splitK=2) with L3 flush."""
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)
x_q, x_sc = quant_func(x_fp, shuffle=True)
w_q, w_sc = quant_func(w_fp, shuffle=True)
w_s = shuffle_weight(w_q, layout=(16, 16))
out = torch.zeros(((M+31)//32*32, N), dtype=torch.bfloat16, device="cuda")

def run():
    aiter.gemm_a4w4_blockscale_tune(x_q, w_s, x_sc, w_sc, out, kernelId=14, splitK=2)

def flush_l3():
    f = torch.zeros(256*1024*1024//4, dtype=torch.float32, device="cuda")
    f.fill_(1.0); torch.cuda.synchronize(); del f

print("Warmup...")
for _ in range(10): run()
torch.cuda.synchronize()
print("Profiling...")
for i in range(3):
    flush_l3(); run(); torch.cuda.synchronize()
    print(f"  iter {i}")
print("Done")
