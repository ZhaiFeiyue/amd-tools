#!/usr/bin/env python3
"""Profile v11 kernel with L3 cache flush and warmup."""
import ctypes, torch, subprocess
import aiter
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"

subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3",
    "-o", f"{KDIR}/f4gemm_m1_v11.so", f"{KDIR}/f4gemm_m1_v11.cpp"],
    check=True, capture_output=True)

quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)

x_q_u, x_sc_u = quant_func(x_fp, shuffle=False)
_, w_sc_s = quant_func(w_fp, shuffle=True)
w_q_u, _ = quant_func(w_fp, shuffle=False)
w_shuf = shuffle_weight(w_q_u, layout=(16, 16))

x_u8 = x_q_u.view(torch.uint8)
w_u8 = w_shuf.view(torch.uint8)
x_sc_u8 = x_sc_u.view(torch.uint8)
w_sc_u8 = w_sc_s.view(torch.uint8)

lib = ctypes.CDLL(f"{KDIR}/f4gemm_m1_v11.so")
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8

D = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
stride_SB = w_sc_u8.shape[1]

def run_kernel():
    lib.launch_f4gemm_m1(
        ctypes.c_void_p(D.data_ptr()),
        ctypes.c_void_p(x_u8.data_ptr()),
        ctypes.c_void_p(w_u8.data_ptr()),
        ctypes.c_void_p(x_sc_u8.data_ptr()),
        ctypes.c_void_p(w_sc_u8.data_ptr()),
        M, N, K, K//2, K//2, N, K//32, stride_SB)

def flush_l3():
    flush_buf = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush_buf.fill_(1.0)
    torch.cuda.synchronize()
    del flush_buf

print("Warmup...")
for _ in range(10):
    run_kernel()
torch.cuda.synchronize()
print("Warmup done")

print("Flushing L3...")
flush_l3()

print("Profiling 3 iterations with L3 flush...")
for i in range(3):
    flush_l3()
    run_kernel()
    torch.cuda.synchronize()
    print(f"  Iteration {i} done")

print("Profile complete")
