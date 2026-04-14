#!/usr/bin/env python3
"""Profile v12."""
import ctypes, torch, subprocess
import aiter
from aiter.ops.shuffle import shuffle_weight
torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"
subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3",
    "-o", f"{KDIR}/f4gemm_m1_v12.so", f"{KDIR}/f4gemm_m1_v12.cpp"],
    check=True, capture_output=True)
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)
x_q_u, x_sc_u = quant_func(x_fp, shuffle=False)
w_q_u, _ = quant_func(w_fp, shuffle=False)
_, w_sc_s = quant_func(w_fp, shuffle=True)
w_shuf = shuffle_weight(w_q_u, layout=(16, 16))
lib = ctypes.CDLL(f"{KDIR}/f4gemm_m1_v12.so")
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8
D = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
def run():
    lib.launch_f4gemm_m1(
        ctypes.c_void_p(D.data_ptr()),
        ctypes.c_void_p(x_q_u.view(torch.uint8).data_ptr()),
        ctypes.c_void_p(w_shuf.view(torch.uint8).data_ptr()),
        ctypes.c_void_p(x_sc_u.view(torch.uint8).data_ptr()),
        ctypes.c_void_p(w_sc_s.view(torch.uint8).data_ptr()),
        M, N, K, K//2, K//2, N, K//32, w_sc_s.view(torch.uint8).shape[1])
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
