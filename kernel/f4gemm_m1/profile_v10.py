#!/usr/bin/env python3
"""Profile v10 kernel with L3 cache flush and warmup.

Run under rocprofv3:
  rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR \
            SQ_INSTS_LDS SQ_INSTS_SALU SQ_BUSY_CYCLES SQ_WAIT_ANY \
            -- python3 profile_v10.py
"""
import ctypes, torch, subprocess
import aiter

torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"

# Compile
subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3",
    "-o", f"{KDIR}/f4gemm_m1_v10.so", f"{KDIR}/f4gemm_m1_v10.cpp"],
    check=True, capture_output=True)

# Prepare inputs (unshuffled)
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)
x_q, x_sc = quant_func(x_fp, shuffle=False)
w_q, w_sc = quant_func(w_fp, shuffle=False)
x_u8 = x_q.view(torch.uint8)
w_u8 = w_q.view(torch.uint8)
x_sc_u8 = x_sc.view(torch.uint8)
w_sc_u8 = w_sc.view(torch.uint8)

lib = ctypes.CDLL(f"{KDIR}/f4gemm_m1_v10.so")
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8

D = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

def run_kernel():
    lib.launch_f4gemm_m1(
        ctypes.c_void_p(D.data_ptr()),
        ctypes.c_void_p(x_u8.data_ptr()),
        ctypes.c_void_p(w_u8.data_ptr()),
        ctypes.c_void_p(x_sc_u8.data_ptr()),
        ctypes.c_void_p(w_sc_u8.data_ptr()),
        M, N, K, K//2, K//2, N, K//32, K//32)

def flush_l3():
    """Flush L3 cache (256 MB) by writing a large buffer."""
    flush_buf = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush_buf.fill_(1.0)
    torch.cuda.synchronize()
    del flush_buf

# Warmup: run kernel 10 times to stabilize clocks and JIT
print("Warmup...")
for _ in range(10):
    run_kernel()
torch.cuda.synchronize()
print("Warmup done")

# Flush L3 cache
print("Flushing L3 cache (256 MB)...")
flush_l3()
print("L3 flushed")

# Profiled runs: 3 iterations with L3 flush between each
print("Profiling 3 iterations with L3 flush...")
for i in range(3):
    flush_l3()
    run_kernel()
    torch.cuda.synchronize()
    print(f"  Iteration {i} done")

print("Profile complete")
