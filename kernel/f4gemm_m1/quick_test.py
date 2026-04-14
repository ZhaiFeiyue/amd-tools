#!/usr/bin/env python3
"""Quick test for any version with unshuffled inputs."""
import ctypes, torch, triton, subprocess, sys
import aiter
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"
ver = sys.argv[1] if len(sys.argv) > 1 else "v19"
SO = f"{KDIR}/f4gemm_m1_{ver}.so"

quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)
x_q, x_sc = quant_func(x_fp, shuffle=False)
w_q, w_sc = quant_func(w_fp, shuffle=False)
x_u8 = x_q.view(torch.uint8); w_u8 = w_q.view(torch.uint8)
x_sc8 = x_sc.view(torch.uint8); w_sc8 = w_sc.view(torch.uint8)

a_f = mxfp4_to_f32(x_q); b_f = mxfp4_to_f32(w_q)
a_s = e8m0_to_f32(x_sc8[:M]).repeat_interleave(32, dim=1)
b_s = e8m0_to_f32(w_sc8[:N]).repeat_interleave(32, dim=1)
y_g = (a_f * a_s @ (b_f * b_s).T).to(torch.bfloat16)

D = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
lib = ctypes.CDLL(SO)
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8

def run():
    lib.launch_f4gemm_m1(
        ctypes.c_void_p(D.data_ptr()), ctypes.c_void_p(x_u8.data_ptr()),
        ctypes.c_void_p(w_u8.data_ptr()), ctypes.c_void_p(x_sc8.data_ptr()),
        ctypes.c_void_p(w_sc8.data_ptr()), M, N, K, K//2, K//2, N, K//32, K//32)

run()
md = (D[0].float() - y_g[0].float()).abs().max().item()
print(f"{ver}: max_diff={md:.6f} {'PASS' if md < 1.0 else 'FAIL'}")

ms = triton.testing.do_bench(run, warmup=25, rep=100)
tf = 2.0 * M * N * K / ms * 1e-9
print(f"{ver}: {ms:.4f} ms, {tf:.2f} TFLOPS")
