#!/usr/bin/env python3
"""Small test: N=128, K=128 (single K-iter, single N-tile)."""
import ctypes, torch
import aiter
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
torch.set_default_device("cuda"); torch.manual_seed(42)

M, N, K = 1, 128, 128
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"

qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x = torch.randn((M, K), dtype=torch.bfloat16)
w = torch.randn((N, K), dtype=torch.bfloat16)
xq, xs = qf(x, shuffle=False); wq, ws = qf(w, shuffle=False)
xu = xq.view(torch.uint8); wu = wq.view(torch.uint8)
xs8 = xs.view(torch.uint8); ws8 = ws.view(torch.uint8)

af = mxfp4_to_f32(xq); bf = mxfp4_to_f32(wq)
a_s = e8m0_to_f32(xs8[:M]).repeat_interleave(32, dim=1)
b_s = e8m0_to_f32(ws8[:N]).repeat_interleave(32, dim=1)
yg = (af * a_s @ (bf * b_s).T).to(torch.bfloat16)

D = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
lib = ctypes.CDLL(KDIR + "/f4gemm_m1_v19.so")
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8
lib.launch_f4gemm_m1(
    ctypes.c_void_p(D.data_ptr()), ctypes.c_void_p(xu.data_ptr()),
    ctypes.c_void_p(wu.data_ptr()), ctypes.c_void_p(xs8.data_ptr()),
    ctypes.c_void_p(ws8.data_ptr()), M, N, K, K//2, K//2, N, K//32, K//32)

md = (D[0].float() - yg[0].float()).abs().max().item()
print("Correctness:", "PASS" if md < 1.0 else "FAIL", "max_diff=%.4f" % md)

# Per-subtile comparison
for s in range(8):
    g_slice = yg[0, s*16:(s+1)*16].float().cpu().tolist()
    d_slice = D[0, s*16:(s+1)*16].float().cpu().tolist()
    diff = max(abs(g_slice[i] - d_slice[i]) for i in range(16))
    print("  subtile %d: max_diff=%.4f golden[0]=%.2f v19[0]=%.2f" % (s, diff, g_slice[0], d_slice[0]))
