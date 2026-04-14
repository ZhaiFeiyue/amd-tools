#!/usr/bin/env python3
"""Debug v11 B data shuffle: test with uniform scales (=1.0) to isolate data issue."""
import ctypes, torch, subprocess
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import mxfp4_to_f32
torch.set_default_device("cuda"); torch.manual_seed(42)

M, N, K = 1, 16, 128  # Single MFMA tile, single K iter
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"

subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3",
    "-o", f"{KDIR}/f4gemm_m1_v11.so", f"{KDIR}/f4gemm_m1_v11.cpp"],
    check=True, capture_output=True)

quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)

x_q, _ = quant_func(x_fp, shuffle=False)
w_q, _ = quant_func(w_fp, shuffle=False)
w_shuf = shuffle_weight(w_q.clone(), layout=(16, 16))

# Uniform scales = 127 (factor 1.0) — isolate data issue
scale_a = torch.full((1, K//32), 127, dtype=torch.uint8, device="cuda")
scale_b_shuf_shape = (N, K//32)
# For ScaleB shuffled with all 127: shuffle doesn't change uniform values
scale_b = torch.full(scale_b_shuf_shape, 127, dtype=torch.uint8, device="cuda")

# Golden: dequant unshuffled + matmul (scales=1.0 means just raw fp4 product)
a_f = mxfp4_to_f32(x_q)
b_f = mxfp4_to_f32(w_q)
y_g = (a_f @ b_f.T).to(torch.bfloat16)
print(f"Golden (scale=1): {y_g[0].float().cpu().tolist()}")

# v11 with shuffled B data, uniform scales
D = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
lib = ctypes.CDLL(f"{KDIR}/f4gemm_m1_v11.so")
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8

lib.launch_f4gemm_m1(
    ctypes.c_void_p(D.data_ptr()),
    ctypes.c_void_p(x_q.view(torch.uint8).data_ptr()),  # A unshuffled
    ctypes.c_void_p(w_shuf.view(torch.uint8).data_ptr()),  # B shuffled
    ctypes.c_void_p(scale_a.data_ptr()),      # ScaleA uniform 127
    ctypes.c_void_p(scale_b.data_ptr()),      # ScaleB uniform 127
    M, N, K, K//2, K//2, N, K//32, K//32)

print(f"v11 shuf: {D[0].float().cpu().tolist()}")

# Also test with UNSHUFFLED B for comparison
# Recompile v10 for this
subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3",
    "-o", f"{KDIR}/f4gemm_m1_v10.so", f"{KDIR}/f4gemm_m1_v10.cpp"],
    check=True, capture_output=True)
lib2 = ctypes.CDLL(f"{KDIR}/f4gemm_m1_v10.so")
lib2.launch_f4gemm_m1.restype = None
lib2.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8
D2 = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
lib2.launch_f4gemm_m1(
    ctypes.c_void_p(D2.data_ptr()),
    ctypes.c_void_p(x_q.view(torch.uint8).data_ptr()),
    ctypes.c_void_p(w_q.view(torch.uint8).data_ptr()),  # B UNSHUFFLED
    ctypes.c_void_p(scale_a.data_ptr()),
    ctypes.c_void_p(scale_b.data_ptr()),
    M, N, K, K//2, K//2, N, K//32, K//32)

print(f"v10 unsh: {D2[0].float().cpu().tolist()}")

# Compare
print(f"\nPer-element:")
for i in range(N):
    g = y_g[0,i].float().item()
    s = D[0,i].float().item()
    u = D2[0,i].float().item()
    ok_s = "OK" if abs(g-s) < 0.5 else "BAD"
    ok_u = "OK" if abs(g-u) < 0.5 else "BAD"
    print(f"  N={i:>2}: golden={g:>8.2f} v11_shuf={s:>8.2f}({ok_s}) v10_unsh={u:>8.2f}({ok_u})")
