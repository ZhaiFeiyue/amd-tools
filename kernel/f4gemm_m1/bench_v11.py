#!/usr/bin/env python3
"""Benchmark v11 with ALL preshuffled inputs (same format as CK/ASM)."""
import ctypes, torch, triton, subprocess, sys
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"

# Compile
ver = "v19"
src = f"{KDIR}/f4gemm_m1_{ver}.cpp"
SO = f"{KDIR}/f4gemm_m1_{ver}.so"
r = subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3", "-o", SO, src],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"COMPILE FAILED:\n{r.stderr}")
    sys.exit(1)
print(f"Compiled -> {SO}")

# Generate inputs
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)

# Shuffled path (for v11 and CK)
x_q_s, x_sc_s = quant_func(x_fp, shuffle=True)
w_q_s, w_sc_s = quant_func(w_fp, shuffle=True)
w_shuffled = shuffle_weight(w_q_s, layout=(16, 16))

# Unshuffled path (for golden)
x_q_u, x_sc_u = quant_func(x_fp, shuffle=False)
w_q_u, w_sc_u = quant_func(w_fp, shuffle=False)

# Golden reference
a_f = mxfp4_to_f32(x_q_u)
b_f = mxfp4_to_f32(w_q_u)
x_sc_u8 = x_sc_u.view(torch.uint8)
w_sc_u8 = w_sc_u.view(torch.uint8)
a_s = e8m0_to_f32(x_sc_u8[:M]).repeat_interleave(32, dim=1)
b_s = e8m0_to_f32(w_sc_u8[:N]).repeat_interleave(32, dim=1)
y_golden = (a_f * a_s @ (b_f * b_s).T).to(torch.bfloat16)

# v17: ALL UNSHUFFLED (LDS cooperative load does coalescing internally)
x_u8 = x_q_u.view(torch.uint8)
w_u8 = w_q_u.view(torch.uint8)           # B UNSHUFFLED
x_sc_u8_s = x_sc_u.view(torch.uint8)
w_sc_u8_s = w_sc_u.view(torch.uint8)     # ScaleB UNSHUFFLED

print(f"A data: {x_u8.shape}")
print(f"B data (shuffled): {w_u8.shape}")
print(f"ScaleA (shuffled): {x_sc_u8_s.shape}, stride={x_sc_u8_s.stride(0)}")
print(f"ScaleB (shuffled): {w_sc_u8_s.shape}, stride={w_sc_u8_s.stride(0)}")

D_hip = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
lib = ctypes.CDLL(SO)
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8

# All unshuffled: simple K/32 strides
stride_SA = K // 32
stride_SB = K // 32

lib.launch_f4gemm_m1(
    ctypes.c_void_p(D_hip.data_ptr()),
    ctypes.c_void_p(x_u8.data_ptr()),       # A (shuffled)
    ctypes.c_void_p(w_u8.data_ptr()),        # B (shuffled)
    ctypes.c_void_p(x_sc_u8_s.data_ptr()),   # ScaleA (shuffled)
    ctypes.c_void_p(w_sc_u8_s.data_ptr()),   # ScaleB (shuffled)
    M, N, K,
    K // 2,     # stride_A (bytes per row)
    K // 2,     # stride_B (bytes per row)
    N,          # stride_D
    stride_SA,  # ScaleA padded column count
    stride_SB,  # ScaleB padded column count
)

# Correctness check
max_diff = (D_hip[0].float() - y_golden[0].float()).abs().max().item()
mean_diff = (D_hip[0].float() - y_golden[0].float()).abs().mean().item()
print(f"\nCorrectness: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")
print(f"  Golden[:8]: {y_golden[0,:8].float().cpu().tolist()}")
print(f"  v11[:8]:    {D_hip[0,:8].float().cpu().tolist()}")

# Benchmark v11
def run_v11():
    lib.launch_f4gemm_m1(
        ctypes.c_void_p(D_hip.data_ptr()),
        ctypes.c_void_p(x_u8.data_ptr()),
        ctypes.c_void_p(w_u8.data_ptr()),
        ctypes.c_void_p(x_sc_u8_s.data_ptr()),
        ctypes.c_void_p(w_sc_u8_s.data_ptr()),
        M, N, K, K//2, K//2, N, stride_SA, stride_SB)

ms_v11 = triton.testing.do_bench(run_v11, warmup=25, rep=100)
tflops_v11 = 2.0 * M * N * K / ms_v11 * 1e-9
print(f"\nv11 (all shuffled): {ms_v11:.4f} ms, {tflops_v11:.2f} TFLOPS")

# Benchmark CK best
out_ck = torch.zeros(((M+31)//32*32, N), dtype=torch.bfloat16, device="cuda")
def run_ck():
    aiter.gemm_a4w4_blockscale_tune(x_q_s, w_shuffled, x_sc_s, w_sc_s, out_ck, kernelId=14, splitK=2)
run_ck(); torch.cuda.synchronize()
ms_ck = triton.testing.do_bench(run_ck, warmup=25, rep=100)
tflops_ck = 2.0 * M * N * K / ms_ck * 1e-9
print(f"CK best:            {ms_ck:.4f} ms, {tflops_ck:.2f} TFLOPS")

# ASM default
def run_asm():
    aiter.gemm_a4w4_asm(x_q_s, w_shuffled, x_sc_s, w_sc_s,
                        torch.zeros(((M+31)//32*32, N), dtype=torch.bfloat16, device="cuda"),
                        "", None, 1.0, 0.0, True, 0)
run_asm(); torch.cuda.synchronize()
ms_asm = triton.testing.do_bench(run_asm, warmup=25, rep=100)
tflops_asm = 2.0 * M * N * K / ms_asm * 1e-9
print(f"ASM default:        {ms_asm:.4f} ms, {tflops_asm:.2f} TFLOPS")

print(f"\n{'='*50}")
print(f"v11 vs CK:  {tflops_v11/tflops_ck:.3f}x")
print(f"v11 vs ASM: {tflops_v11/tflops_asm:.3f}x")
