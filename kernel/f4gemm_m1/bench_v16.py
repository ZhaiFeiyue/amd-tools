#!/usr/bin/env python3
"""Benchmark v16 with FULLY preshuffled inputs (same format as CK/ASM)."""
import ctypes, torch, triton, subprocess, sys
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

torch.set_default_device("cuda")
M, N, K = 1, 4096, 7168
KDIR = "/apps/feiyue/rocm-optimization-skill/kernel/f4gemm_m1"

# Compile
src = f"{KDIR}/f4gemm_m1_v16.cpp"
SO = f"{KDIR}/f4gemm_m1_v16.so"
subprocess.run(["hipcc", "-shared", "-fPIC", "--offload-arch=gfx950", "-O3", "-o", SO, src],
               check=True, capture_output=True)
print(f"Compiled -> {SO}")

# Generate inputs
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x_fp = torch.randn((M, K), dtype=torch.bfloat16)
w_fp = torch.randn((N, K), dtype=torch.bfloat16)

# Shuffled path (for v16 and CK/ASM)
x_q_s, x_sc_s = quant_func(x_fp, shuffle=True)
w_q_s, w_sc_s = quant_func(w_fp, shuffle=True)
w_shuffled = shuffle_weight(w_q_s, layout=(16, 16))

# Unshuffled path (for golden reference)
x_q_u, x_sc_u = quant_func(x_fp, shuffle=False)
w_q_u, w_sc_u = quant_func(w_fp, shuffle=False)
x_sc_u8 = x_sc_u.view(torch.uint8)
w_sc_u8 = w_sc_u.view(torch.uint8)

# Golden reference
a_f = mxfp4_to_f32(x_q_u)
b_f = mxfp4_to_f32(w_q_u)
a_s = e8m0_to_f32(x_sc_u8[:M]).repeat_interleave(32, dim=1)
b_s = e8m0_to_f32(w_sc_u8[:N]).repeat_interleave(32, dim=1)
y_golden = (a_f * a_s @ (b_f * b_s).T).to(torch.bfloat16)

# Prepare shuffled tensors as uint8 for our kernel
x_u8_s = x_q_s.view(torch.uint8)      # A data (shuffled, but M=1 trivial)
w_u8_s = w_shuffled.view(torch.uint8)  # B data (preshuffled)
x_sc_u8_s = x_sc_s.view(torch.uint8)  # ScaleA (shuffled)
w_sc_u8_s = w_sc_s.view(torch.uint8)  # ScaleB (shuffled)

print(f"A data:  {x_u8_s.shape}")
print(f"B data:  {w_u8_s.shape} (preshuffled)")
print(f"ScaleA:  {x_sc_u8_s.shape} stride={x_sc_u8_s.stride(0)} (shuffled)")
print(f"ScaleB:  {w_sc_u8_s.shape} stride={w_sc_u8_s.stride(0)} (shuffled)")

# Load kernel
D_hip = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
lib = ctypes.CDLL(SO)
lib.launch_f4gemm_m1.restype = None
lib.launch_f4gemm_m1.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*8

# Run with ALL shuffled inputs
# stride_SA = padded column count of shuffled ScaleA tensor
# stride_SB = padded column count of shuffled ScaleB tensor
stride_SA = x_sc_u8_s.stride(0)
stride_SB = w_sc_u8_s.stride(0)

lib.launch_f4gemm_m1(
    ctypes.c_void_p(D_hip.data_ptr()),
    ctypes.c_void_p(x_u8_s.data_ptr()),     # A shuffled
    ctypes.c_void_p(w_u8_s.data_ptr()),      # B preshuffled
    ctypes.c_void_p(x_sc_u8_s.data_ptr()),   # ScaleA shuffled
    ctypes.c_void_p(w_sc_u8_s.data_ptr()),   # ScaleB shuffled
    M, N, K,
    K // 2,      # stride_A (bytes per row)
    K // 2,      # stride_B (bytes per row, logical)
    N,           # stride_D
    stride_SA,   # padded column count for shuffled ScaleA
    stride_SB,   # padded column count for shuffled ScaleB
)

# Correctness check
max_diff = (D_hip[0].float() - y_golden[0].float()).abs().max().item()
mean_diff = (D_hip[0].float() - y_golden[0].float()).abs().mean().item()
print(f"\nCorrectness: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")
print(f"  Golden[:8]: {y_golden[0,:8].float().cpu().tolist()}")
print(f"  v16[:8]:    {D_hip[0,:8].float().cpu().tolist()}")

nonzero = (D_hip[0].abs() > 0).sum().item()
print(f"  Nonzero: {nonzero}/{N}")

# Benchmark v16
def run_v16():
    lib.launch_f4gemm_m1(
        ctypes.c_void_p(D_hip.data_ptr()),
        ctypes.c_void_p(x_u8_s.data_ptr()),
        ctypes.c_void_p(w_u8_s.data_ptr()),
        ctypes.c_void_p(x_sc_u8_s.data_ptr()),
        ctypes.c_void_p(w_sc_u8_s.data_ptr()),
        M, N, K, K//2, K//2, N, stride_SA, stride_SB)

ms_v16 = triton.testing.do_bench(run_v16, warmup=15, rep=80)
tflops_v16 = 2.0 * M * N * K / ms_v16 * 1e-9
print(f"\nv16 preshuffled: {ms_v16:.4f} ms, {tflops_v16:.2f} TFLOPS")

# Benchmark CK best (id=14, splitK=2)
out_ck = torch.zeros(((M+31)//32*32, N), dtype=torch.bfloat16, device="cuda")
def run_ck():
    aiter.gemm_a4w4_blockscale_tune(x_q_s, w_shuffled, x_sc_s, w_sc_s, out_ck, kernelId=14, splitK=2)
run_ck(); torch.cuda.synchronize()
ms_ck = triton.testing.do_bench(run_ck, warmup=15, rep=80)
tflops_ck = 2.0 * M * N * K / ms_ck * 1e-9
print(f"CK best:         {ms_ck:.4f} ms, {tflops_ck:.2f} TFLOPS")

# ASM default
def run_asm():
    aiter.gemm_a4w4_asm(x_q_s, w_shuffled, x_sc_s, w_sc_s,
                        torch.zeros(((M+31)//32*32, N), dtype=torch.bfloat16, device="cuda"),
                        "", None, 1.0, 0.0, True, 0)
run_asm(); torch.cuda.synchronize()
ms_asm = triton.testing.do_bench(run_asm, warmup=15, rep=80)
tflops_asm = 2.0 * M * N * K / ms_asm * 1e-9
print(f"ASM default:     {ms_asm:.4f} ms, {tflops_asm:.2f} TFLOPS")

print(f"\n{'='*50}")
print(f"v16 vs CK:  {tflops_v16/tflops_ck:.3f}x")
print(f"v16 vs ASM: {tflops_v16/tflops_asm:.3f}x")
