import torch
import triton
import triton.language as tl
import time


# -------------------------------
# Autotuned Triton GEMM Kernel
# -------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8),
    ],
    key=['N', 'K'],  # autotune based on N and K (M is fixed)
)
@triton.jit
def gemm_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,   # fixed to 16
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m * BLOCK_SIZE_M >= M:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load with masking (though K divisible by BLOCK_SIZE_K here)
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# -------------------------------
# Wrapper with fixed BLOCK_SIZE_M=16
# -------------------------------
def gemm_triton_tuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, 16) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    gemm_kernel_autotuned[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=16,  # fixed
    )
    return c


# -------------------------------
# Benchmarking
# -------------------------------
def benchmark():
    torch.manual_seed(0)
    M, N, K = 16, 256, 256
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # Warmup
    #for _ in range(10):
    #    _ = gemm_triton_tuned(a, b)
    #    _ = torch.matmul(a, b)
    #torch.cuda.synchronize()

    # Triton timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        c_triton = gemm_triton_tuned(a, b)
    end_event.record()
    torch.cuda.synchronize()
    triton_time_ms = start_event.elapsed_time(end_event)

    # PyTorch timing
    #start_event.record()
    #for _ in range(1000):
    #    c_torch = torch.matmul(a, b)
    #end_event.record()
    #torch.cuda.synchronize()
    #torch_time_ms = start_event.elapsed_time(end_event)

    # Correctness check
    # max_diff = torch.max(torch.abs(c_torch - c_triton.to(torch.float32))).item()
    # print(f"Max diff: {max_diff:.6f}")
    print(f"Triton 1000 runs: {triton_time_ms:.3f} ms → {triton_time_ms / 1000:.3f} ms per call")
    #print(f"PyTorch 1000 runs: {torch_time_ms:.3f} ms → {torch_time_ms / 1000:.3f} ms per call")
    #print(f"Speedup: {torch_time_ms / triton_time_ms:.2f}x")


if __name__ == "__main__":
    benchmark()
