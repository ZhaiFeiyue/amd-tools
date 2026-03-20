#!/usr/bin/env python3
"""
MORI shmem P2P bandwidth benchmark — direct xGMI via shmem_ptr_p2p.

Uses MORI's symmetric memory allocation + P2P pointer translation to perform
direct GPU-to-GPU memcpy over xGMI, measuring true hardware bandwidth.

Usage:
    torchrun --nproc_per_node=2 mori_p2p_bench.py [--size-mb 256] [--iters 200]
    torchrun --nproc_per_node=2 mori_p2p_bench.py --duration 15
"""

import os
import sys
import time
import ctypes
import argparse

import torch
import torch.distributed as dist

libhip = ctypes.CDLL("libamdhip64.so")
hipMemcpyAsync = libhip.hipMemcpyAsync
hipMemcpyDeviceToDevice = 3


def setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    import mori.shmem as ms
    ms.shmem_torch_process_group_init("default")
    return ms.shmem_mype(), ms.shmem_npes()


def cleanup():
    import mori.shmem as ms
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="MORI shmem P2P bandwidth test")
    parser.add_argument("--size-mb", type=int, default=256, help="Transfer size in MB")
    parser.add_argument("--iters", type=int, default=200, help="Iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--duration", type=int, default=15,
                        help="Run for N seconds (0 = use --iters)")
    args = parser.parse_args()

    mype, npes = setup()

    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    nbytes = args.size_mb * 1024 * 1024
    nelems = nbytes // 4
    peer = (mype + 1) % npes

    if mype == 0:
        print(f"MORI shmem P2P benchmark")
        print(f"  GPUs: {npes}, size: {args.size_mb} MB")
        print(f"  GPU{0} -> GPU{1} (via shmem_ptr_p2p + hipMemcpyAsync)")
        print("=" * 60)

    # Allocate symmetric memory (visible to all PEs)
    src_buf = mori_shmem_create_tensor((nelems,), torch.float32)
    src_buf.fill_(float(mype) + 0.5)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # Get P2P address of peer's buffer
    remote_ptr = ms.shmem_ptr_p2p(src_buf.data_ptr(), mype, peer)
    local_ptr = src_buf.data_ptr()

    if mype == 0:
        print(f"  GPU{mype} local_ptr=0x{local_ptr:x}")
        print(f"  GPU{mype} remote_ptr(GPU{peer})=0x{remote_ptr:x}")
        if remote_ptr == 0:
            print("ERROR: shmem_ptr_p2p returned 0 (no P2P access)")
            sys.exit(1)
        print(f"  P2P pointer valid!")

    hip_stream = torch.cuda.current_stream().cuda_stream

    def do_copy():
        hipMemcpyAsync(
            ctypes.c_void_p(remote_ptr),
            ctypes.c_void_p(local_ptr),
            ctypes.c_size_t(nbytes),
            hipMemcpyDeviceToDevice,
            ctypes.c_void_p(hip_stream),
        )

    # Warmup
    if mype == 0:
        print(f"\nWarmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        do_copy()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    if args.duration > 0:
        if mype == 0:
            print(f"Running for {args.duration}s...")
        count = 0
        t0 = time.time()
        while time.time() - t0 < args.duration:
            for _ in range(50):
                do_copy()
                count += 1
            torch.cuda.synchronize()
            if mype == 0 and count % 200 == 0:
                elapsed = time.time() - t0
                bw = count * nbytes / elapsed / 1e9
                print(f"  [{elapsed:.1f}s] {count} iters, {bw:.2f} GB/s")
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        elapsed = time.time() - t0
    else:
        if mype == 0:
            print(f"Benchmark ({args.iters} iters)...")
        count = args.iters

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        ms.shmem_barrier_all()
        start_ev.record()
        for _ in range(count):
            do_copy()
        end_ev.record()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        elapsed = start_ev.elapsed_time(end_ev) / 1000.0

    bw = count * nbytes / elapsed / 1e9
    lat = elapsed / count * 1e6

    # Gather results
    bw_t = torch.tensor([bw])
    bw_list = [torch.zeros(1) for _ in range(npes)]
    dist.all_gather(bw_list, bw_t)

    if mype == 0:
        print(f"\n{'=' * 60}")
        print(f"Results ({args.size_mb} MB, {count} iters, {elapsed:.2f}s):")
        for i in range(npes):
            src_gpu = i
            dst_gpu = (i + 1) % npes
            print(f"  GPU{src_gpu} -> GPU{dst_gpu}: {bw_list[i].item():.2f} GB/s")
        print(f"  Latency: {lat:.1f} us per {args.size_mb} MB copy")
        print(f"{'=' * 60}")

    cleanup()


if __name__ == "__main__":
    main()
