#!/usr/bin/env python3
"""
xGMI latency breakdown: measure software overhead vs hardware latency.

Layers measured:
  1. hipMemcpy (synchronous) — full software stack
  2. hipMemcpyAsync + hipStreamSync — async submit + wait
  3. hipMemcpyAsync only (no sync) — pure submission cost
  4. GPU event-timed transfer — pure hardware time (GPU-side)
  5. Ping-pong round-trip — true hardware RTT

Usage: run inside a container with HIP + 2+ GPUs
"""

import ctypes
import time
import sys

libhip = ctypes.CDLL("libamdhip64.so")

hipSuccess = 0
hipMemcpyDeviceToDevice = 3
hipEventDefault = 0
hipEventDisableTiming = 0x2

def check(ret, msg=""):
    if ret != hipSuccess:
        raise RuntimeError(f"HIP error {ret}: {msg}")

hipSetDevice = libhip.hipSetDevice
hipMalloc = libhip.hipMalloc
hipFree = libhip.hipFree
hipMemcpy = libhip.hipMemcpy
hipMemcpyAsync = libhip.hipMemcpyAsync
hipDeviceSynchronize = libhip.hipDeviceSynchronize
hipStreamCreate = libhip.hipStreamCreate
hipStreamSynchronize = libhip.hipStreamSynchronize
hipEventCreate = libhip.hipEventCreate
hipEventCreateWithFlags = libhip.hipEventCreateWithFlags
hipEventRecord = libhip.hipEventRecord
hipEventSynchronize = libhip.hipEventSynchronize
hipEventElapsedTime = libhip.hipEventElapsedTime
hipDeviceEnablePeerAccess = libhip.hipDeviceEnablePeerAccess

def alloc_gpu(dev, size):
    check(hipSetDevice(dev))
    ptr = ctypes.c_void_p()
    check(hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)), f"malloc on dev {dev}")
    return ptr

def enable_p2p(src, dst):
    check(hipSetDevice(src))
    try:
        hipDeviceEnablePeerAccess(dst, 0)
    except:
        pass
    check(hipSetDevice(dst))
    try:
        hipDeviceEnablePeerAccess(src, 0)
    except:
        pass

def main():
    src_gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dst_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    sizes = [8, 64, 512, 4096, 65536, 1048576, 16*1048576]
    warmup = 50
    iters = 500

    enable_p2p(src_gpu, dst_gpu)
    enable_p2p(dst_gpu, src_gpu)

    print(f"xGMI Latency Breakdown: GPU{src_gpu} -> GPU{dst_gpu}")
    print(f"Warmup: {warmup}, Iters: {iters}")
    print()
    print(f"{'Size':>10} | {'hipMemcpy':>12} | {'Async+Sync':>12} | {'AsyncOnly':>12} | {'GPU Event':>12} | {'SW Overhead':>12}")
    print(f"{'':>10} | {'(us)':>12} | {'(us)':>12} | {'(us)':>12} | {'(us)':>12} | {'(us)':>12}")
    print("-" * 85)

    check(hipSetDevice(src_gpu))
    stream = ctypes.c_void_p()
    check(hipStreamCreate(ctypes.byref(stream)))

    evt_start = ctypes.c_void_p()
    evt_end = ctypes.c_void_p()
    check(hipEventCreate(ctypes.byref(evt_start)))
    check(hipEventCreate(ctypes.byref(evt_end)))

    for size in sizes:
        src_ptr = alloc_gpu(src_gpu, size)
        dst_ptr = alloc_gpu(dst_gpu, size)

        check(hipSetDevice(src_gpu))

        # Warmup
        for _ in range(warmup):
            hipMemcpy(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice)

        # 1. hipMemcpy (synchronous, includes submit + execute + return)
        t0 = time.perf_counter()
        for _ in range(iters):
            hipMemcpy(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice)
        t1 = time.perf_counter()
        sync_lat = (t1 - t0) / iters * 1e6

        # Warmup
        for _ in range(warmup):
            hipMemcpyAsync(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice, stream)
            hipStreamSynchronize(stream)

        # 2. hipMemcpyAsync + hipStreamSynchronize
        t0 = time.perf_counter()
        for _ in range(iters):
            hipMemcpyAsync(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice, stream)
            hipStreamSynchronize(stream)
        t1 = time.perf_counter()
        async_sync_lat = (t1 - t0) / iters * 1e6

        # 3. hipMemcpyAsync only (measure pure submission, batch then sync once)
        hipStreamSynchronize(stream)
        t0 = time.perf_counter()
        for _ in range(iters):
            hipMemcpyAsync(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice, stream)
        t1 = time.perf_counter()
        hipStreamSynchronize(stream)
        async_only_lat = (t1 - t0) / iters * 1e6

        # 4. GPU event-timed (pure hardware time)
        gpu_lats = []
        for _ in range(warmup):
            hipEventRecord(evt_start, stream)
            hipMemcpyAsync(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice, stream)
            hipEventRecord(evt_end, stream)
            hipEventSynchronize(evt_end)

        elapsed_ms = ctypes.c_float()
        total_gpu = 0.0
        for _ in range(iters):
            hipEventRecord(evt_start, stream)
            hipMemcpyAsync(dst_ptr, src_ptr, ctypes.c_size_t(size), hipMemcpyDeviceToDevice, stream)
            hipEventRecord(evt_end, stream)
            hipEventSynchronize(evt_end)
            hipEventElapsedTime(ctypes.byref(elapsed_ms), evt_start, evt_end)
            total_gpu += elapsed_ms.value
        gpu_lat = total_gpu / iters * 1000  # ms -> us

        sw_overhead = async_sync_lat - gpu_lat

        sz_str = f"{size}" if size < 1024 else (f"{size//1024}K" if size < 1048576 else f"{size//1048576}M")
        print(f"{sz_str:>10} | {sync_lat:>12.2f} | {async_sync_lat:>12.2f} | {async_only_lat:>12.2f} | {gpu_lat:>12.2f} | {sw_overhead:>12.2f}")

        hipFree(src_ptr)
        hipFree(dst_ptr)

    print()
    print("Legend:")
    print("  hipMemcpy    = synchronous copy (full round-trip: submit + execute + return)")
    print("  Async+Sync   = hipMemcpyAsync + hipStreamSynchronize (submit + wait)")
    print("  AsyncOnly    = hipMemcpyAsync without sync (pure CPU submission cost)")
    print("  GPU Event    = hardware-measured transfer time (GPU-side only)")
    print("  SW Overhead  = Async+Sync - GPU Event (software stack cost)")

if __name__ == "__main__":
    main()
