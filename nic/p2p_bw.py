#!/usr/bin/env python3
"""GPU P2P bandwidth test using hipMemcpy via ctypes (no PyTorch needed)."""
import ctypes, ctypes.util, time, sys, argparse

libhip = ctypes.CDLL("libamdhip64.so")

hipSuccess = 0
hipMemcpyDeviceToDevice = 3

hipSetDevice = libhip.hipSetDevice
hipMalloc = libhip.hipMalloc
hipFree = libhip.hipFree
hipMemcpy = libhip.hipMemcpy
hipDeviceSynchronize = libhip.hipDeviceSynchronize
hipDeviceEnablePeerAccess = libhip.hipDeviceEnablePeerAccess
hipDeviceCanAccessPeer = libhip.hipDeviceCanAccessPeer
hipMemset = libhip.hipMemset

hipEventCreate = libhip.hipEventCreate
hipEventRecord = libhip.hipEventRecord
hipEventSynchronize = libhip.hipEventSynchronize
hipEventElapsedTime = libhip.hipEventElapsedTime
hipEventDestroy = libhip.hipEventDestroy
hipMemcpyAsync = libhip.hipMemcpyAsync

def check(ret, msg=""):
    if ret != hipSuccess:
        print(f"HIP ERROR {ret}: {msg}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="GPU P2P bandwidth test")
    parser.add_argument("--src", type=int, default=0, help="Source GPU ID")
    parser.add_argument("--dst", type=int, default=1, help="Destination GPU ID")
    parser.add_argument("--size-mb", type=int, default=256, help="Transfer size in MB")
    parser.add_argument("--iters", type=int, default=200, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--duration", type=int, default=0,
                        help="If >0, run for this many seconds (ignores --iters)")
    parser.add_argument("--bidir", action="store_true", help="Bidirectional test")
    args = parser.parse_args()

    src, dst = args.src, args.dst
    nbytes = args.size_mb * 1024 * 1024

    can_access = ctypes.c_int(0)
    check(hipDeviceCanAccessPeer(ctypes.byref(can_access), src, dst), "CanAccessPeer")
    if not can_access.value:
        print(f"ERROR: GPU{src} cannot P2P access GPU{dst}")
        sys.exit(1)
    print(f"P2P: GPU{src} <-> GPU{dst}, size={args.size_mb}MB, "
          f"bidir={args.bidir}")

    # Enable peer access
    check(hipSetDevice(src), "SetDevice src")
    check(hipDeviceEnablePeerAccess(dst, 0), "EnablePeerAccess src->dst")
    check(hipSetDevice(dst), "SetDevice dst")
    check(hipDeviceEnablePeerAccess(src, 0), "EnablePeerAccess dst->src")

    # Allocate
    src_ptr = ctypes.c_void_p()
    dst_ptr = ctypes.c_void_p()
    check(hipSetDevice(src), "SetDevice src")
    check(hipMalloc(ctypes.byref(src_ptr), nbytes), "Malloc src")
    check(hipMemset(src_ptr, 0xAB, nbytes), "Memset src")
    check(hipSetDevice(dst), "SetDevice dst")
    check(hipMalloc(ctypes.byref(dst_ptr), nbytes), "Malloc dst")
    check(hipMemset(dst_ptr, 0, nbytes), "Memset dst")

    if args.bidir:
        src_ptr2 = ctypes.c_void_p()
        dst_ptr2 = ctypes.c_void_p()
        check(hipSetDevice(dst), "SetDevice dst")
        check(hipMalloc(ctypes.byref(src_ptr2), nbytes), "Malloc src2")
        check(hipMemset(src_ptr2, 0xCD, nbytes), "Memset src2")
        check(hipSetDevice(src), "SetDevice src")
        check(hipMalloc(ctypes.byref(dst_ptr2), nbytes), "Malloc dst2")

    check(hipSetDevice(src), "SetDevice src")
    check(hipDeviceSynchronize(), "Sync")

    # Warmup
    print(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        hipMemcpy(dst_ptr, src_ptr, nbytes, hipMemcpyDeviceToDevice)
        if args.bidir:
            hipMemcpy(dst_ptr2, src_ptr2, nbytes, hipMemcpyDeviceToDevice)
    check(hipDeviceSynchronize(), "Sync warmup")

    # Benchmark
    if args.duration > 0:
        print(f"Running for {args.duration}s...")
        count = 0
        t0 = time.time()
        while time.time() - t0 < args.duration:
            hipMemcpy(dst_ptr, src_ptr, nbytes, hipMemcpyDeviceToDevice)
            if args.bidir:
                hipMemcpy(dst_ptr2, src_ptr2, nbytes, hipMemcpyDeviceToDevice)
            count += 1
            if count % 50 == 0:
                check(hipDeviceSynchronize(), "Sync")
                elapsed = time.time() - t0
                directions = 2 if args.bidir else 1
                bw = count * nbytes * directions / elapsed / 1e9
                print(f"  [{elapsed:.1f}s] {count} iters, {bw:.1f} GB/s")
        check(hipDeviceSynchronize(), "Sync final")
        elapsed = time.time() - t0
        directions = 2 if args.bidir else 1
        total_bytes = count * nbytes * directions
    else:
        print(f"Benchmark ({args.iters} iters)...")
        t0 = time.time()
        for i in range(args.iters):
            hipMemcpy(dst_ptr, src_ptr, nbytes, hipMemcpyDeviceToDevice)
            if args.bidir:
                hipMemcpy(dst_ptr2, src_ptr2, nbytes, hipMemcpyDeviceToDevice)
        check(hipDeviceSynchronize(), "Sync bench")
        elapsed = time.time() - t0
        count = args.iters
        directions = 2 if args.bidir else 1
        total_bytes = count * nbytes * directions

    bw = total_bytes / elapsed / 1e9
    print(f"\nResult: {count} iters in {elapsed:.2f}s")
    print(f"  Bandwidth: {bw:.2f} GB/s")
    print(f"  Per-copy latency: {elapsed/count*1e6:.1f} us")

    hipFree(src_ptr)
    hipFree(dst_ptr)
    print("Done.")

if __name__ == "__main__":
    main()
