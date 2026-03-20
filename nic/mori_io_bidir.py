#!/usr/bin/env python3
"""
Simultaneous bidirectional MORI-IO xGMI write benchmark.

Launches two threads writing in opposite directions between a GPU pair,
measuring true concurrent bidirectional bandwidth.

Usage (inside a container with MORI + PyTorch):
    python3 mori_io_bidir.py [--src-gpu 3] [--dst-gpu 4] [--buf-size 1048576]
"""
import argparse
import sys
import time
import threading

import torch
from mori.io import (
    IOEngineConfig, IOEngine, BackendType, XgmiBackendConfig,
)


def run_direction(label, src_gpu, dst_gpu, buf_size, batch, iters, warmup,
                  barrier, results, idx):
    src_dev = torch.device("cuda", src_gpu)
    dst_dev = torch.device("cuda", dst_gpu)

    src_tensor = torch.ones(buf_size * batch, device=src_dev, dtype=torch.float32).to(torch.float8_e4m3fnuz)
    dst_tensor = torch.zeros(buf_size * batch, device=dst_dev, dtype=torch.float32).to(torch.float8_e4m3fnuz)

    cfg = IOEngineConfig(host="", port=0)
    engine = IOEngine(key=f"xgmi-bidir-{label}", config=cfg)
    xgmi_cfg = XgmiBackendConfig(num_streams=64, num_events=64)
    engine.create_backend(BackendType.XGMI, xgmi_cfg)

    mem = engine.register_torch_tensor(src_tensor)
    target_mem = engine.register_torch_tensor(dst_tensor)
    sess = engine.create_session(mem, target_mem)

    offsets = [i * buf_size for i in range(batch)]
    sizes = [buf_size] * batch

    # Warmup
    for _ in range(warmup):
        uid = engine.allocate_transfer_uid()
        st = sess.batch_write(offsets, offsets, sizes, uid)
        st.Wait()

    # Sync both threads before timed run
    barrier.wait()

    latencies = []
    t_start = time.time()
    for _ in range(iters):
        uid = engine.allocate_transfer_uid()
        t0 = time.time()
        st = sess.batch_write(offsets, offsets, sizes, uid)
        st.Wait()
        latencies.append(time.time() - t0)
    t_total = time.time() - t_start

    total_bytes = buf_size * batch
    bw_per_iter = [total_bytes / lat / 1e9 for lat in latencies]
    avg_bw = sum(bw_per_iter) / len(bw_per_iter)
    max_bw = max(bw_per_iter)
    agg_bw = total_bytes * iters / t_total / 1e9

    results[idx] = {"src": src_gpu, "dst": dst_gpu,
                    "avg": avg_bw, "max": max_bw, "agg": agg_bw,
                    "lat_avg_us": sum(latencies) / len(latencies) * 1e6,
                    "lat_min_us": min(latencies) * 1e6}


def main():
    parser = argparse.ArgumentParser(description="Bidirectional MORI-IO xGMI write")
    parser.add_argument("--src-gpu", type=int, default=3, help="First GPU ID")
    parser.add_argument("--dst-gpu", type=int, default=4, help="Second GPU ID")
    parser.add_argument("--buf-size", type=int, default=1048576, help="Message size in bytes")
    parser.add_argument("--batch", type=int, default=256, help="Transfers per iteration")
    parser.add_argument("--iters", type=int, default=128, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    src, dst = args.src_gpu, args.dst_gpu

    print(f"=== Simultaneous Bidirectional MORI-IO Write ===")
    print(f"  GPU{src} <-> GPU{dst}, msg={args.buf_size}B, batch={args.batch}, iters={args.iters}")
    print()

    barrier = threading.Barrier(2)
    results = [None, None]

    t1 = threading.Thread(target=run_direction,
        args=(f"{src}to{dst}", src, dst, args.buf_size, args.batch,
              args.iters, args.warmup, barrier, results, 0))
    t2 = threading.Thread(target=run_direction,
        args=(f"{dst}to{src}", dst, src, args.buf_size, args.batch,
              args.iters, args.warmup, barrier, results, 1))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"{'='*60}")
    for r in results:
        print(f"  GPU{r['src']} -> GPU{r['dst']}: "
              f"max={r['max']:.2f} GB/s, avg={r['avg']:.2f} GB/s, "
              f"lat_min={r['lat_min_us']:.0f} us")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
