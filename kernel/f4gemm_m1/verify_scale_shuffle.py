#!/usr/bin/env python3
"""Verify scale shuffle formula by placing 1s at known positions."""
import torch
import aiter
from aiter.utility.fp4_utils import e8m0_shuffle

torch.set_default_device("cuda")

# Test with known scale values
# ScaleB: [N, K/32] → shuffled [N_pad, K/32_pad]
N, K = 16, 128  # K/32 = 4 scale columns
s_n = K // 32   # = 4

# Create with unique values: scale[r, c] = r*s_n + c
scale_orig = torch.zeros(N, s_n, dtype=torch.uint8, device="cuda")
for r in range(N):
    for c in range(s_n):
        scale_orig[r, c] = (r * s_n + c) % 200 + 50  # avoid 0

# Shuffle
scale_shuf = e8m0_shuffle(scale_orig.view(torch.float8_e8m0fnu)).view(torch.uint8)

print(f"Original shape: {scale_orig.shape}")
print(f"Shuffled shape: {scale_shuf.shape}")

# The formula: for original (r, c):
#   i0 = r // 32, i1 = (r%32) // 16, i2 = r % 16
#   j0 = c // 8, j1 = (c%8) // 4, j2 = c % 4
#   off = 256*j0 + 64*j2 + 4*i2 + 2*j1 + i1
#   flat_L = i0 * 32 * s_n_pad + off
# Then read scale_shuf.flatten()[flat_L]

s_n_pad = scale_shuf.shape[1]  # padded column count
print(f"s_n_pad (stride_SB) = {s_n_pad}")

scale_shuf_flat = scale_shuf.cpu().flatten()
scale_orig_cpu = scale_orig.cpu()

print(f"\nVerify formula for all (r, c):")
all_ok = True
for r in range(N):
    for c in range(s_n):
        expected = scale_orig_cpu[r, c].item()
        
        i0 = r // 32
        i1 = (r % 32) // 16
        i2 = r % 16
        j0 = c // 8
        j1 = (c % 8) // 4
        j2 = c % 4
        off = 256 * j0 + 64 * j2 + 4 * i2 + 2 * j1 + i1
        flat_L = i0 * 32 * s_n_pad + off
        
        if flat_L < len(scale_shuf_flat):
            actual = scale_shuf_flat[flat_L].item()
        else:
            actual = -1
        
        ok = (actual == expected)
        if not ok:
            all_ok = False
            print(f"  FAIL: r={r}, c={c}: expected={expected}, got={actual} (flat_L={flat_L})")

if all_ok:
    print(f"  ALL {N * s_n} entries CORRECT!")

# Test with actual M=1, K=7168 dimensions
print(f"\n=== Test with M=1, K=7168 ===")
M2, K2 = 1, 7168
s_n2 = K2 // 32  # = 224

# ScaleA for M=1
sa = torch.randint(120, 130, (M2, s_n2), dtype=torch.uint8, device="cuda")
sa_shuf = e8m0_shuffle(sa.view(torch.float8_e8m0fnu)).view(torch.uint8)
print(f"ScaleA: orig={sa.shape}, shuffled={sa_shuf.shape}")
s_n_pad_a = sa_shuf.shape[1]
print(f"ScaleA stride (s_n_pad) = {s_n_pad_a}")

# Verify ScaleA: r=0, c=0..223
sa_flat = sa_shuf.cpu().flatten()
sa_cpu = sa.cpu()
ok_count = 0
for c in range(s_n2):
    r = 0
    i0, i1, i2 = 0, 0, 0
    j0 = c // 8
    j1 = (c % 8) // 4
    j2 = c % 4
    off = 256 * j0 + 64 * j2 + 4 * i2 + 2 * j1 + i1
    flat_L = i0 * 32 * s_n_pad_a + off
    expected = sa_cpu[r, c].item()
    actual = sa_flat[flat_L].item() if flat_L < len(sa_flat) else -1
    if actual == expected:
        ok_count += 1
    else:
        print(f"  ScaleA FAIL: c={c}, expected={expected}, got={actual}, flat_L={flat_L}")

print(f"ScaleA: {ok_count}/{s_n2} correct")

# ScaleB: N=4096
N3 = 32  # test with 32 rows to be fast
sb = torch.randint(120, 130, (N3, s_n2), dtype=torch.uint8, device="cuda")
sb_shuf = e8m0_shuffle(sb.view(torch.float8_e8m0fnu)).view(torch.uint8)
print(f"\nScaleB: orig={sb.shape}, shuffled={sb_shuf.shape}")
s_n_pad_b = sb_shuf.shape[1]
print(f"ScaleB stride (s_n_pad) = {s_n_pad_b}")

sb_flat = sb_shuf.cpu().flatten()
sb_cpu = sb.cpu()
ok_count = 0
fail_count = 0
for r in range(N3):
    for c in range(min(s_n2, 20)):  # spot check
        i0 = r // 32
        i1 = (r % 32) // 16
        i2 = r % 16
        j0 = c // 8
        j1 = (c % 8) // 4
        j2 = c % 4
        off = 256 * j0 + 64 * j2 + 4 * i2 + 2 * j1 + i1
        flat_L = i0 * 32 * s_n_pad_b + off
        expected = sb_cpu[r, c].item()
        actual = sb_flat[flat_L].item() if flat_L < len(sb_flat) else -1
        if actual == expected:
            ok_count += 1
        else:
            fail_count += 1
            if fail_count <= 5:
                print(f"  ScaleB FAIL: r={r}, c={c}, expected={expected}, got={actual}, flat_L={flat_L}, len={len(sb_flat)}")

print(f"ScaleB: {ok_count}/{ok_count+fail_count} correct, {fail_count} failures")
