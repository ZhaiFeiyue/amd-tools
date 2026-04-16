---
name: cdna4-architecture
description: AMD CDNA 4 architecture reference for MI350 Series GPUs (MI350X/MI355X, gfx950). Covers chiplet topology, CU internals, Matrix Cores, numerical formats (MXFP8/6/4), memory hierarchy, Infinity Fabric, and partitioning. Use when optimizing kernels for MI355X, understanding CDNA4 hardware capabilities, or comparing MI355X vs MI325X/MI300X specs.
---

# AMD CDNA 4 Architecture (MI350 Series, gfx950)

## Package Topology

MI350 Series = 8 XCDs (compute) + 2 IODs (memory/IO), 3D-stacked via advanced packaging.

```
             ┌─── XCD0 ── XCD1 ── XCD2 ── XCD3 ───┐
             │          IOD0 (N6)                   │
   HBM×4 ───┤    Infinity Cache 128MB              ├─── IF Links ×4
             │    Memory Controllers ×4             │
             └──────────── IF ──────────────────────┘
                           │  ~14% faster vs CDNA3
             ┌──────────── IF ──────────────────────┐
             │          IOD1 (N6)                   │
   HBM×4 ───┤    Infinity Cache 128MB              ├─── IF Links ×4
             │    Memory Controllers ×4             │
             │          PCIe Gen5 ×16               │
             └─── XCD4 ── XCD5 ── XCD6 ── XCD7 ───┘
```

| Component | Process | Count | Key change vs CDNA3 |
|-----------|---------|-------|---------------------|
| XCD | TSMC N3P | 8 | N5→N3P, CU redesigned |
| IOD | TSMC N6 | 2 | 4→2, simpler fabric, lower latency |
| HBM3E stacks | — | 8 (12-Hi) | 36GB/stack, 8Gbps |

## XCD Architecture

Each XCD: 36 CUs (4 arrays × 9), **32 active** (4 disabled for yield).

| Resource | Per XCD |
|----------|---------|
| Active CUs | 32 |
| L2 Cache | 4 MB, 16-way SA, 16 channels |
| Instruction Cache | 64 KB per 2 CUs, 8-way SA |
| ACE (Async Compute Engines) | 4 |

Full GPU: 8 XCDs × 32 = **256 CUs**, 1024 Matrix Cores, 16384 Stream Processors.

## Compute Unit (CU) Internals

```
┌─────────────────────────────────────────────┐
│                  CU (CDNA4)                 │
│                                             │
│  Scalar Unit ─── SALU pipeline              │
│  Vector Unit ─── VALU (64-wide SIMD × 4)    │
│  Matrix Cores ── MFMA (2× resources ≤16b)   │
│  Transcendental ─ 2× rate vs CDNA3          │
│                                             │
│  ┌──────────┐  ┌──────────────────────┐     │
│  │ L1 DCache│  │  LDS (160 KB)        │     │
│  │  32 KB   │  │  256 B/clk read BW   │     │
│  │ 128B line│  │  Direct L1→LDS load  │     │
│  │ 64-way SA│  │  Doubled vs CDNA3    │     │
│  └──────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────┘
```

Key CDNA4 CU improvements over CDNA3:
- **Matrix Cores**: 2× execution resources for ≤16-bit types
- **LDS**: 64KB → **160KB**, read BW 128→**256 B/clk**, supports direct L1→LDS load
- **Transcendental**: 2× rate (benefits softmax in attention)
- **New formats**: MXFP8, MXFP6, MXFP4 hardware support
- **TF32**: Removed from hardware, now BF16 software emulation (same model accuracy)

## Numerical Formats

### OCP FP8 (inherited from CDNA3)

| Format | Exponent | Mantissa | Use case |
|--------|----------|----------|----------|
| E5M2 (BF8) | 5 | 2 | Training (wider range) |
| E4M3 (HF8) | 4 | 3 | Inference (more precision) |

### OCP MX (Microscaled, new in CDNA4)

Shared 8-bit exponent per block of 32 elements + per-element compact value.

| Format | Per-element bits | Variants | bytes/elem |
|--------|-----------------|----------|------------|
| MXFP8 | 8 (E5M2 or E4M3) | 2 | 1.25 |
| MXFP6 | 6 (E3M2 or E2M3) | 2 | 1.0 |
| MXFP4 | 4 (E2M1) | 1 | 0.75 |

Micro-scaling advantage: per-32-element scale factor vs per-tensor scale in traditional FP8, enabling much wider applicability across tensors.

## Peak Throughput (MI355X, 2400 MHz)

| Precision | FLOPS/clk/CU | Peak Theoretical | vs MI325X |
|-----------|-------------|-----------------|-----------|
| FP64 Vector | 128 | 78.6 TF | ~1x |
| FP32 Vector | 256 | 157.3 TF | ~1x |
| FP16 Vector | 256 | 157.3 TF | ~1x |
| FP64 Matrix | 128 | 78.6 TF | 0.5x |
| FP32 Matrix | 256 | 157.3 TF | ~1x |
| **FP16 Matrix** | **4096** | **2.5 PF** | **1.9x** |
| **BF16 Matrix** | **4096** | **2.5 PF** | **1.9x** |
| **FP8 Matrix** | **8192** | **5.0 PF** | **1.9x** |
| **INT8 Matrix** | **8192** | **5.0 POPs** | **1.9x** |
| **MXFP6 Matrix** | **16384** | **10 PF** | new |
| **MXFP4 Matrix** | **16384** | **10 PF** | new |

Sparsity (2:4 structured): doubles effective throughput for FP16/BF16/FP8/INT8.

| With Sparsity | Peak |
|---------------|------|
| FP16 Matrix | 5.0 PF |
| FP8 Matrix | 10 PF |

## Roofline Ridge Points

| Precision | Peak TFLOPS | HBM BW | Ridge Point (FLOPs/Byte) |
|-----------|-------------|--------|--------------------------|
| MXFP4 Matrix | 10,100 | 8,000 GB/s | **1,262.5** |
| MXFP6 Matrix | 10,000 | 8,000 GB/s | **1,250** |
| FP8 Matrix | 5,000 | 8,000 GB/s | **625** |
| FP16 Matrix | 2,500 | 8,000 GB/s | **312.5** |
| FP32 Vector | 157.3 | 8,000 GB/s | **19.7** |

## Memory Hierarchy

```
CU L1 DCache: 32 KB, 128B line, 64-way SA (per CU)
      │
XCD L2 Cache:  4 MB, 16-way SA, 16 channels (per XCD, shared by 32 CUs)
      │         128B read / 64B write per channel per cycle
      │         Writeback + write-allocate, can cache non-coherent DRAM data
      │
Infinity Cache: 256 MB total (128MB per IOD), 16-way SA, memory-side cache
      │          16 channels × 64B wide per HBM stack
      │
HBM3E:  288 GB total (36GB × 8 stacks), 8 TB/s peak BW
         1024-bit interface per stack, 8 Gbps
```

| Level | Capacity | Key specs |
|-------|----------|-----------|
| L1 DCache | 32 KB × 256 CUs | Per-CU, 64-way |
| L2 Cache | 4 MB × 8 XCDs = 32 MB | Per-XCD, coherent |
| Infinity Cache | 256 MB | 2 IODs × 128 MB |
| HBM3E | 288 GB | 8 stacks × 36 GB |
| HBM BW | 8 TB/s | 33% more than MI325X (6 TB/s) |

## Communication & Scaling

| Feature | Spec |
|---------|------|
| Infinity Fabric Links | 7 × 16-bit bidirectional (+ 1 for PCIe) |
| Link speed | 38.4 Gbps (20% faster vs CDNA3 32Gbps) |
| Per-link BW | 76.8 GB/s bidirectional |
| Total P2P BW (8 GPUs) | 1,075.2 GB/s |
| Total aggregate IO BW | 1,203.2 GB/s |
| Host | PCIe Gen 5 ×16 (128 GB/s) |
| Node topology | Fully connected 8-GPU (identical to CDNA3) |
| Internal IOD bisection | 5.5 TB/s |

## Partitioning

### Compute Partitions

| Mode | XCDs/Partition | Max Partitions | Memory/Partition (NPS2) |
|------|---------------|----------------|------------------------|
| SPX | 8 | 1 | 288 GB |
| DPX | 4 | 2 | 144 GB |
| QPX | 2 | 4 | 72 GB |
| CPX | 1 | 8 | 36 GB |

### Memory (NUMA) Modes

| Mode | Description | Best for |
|------|-------------|----------|
| NPS1 | Interleave across all 8 HBM stacks | Easy porting, uniform access |
| NPS2 | 2 pools of 144 GB (1 per IOD) | Lower latency, less cross-IOD traffic |

**DPX+NPS2** is the most efficient mode on CDNA4, yielding 7.7× peak compute, 2.25× memory capacity, 2.67× memory BW vs the most efficient CDNA3 mode (QPX+NPS4).

## Product Comparison

| Spec | MI350X | MI355X |
|------|--------|--------|
| Clock | 2,200 MHz | 2,400 MHz |
| Power | 1000W | 1400W |
| Cooling | Air (passive) | Direct Liquid |
| Form factor | OAM (UBB8, 4RU) | OAM (UBB8, 2RU) |
| FP8 Peak | 4.6 PF | 5.0 PF |
| MXFP4 Peak | 9.2 PF | 10 PF |
| Memory | 288 GB HBM3E | 288 GB HBM3E |
| HBM BW | 8 TB/s | 8 TB/s |
| MI325X compatible | Drop-in | Needs higher power/cooling |
| Transistors | 185 Billion | 185 Billion |

## CDNA4 vs CDNA3 Quick Comparison

| Feature | CDNA3 (MI300X/MI325X) | CDNA4 (MI355X) |
|---------|----------------------|----------------|
| XCD process | N5 | **N3P** |
| IOD count | 4 (N6) | **2 (N6)** |
| Active CUs | 304 | **256** |
| MFMA ≤16-bit | 1× | **2×** |
| LDS per CU | 64 KB | **160 KB** |
| LDS read BW | 128 B/clk | **256 B/clk** |
| MX formats | None | **MXFP8/6/4** |
| TF32 | Hardware | **BF16 emulation** |
| HBM | 192/256 GB, 5.3/6.0 TB/s | **288 GB, 8.0 TB/s** |
| FP16 Matrix Peak | 1.3 PF | **2.5 PF** |
| FP8 Matrix Peak | 2.6 PF | **5.0 PF** |
| FP4/FP6 Matrix | N/A | **10 PF** |
| IF link speed | 32 Gbps | **38.4 Gbps** |
| Total P2P BW | 896 GB/s | **1,075 GB/s** |

## Optimization Implications

For kernel developers targeting CDNA4:

1. **Exploit MXFP4/6**: 10 PF peak — use OCP MX quantization wherever model accuracy allows
2. **LDS is now 160 KB**: larger tiles in GEMM, more data reuse before hitting L2
3. **LDS direct load from L1**: reduces VGPR pressure and latency for matrix multiply
4. **2× transcendental rate**: softmax/attention kernels benefit directly
5. **NPS2 preferred**: keep memory traffic within one IOD to reduce cross-IOD hops
6. **Fewer CUs (256 vs 304)**: per-CU throughput is higher; occupancy tuning may differ
7. **FP64 Matrix halved**: scientific HPC workloads using FP64 MFMA see regression; prefer vector FP64

For detailed specs, see the [AMD CDNA 4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf).
