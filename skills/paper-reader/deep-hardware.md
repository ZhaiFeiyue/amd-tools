# deep-hardware.md — Hardware whitepaper / ISA-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **hardware-specific**.

Covers: GPU / accelerator whitepapers, ISA specifications, chiplet
architectures, new data types.

## §1 Architecture overview & naming

- Product name + generation + codename (e.g. MI355X / CDNA4 / gfx950)
- Die / package: chiplet layout, process node, transistor count
- Variants: e.g. MI350X vs MI355X — what's gated on?
- Launch date, competitive positioning (vs H100 / B200 / MI300X)

## §2 Generational delta — 增 / 删 / 改 / 增强

Structured comparison to the previous generation:

| Change type | Subsystem | Detail | Impact |
|---|---|---|---|
| **增** | (new feature added) | ... | ... |
| **删** | (deprecated feature) | ... | ... |
| **改** | (replaced / modified) | ... | ... |
| **增强** | (quantitative improvement) | ... | ... |

Every whitepaper analysis MUST have this table. Without it, the analysis
is a reskin of the spec sheet.

## §3 Compute unit deep dive

For each compute unit (SM / CU / XPU):
- Resource budget: registers / thread, shared memory / smem, LDS size,
  L0/L1 sizes, scheduler slots, warp/wave size
- Matrix core: size, precision support, peak ops per cycle
- Vector ALU: lanes, ops per cycle, supported precisions
- Special function units (SFU): sqrt / rcp / sin throughput
- Async copy / DMA engines: count, modes (bulk, bulk-tensor, remote)

## §4 Peak performance derivation — from first principles

Compute peak FLOPs from:
- Unit count (SMs / CUs)
- Clock (base / boost)
- Ops per cycle per unit per precision

Verify derived number matches the whitepaper's published peak. If they
don't match, explain the discrepancy (boost clock? sparsity multiplier?
structured sparsity assumed?).

Do this for: FP64 / FP32 / TF32 / BF16 / FP16 / FP8 (E4M3/E5M2) / FP6 /
FP4 / INT8 / INT4. Include "dense" and "sparse" variants.

## §5 Memory subsystem

- HBM: type (HBM3 / HBM3E), stack count, capacity per stack, total capacity
- BW: total HBM bandwidth, per-stack BW
- L2: size, banks, bandwidth
- Chiplet-specific: shared LLC vs per-chiplet L2, coherence protocol
- Tiering: any on-package DRAM / HBF / NVM?

## §6 Interconnect & I/O

- Intra-package (chiplet ↔ chiplet): fabric name, BW, latency, protocol
- Inter-GPU (same node): NVLink gen, port count, total BW; or IF AWS-style;
  or PCIe fallback
- Inter-node: embedded NICs? PCIe gen to host?
- I/O lanes: PCIe gen + lane count

## §7 Power, thermal, packaging

- TDP (W), sustained vs peak
- Cooling: air / liquid, required inlet temp
- Physical: form factor (OAM / SXM / PCIe), power delivery

## §8 Software & ISA impact

- ISA changes: new instructions, deprecated instructions (see `isa-reference`)
- Driver / runtime versions required
- Compiler (HIPCC / NVCC / Triton) support status and date
- Backwards compatibility: does existing code run? does it get the new
  speedup automatically or need recompilation?

## §9 AI workload impact analysis

For each workload class, what's the expected speedup and why:
- Large-model training (dense, FP8)
- Large-model training (MoE)
- Long-context prefill
- Batched decode
- Small-model high-concurrency serving
- Fine-tuning / RL (short sequences, high turnover)

## §10 Competitive positioning

Point-by-point comparison table vs the closest competitor (e.g. MI355X vs
B200). At minimum: peak FP8 / peak FP16 / HBM capacity / HBM BW / NVLink
BW / TDP / launch date / $.

## §11 Spec sheet — complete

Single large table with every official number. This is the artifact
engineers copy-paste when sizing a cluster.

## §12 作者证明 — Hardware-specific asks

Hardware whitepapers don't have a "Theorem" — their "proof" is the
performance projection model. Reproduce it:
- Budget equations (compute budget / BW budget / power budget)
- Constraint satisfaction: how the design hits each budget
- Performance projection for representative workloads (matmul N=8192,
  attention with S=2K/8K/32K, etc.) with derivation

If the whitepaper provides no projection numbers (marketing-only), mark
**无形式化作者证明 — 仅广告数字**.

## §13 Hardware → Software forward implication (MANDATORY for hardware)

**This is the reverse of the kernel / cluster SW→HW rule.** What NEW
software can be written because of this HW?

- New kernel patterns enabled (persistent, multi-stream, async-first, etc.)
- New algorithms feasible (MoE with more experts, longer context, larger
  batch)
- Existing kernels that should be rewritten
- Existing libraries that will / must add a new backend (cuDNN /
  rocBLAS / Triton / CUTLASS / CK)
- New deployment topologies (denser racks, liquid cooling requirements)
