# deep-kernel.md — Kernel-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **kernel-specific**.

## §3 Target operation precisely defined

Before writing, lock down:
- Input / output tensor shapes and dtypes (with the specific regime, e.g.
  "decode: [B, 1, H, D]" vs "prefill: [B, S, H, D]")
- Sparsity / causal mask structure
- Numerical precision contract: input dtype, accumulator dtype, output
  dtype, rounding mode (RNE / stochastic)

## §3 Hardware model — concrete numbers

Pin down the target hardware and its relevant peak numbers:
- GPU: model (H100 / MI300X / MI355X / B200) + SM or CU count
- Matrix core peak (FLOPs at the kernel's precision)
- HBM bandwidth (bytes/s), L2 size + bandwidth, shared memory / LDS size
  per SM/CU, register file size
- Launch config: grid × block, waves per SM/CU, smem/register budget
  per wave

Record **file:line** of the kernel in the paper's repo or the closest
open-source equivalent (CUTLASS / CK / FlashAttention / Triton / HipKittens).

## §4 作者证明 — Kernel-specific asks

In addition to the 6 base checks (SKILL.md):

- **Roofline placement**: compute arithmetic intensity (FLOPs / Bytes) of
  the target op; mark where it lands on the HW's roofline. Is the kernel
  compute-bound or memory-bound?
- **% peak derivation**: reproduce the measured throughput as % of roofline
  peak. If they claim X% peak, derive it from launched waves × utilization,
  not just divide throughput by peak.
- **Why this tile / launch is optimal**: show register pressure budget,
  smem occupancy, wave scheduling. Explain why a 2× larger tile doesn't
  fit and a 2× smaller tile leaves peak on the table.

## §4 Design space & constraint derivation (kernel-specific)

List the main optimization axes (tiling, pipelining, swizzling, split-K,
software pipeline depth, async copies, matrix-core variant, epilogue
fusion). For each, state what value was chosen and why alternatives fail.
Build a feasibility matrix:

| Axis | Value chosen | Rejected alternatives | Constraint that blocks them |
|---|---|---|---|

## §5 Optimization techniques inventory

For each technique in the paper, record:
- What (tiling / pipelining / swizzling / async copy / warp-specialization
  / persistent kernel / multi-buffering)
- Target bottleneck it attacks
- Hardware primitive it relies on (wgmma / mfma / cp.async / tma /
  ldmatrix / ds_read)
- Measured contribution (speedup with / without — must be in the ablation)

## §5 Numerical considerations

- Accumulator precision (FP32 / FP16 / BF16)
- Loss of accuracy at edge-case inputs (overflow / underflow / denormals)
- Comparison with reference (max abs diff, mean rel diff on a standard
  test set)
- FP8 / FP4 scaling strategy if applicable (per-tensor vs per-block vs
  per-channel)

## §7 Portability analysis

- Which GPU families does the kernel support? (sm_80 / sm_90 / gfx942 / gfx950)
- Instruction dependencies that block porting (wgmma is H100+, mfma is
  CDNA, ds_read_b128 is gfx9+)
- What would a CK / Triton port look like? Which optimization survives,
  which doesn't?

## §8 Source walkthrough (mandatory if code public)

Minimum 3 file:line citations:
1. The kernel entry (launcher / host stub)
2. The inner loop (tile iteration or main mma loop)
3. The epilogue (write-back + fused ops)

Explain the critical 10–30 lines with inline comments. Skip the boilerplate.

## §9 Software → Hardware reverse implication (MANDATORY for kernel)

What does this kernel's optimization reveal about what future hardware
should provide?

- Missing instructions / cache scope / sync primitive that would make
  this simpler
- Occupancy / register pressure tension that suggests a larger register
  file or different SM/CU split
- Memory-hierarchy demand that suggests more L2 / shared scratchpad /
  async-copy engines
- Concrete proposal: "If the next-gen ISA had X, this kernel would
  shed Y lines / gain Z% speedup"
