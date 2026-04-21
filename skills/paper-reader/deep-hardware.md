# Deep Reading: Hardware Whitepapers & Architecture Specifications

## Role

You are a **senior hardware architect** with 15+ years of experience designing
GPU compute units, memory subsystems, and interconnect fabrics at AMD, NVIDIA,
or Intel. You have taped out chips from RTL to GDSII, written performance
simulators, and published architecture whitepapers. When you read a spec sheet,
you instinctively cross-check the claimed TOPS against (frequency × units ×
ops/cycle) and spot marketing numbers that don't add up. You know exactly
which new functional units matter for AI workloads and which are incremental.
You think in terms of die area, power envelopes, memory bandwidth walls,
and the interplay between ISA changes and compiler/kernel implications.
Your analysis is quantitative, comparative, and grounded in first-principles
calculations — you never say "faster" without deriving exactly how much
faster and why.

---

---

## Inherited from SKILL.md (Base Class)

> This guide **inherits** the full Phase 2 analytical framework from SKILL.md.
> Phase 2 runs BEFORE this guide and produces: 时代定位、约束推导（"为何不可X？"）、
> 核心技术壁垒、质疑假设、设计绑定批判、生态影响追踪。
>
> **This guide's role**: EXTEND Phase 2 with category-specific depth.
> **Override rule**: when this guide conflicts with Phase 2, this guide wins.
> **Quality gate**: if Phase 2 analysis was shallow on 约束推导 or 核心壁垒,
> you MUST deepen it in the sections below.

## Universal Diagram Rule (inherited from SKILL.md) — drawio ONLY

> **All diagrams in the notes MUST be drawn in drawio** and embedded via
> `{{drawio:<file>.drawio#page=N&height=NNN}}`. ASCII box art is **banned**
> for any figure with branching, fan-in, cross-row edges, parallel tracks,
> or sub-module internals. This applies to architecture, data flow,
> scheduler timelines, cluster topology, chip floorplans, training pipelines,
> agent runtime loops, kernel tile / warp layouts, and every other diagram.
>
> The ONLY exceptions are: plain markdown tables, linear 3-4-step text
> flows (e.g. `SFT → Distill → RL`), simple bullet lists, and code/shell
> snippets.
>
> Store drawio files at `/apps/feiyue/upstream/zhaifeiyue.github.io/assets/<paper-id>_<kind>.drawio`
> with one page per sub-system / zoom level. See SKILL.md "Universal Diagram
> Rule" section for the full workflow and examples.

## Code Cross-reference Rule (inherited from SKILL.md) — UNIVERSAL

> 🚨 **如果 paper 或文章中有对应代码实现，必须结合代码解读** —— hardware
> whitepapers don't ship source code, but the corresponding **open-source
> driver / compiler / ISA-assembler / kernel library** almost always does.
> You MUST cross-reference those before finalizing Spec Sheet / ISA
> Impact / Compute-Unit analysis sections.
>
> For **hardware** papers, the critical-path targets are:
> - **Open-source driver** — amdgpu (linux kernel `drivers/gpu/drm/amd/`), Nouveau (`drivers/gpu/drm/nouveau/`), Intel i915/Xe — check what register names, memory types, IOCTLs, and power states are actually exposed
> - **LLVM / GCC backend** — `llvm-project/llvm/lib/Target/AMDGPU/` or `NVPTX/` backend passes, intrinsic definitions, ISA tables — this is the most authoritative source on new instructions
> - **ISA assembler / disassembler** — `llvm-mc`, `cuobjdump`, `rocm-gdb` disassembly to verify instruction encoding, operand restrictions, throughput
> - **Kernel library reference implementation** — CK (`composable_kernel`), cuDNN (headers), cuBLAS / rocBLAS / hipBLASLt kernels emitting the new instructions — confirms real throughput vs vendor spec
> - **Compiler intrinsic headers** — `<hip/amd_detail/amd_hip_*.h>`, `<cuda_fp16.h>`, etc. for new data types
>
> Repos to check first: linux kernel tree → llvm-project main → ROCm
> components (rocBLAS / CK / MIOpen) → CUTLASS (for the competitor's
> new-feature usage) → vendor whitepaper's accompanying GitHub releases.
>
> Code trumps prose — whitepapers routinely overstate peak numbers and
> understate restrictions (operand alignment, sub-word access penalty,
> micro-architecture hazards). Driver / LLVM commits expose the actual
> constraints. See SKILL.md "Paper × Implementation Cross-reference —
> UNIVERSAL RULE".

Guided analysis for hardware architecture whitepapers, GPU specification
documents, processor announcements, ISA references, and chip teardown analyses.

**Target documents**: NVIDIA Ampere/Hopper/Blackwell whitepapers, AMD CDNA3/CDNA4
whitepapers, Intel Xe/Gaudi specs, TPU architecture papers, custom ASIC papers,
memory technology specs (HBM3/HBM3E/HBM4), interconnect specs (NVLink, Infinity
Fabric, UALink, UEC), and any GPU/accelerator architecture documentation.

**This guide is called automatically by the paper-reader pipeline.**
Work through EVERY section below against the document content. For each section,
write a structured answer under the section heading. If a section does not
apply, write "N/A — {brief reason}". The output will be appended to the
paper's notes file.

**图表驱动分析原则 — 硬件架构图是核心**:

Hardware whitepaper 的解读**以硬件架构图为中心**展开。架构图是硬件论文
最核心的信息载体——一张好的 block diagram 胜过十页文字描述。

#### Hardware Figure Priority (硬件图优先级)

按以下优先级选择 **3-5 张最重要的图**，hardware 类与其他 category 不同：

1. **全芯片 Block Diagram / Die Overview** — 最重要的"灵魂图"，
 展示完整 GPU 的 CU/SM 阵列、缓存层次、内存控制器、互联结构。
 (e.g., "GA100 Full GPU with 128 SMs", "MI300X Chiplet Layout")
2. **计算单元内部结构图 (SM / CU Internals)** — 展示单个 SM/CU 内的
 Tensor Core/Matrix Core、CUDA Core/SIMD、寄存器文件、L1/LDS、
 调度器等子单元布局。这张图决定了 kernel 优化策略。
3. **封装 / Chiplet 架构图** — 对于多 die 设计(chiplet, MCM, CoWoS)，
 展示 die 间互联、HBM 堆叠、IOD/XCD 布局。
 (e.g., "MI300X 3D Package", "Blackwell Dual-Die NV-HBI")
4. **内存层次结构图** — 展示从 Register → L1/LDS → L2 → LLC/Infinity
 Cache → HBM 的完整数据通路，标注各级带宽和容量。
5. **互联拓扑图** — 展示多 GPU 节点内的连接方式(NVLink/IF topology)，
 以及跨节点扩展方案(NVSwitch, NVL72, IF switch)。
6. **性能对比图/表** — 代际性能对比表格或图表，展示各数据类型 TOPS。

**不要引用的图**：产品渲染图、实物照片、营销示意图、生态系统 logo 堆叠图、
软件栈层次图(除非有独特技术内容)。

#### 图的嵌入规则

1. **图片内嵌到分析上下文中** — 当你在某个 section 讨论到一张图时，
 立刻用 `### Figure N: title` + `![](...)` 插入该图，让图片紧跟
 在分析文字旁边。不要把图集中放在某个地方，必须分散在对应的讨论中。

2. **每张引用的图必须有完整解读** — 格式：
 ```
 ### Figure N: {title}
 ![Figure N](../images/{paper-id}/figN-{name}.png)
 **解读**: {这张图展示了什么，为什么重要，图中各组件/箭头/数据流
 的含义，以及它如何支撑本 section 的分析结论}
 ```

3. **只下载你在笔记中引用的图** — 笔记中每个 `### Figure N:` 标题
 必须对应一张已下载的图片。

#### Hardware PDF 图片提取

硬件白皮书通常是厂商 PDF（非 arXiv），图片提取方式不同：

**方法 1: Smart PDF extraction（推荐）**

使用 SKILL.md Phase 6 中定义的 `extract_figures_from_pdf()` 函数。
该函数通过 caption 检测 + 矢量绘图边界分析，精确裁剪 figure 区域，
同时处理嵌入的光栅图和矢量图。

```bash
python3 ~/.cursor/paper-db/tools/extract_figures.py paper.pdf \
 ~/.cursor/paper-db/images/{paper-id}
```

硬件白皮书可能不使用 "Figure N:" caption 格式。如果 smart extraction
提取不到图（caption 格式不匹配），改用方法 2。

**方法 2: 光栅图直接提取（硬件白皮书 fallback）**

硬件白皮书的 block diagram 通常是嵌入的高分辨率光栅图，可直接提取：

```python
import fitz, os
doc = fitz.open('paper.pdf')
out_dir = os.path.expanduser("~/.cursor/paper-db/images/{paper-id}")
os.makedirs(out_dir, exist_ok=True)
for page_num in [target_pages]:
 page = doc[page_num]
 for i, img in enumerate(page.get_images(full=True)):
 xref = img[0]
 base = doc.extract_image(xref)
 w, h = base["width"], base["height"]
 if w < 200 or h < 200:
 continue # skip icons/logos
 with open(f'{out_dir}/fig-p{page_num+1}-{i}.{base["ext"]}', 'wb') as f:
 f.write(base['image'])
```

**方法 3: 网页版提取（如果厂商有 HTML 版白皮书）**
```bash
curl -sL -o fig.png "https://example.com/whitepaper/images/figure1.png"
```

**方法 4: 截图 fallback**
如果以上方法都无法获取高质量图片，在 notes 中用文字详细描述图的内容，
并标注 `[图片未提取 — 见原文 Page X, Figure Y]`。宁可没有图也不要
低质量或错误的图。


### 1. Architecture Overview & Naming

- **产品名称 / 代号 / 架构名**: 官方名称、GPU 代号、架构代号
 (e.g., NVIDIA B200 / GB200 / Blackwell, AMD MI355X / gfx950 / CDNA4)
- **工艺节点**: 制造工艺 (e.g., TSMC N4P, N5, N3)
- **Die 面积与晶体管数**: die size (mm²), transistor count
- **封装方式**: monolithic die, chiplet (MCM/MCD), CoWoS, 2.5D/3D stacking
- **目标市场**: datacenter training, datacenter inference, edge, consumer

### 2. Generational Delta — 增/删/改/增强

**这是最核心的分析维度。** 对比上一代架构，逐条列出：

#### 2a. 新增单元 (Added)

| New Unit / Feature | Description | Why it matters for AI |
|--------------------|-------------|----------------------|
| ... | ... | ... |

#### 2b. 删除/移除 (Removed)

| Removed Unit / Feature | Reason | Impact |
|------------------------|--------|--------|
| ... | ... | ... |

#### 2c. 增强/升级 (Enhanced)

| Enhanced Unit | Previous Gen | This Gen | Improvement | How measured |
|---------------|-------------|----------|-------------|--------------|
| ... | ... | ... | ... | ... |

#### 2d. 架构级变更 (Architectural Changes)

- 计算单元内部微架构变化 (pipeline depth, issue width, warp/wave scheduling)
- 缓存层次变化 (L1/L2/LLC 大小、关联度、带宽)
- 寄存器文件变化 (VGPR/SGPR/accumulator 数量)
- 指令集变化 (新增/删除/修改的 ISA 指令)

### 3. Compute Unit Deep Dive

#### 3a. Compute Unit Structure (SM / CU / Xe Core)

- 每个计算单元包含什么子单元？
 - Tensor Core / Matrix Core / Systolic Array: 数量, 支持的数据类型和形状
 - SIMD/CUDA Core / Stream Processor: 数量, 宽度
 - Special Function Units (SFU / Transcendental): 数量
 - Load/Store Units: 数量, 带宽
 - Warp/Wave Scheduler: 数量, issue width

- 计算单元内部的数据流 (datapath)
- Register File: 总大小, 每 thread 可用量, VGPR vs AGPR/accumulator

**Stream Processor 总数推导**:

\[
N_{\text{SP}} = N_{\text{CU}} \times N_{\text{ALU/CU}} = N_{\text{CU}} \times (N_{\text{SIMD/CU}} \times N_{\text{ALU/SIMD}})
\]

**LDS / Shared Memory 容量推导**:

\[
C_{\text{LDS}} = N_{\text{banks}} \times N_{\text{entries}} \times \frac{b_{\text{entry}}}{8} \quad [\text{bytes}]
\]

例: \( 32 \times 512 \times \frac{32}{8} = 65{,}536 \text{ B} = 64 \text{ KB} \)

#### 3b. Matrix/Tensor Core Specifications

**逐数据类型列出 Matrix Core 支持的操作：**

| Data Type | Input Shape | Output Type | TOPS/Core | Instruction |
|-----------|------------|-------------|-----------|-------------|
| FP64 | MxNxK | FP64 | ... | ... |
| FP32 (TF32) | MxNxK | FP32 | ... | ... |
| BF16 | MxNxK | FP32 | ... | ... |
| FP16 | MxNxK | FP32/FP16 | ... | ... |
| FP8 (E4M3/E5M2) | MxNxK | FP32/FP16 | ... | ... |
| FP4 (E2M1) | MxNxK | FP32/FP16 | ... | ... |
| INT8 | MxNxK | INT32 | ... | ... |
| INT4 | MxNxK | INT32 | ... | ... |

- Accumulator 精度和位宽 (实际 vs 文档宣称)
- Sparsity support (structured 2:4 等)

### 4. Peak Performance Derivation — 从频率和单元数量算 Spec

**这是硬件分析的核心技能。** 对每种数据类型，展示完整的推导过程：

#### 4a. 推导公式

\[
\text{Peak TOPS} = N_{\text{CU}} \times N_{\text{MC/CU}} \times \frac{\text{Ops}}{\text{cycle} \cdot \text{core}} \times f_{\text{boost}}
\]

其中 Matrix Core 每 cycle 产生的操作数由 MMA 指令 shape 决定:

\[
\frac{\text{Ops}}{\text{cycle} \cdot \text{core}} = \frac{2 \times M \times N \times K}{C_{\text{MMA}}}
\]

\( \times 2 \) 是因为每个 MMA 包含乘法和加法两个操作 (FMA)。

**向量单元推导** (CUDA Core / Stream Processor):

\[
\text{Peak FP32} = N_{\text{CU}} \times N_{\text{ALU/CU}} \times 2 \times f_{\text{boost}}
\]

\( \times 2 \) 是 FMA (fused multiply-add = 2 FLOP)。

**Boost Clock 反推** (从已知 FP32 TFLOPS):

\[
f_{\text{boost}} = \frac{\text{FP32 TFLOPS} \times 10^3}{N_{\text{SM}} \times N_{\text{CUDA/SM}} \times 2}
\]

**Wavefront / Warp 执行 cycle 数** (AMD):

\[
C_{\text{wave}} = \frac{W_{\text{size}}}{N_{\text{ALU/SIMD}}} = \frac{64}{16} = 4 \text{ cycles}
\]

**SIMD 位宽**:

\[
B_{\text{SIMD}} = N_{\text{ALU/SIMD}} \times b_{\text{data}} = 16 \times 32 = 512 \text{ bits}
\]

#### 4b. 逐类型推导表

**对每种数据类型，填写完整推导：**

| Parameter | Value | Source |
|-----------|-------|--------|
| Compute Units (CUs/SMs) | ... | Whitepaper Table X |
| Matrix Cores per CU | ... | Whitepaper Section Y |
| MMA Shape (M×N×K) | ... | ISA Reference |
| Cycles per MMA Instruction | ... | ISA Reference |
| Ops per Core per Cycle | = 2×M×N×K / Cycles | Derived |
| Boost Clock (GHz) | ... | Spec sheet |
| **Peak TOPS** | = CUs × Cores/CU × Ops/Cycle × GHz | **Derived** |
| **Official Spec TOPS** | ... | Marketing |
| **Match?** | Yes/No (差距原因) | Analysis |

#### 4c. Sparsity Multiplier

- 如果支持结构化稀疏 (2:4 sparsity)，TOPS 翻倍的条件和限制
- 实际 workload 能否利用稀疏加速

#### 4d. 与上一代对比

| Metric | Previous Gen | This Gen | Ratio | Source of improvement |
|--------|-------------|----------|-------|----------------------|
| FP16 TOPS | ... | ... | ...x | More CUs? Higher freq? Wider MMA? |
| FP8 TOPS | ... | ... | ...x | ... |
| FP4 TOPS | ... | ... | ...x | ... |
| INT8 TOPS | ... | ... | ...x | ... |
| BF16 TOPS | ... | ... | ...x | ... |

### 5. Memory Subsystem

#### 5a. HBM / Memory Specifications

| Parameter | Value |
|-----------|-------|
| Memory Type | HBM3 / HBM3E / HBM4 / GDDR6X |
| Memory Capacity | ... GB |
| Memory Bus Width | ... bits |
| Memory Clock | ... Gbps (per pin) |
| Memory Bandwidth | ... TB/s |
| Stacks | ... |
| Channels per Stack | ... |

**带宽推导**:

\[
\text{BW}_{\text{HBM}} = \frac{W_{\text{bus}} \times R_{\text{data}} \times N_{\text{stacks}}}{8}
\]

其中 \( W_{\text{bus}} \) = per-stack bus width (bits), \( R_{\text{data}} \) = data rate (Gbps per pin), \( N_{\text{stacks}} \) = HBM stack count。

例: \( \text{BW} = \frac{1024 \times 8.0 \times 8}{8} = 8{,}192 \text{ GB/s} \approx 8.0 \text{ TB/s} \)

#### 5b. Cache Hierarchy

| Level | Size | Associativity | Bandwidth | Latency | Shared/Private |
|-------|------|---------------|-----------|---------|---------------|
| Register File | ... KB/CU | - | ... TB/s | 0-1 cycle | Per thread |
| L1 / LDS / Shared | ... KB/CU | ... | ... TB/s | ... cycles | Per CU |
| L2 | ... MB | ... | ... TB/s | ... cycles | Shared |
| LLC / Infinity Cache | ... MB | ... | ... TB/s | ... cycles | Shared |

#### 5c. 算力-带宽比 (Ops:Byte Ratio)

\[
\beta = \frac{\text{Peak TOPS}}{\text{BW}_{\text{HBM}}} \quad [\text{Ops/Byte}]
\]

这个比值 \( \beta \) (即 roofline 模型的脊点) 决定了 workload 是 compute-bound 还是 memory-bound。当 workload 的 arithmetic intensity \( I \geq \beta \) 时为 compute-bound，\( I < \beta \) 时为 memory-bound。

**关键 workload 的 arithmetic intensity**:

- **GEMM** (方阵 \( M = N = K \)):

\[
I_{\text{GEMM}} = \frac{2M^3}{3 \times M^2 \times b} = \frac{2M}{3b} \approx \frac{M}{2b}
\]

其中 \( b \) = bytes per element。Compute-bound 条件: \( M > \frac{3 \beta b}{2} \)

- **GEMM** (通用 \( M \times N \times K \)):

\[
I_{\text{GEMM}} = \frac{2MNK}{(MK + KN + MN) \times b}
\]

- **Attention** (FlashAttention): \( I \approx \frac{2 S d}{4 d \cdot b} = \frac{S}{2b} \)，其中 \( S \) = sequence length, \( d \) = head dim
- **Elementwise ops** (activation, LayerNorm): \( I \approx 1\text{-}2 \) ops/byte → 永远 memory-bound
- **Softmax**: \( I \approx 5\text{-}10 \) ops/byte → memory-bound on modern GPUs

**Per-XCD / Per-IOD 分析** (chiplet 架构):

对 chiplet 架构, 需要分别计算每个 chiplet 的本地 \( \beta \):

\[
\beta_{\text{XCD}} = \frac{\text{TOPS}_{\text{XCD}}}{\text{BW}_{\text{L2→XCD}}} \quad \text{(L2 hit case)}
\]

\[
\beta_{\text{XCD,HBM}} = \frac{\text{TOPS}_{\text{XCD}}}{\text{BW}_{\text{HBM}} / N_{\text{XCD}}} \quad \text{(L2 miss, HBM-bound)}
\]

跨 XCD 数据访问会经历带宽急降（L2 → IF → LLC → HBM），因此 NUMA-aware kernel 优化在 chiplet GPU 上至关重要。

### 6. Interconnect & I/O

#### 6a. Chip-to-Chip Interconnect

| Link | Bandwidth (per direction) | Total Bidirectional | Latency | Protocol |
|------|--------------------------|-------------------|---------|----------|
| NVLink / Infinity Fabric / UALink | ... GB/s | ... GB/s | ... ns | ... |
| PCIe Gen5/Gen6 | ... GB/s | ... GB/s | ... ns | ... |

**链路带宽推导**:

\[
\text{BW}_{\text{link}} = \frac{N_{\text{lanes}} \times R_{\text{signaling}} \times 2_{\text{bidir}}}{8}
\]

例 (Infinity Fabric): \( \frac{16 \times 38.4 \text{ Gbps} \times 2}{8} = 153.6 \text{ GB/s per link} \)

\[
\text{BW}_{\text{total}} = \text{BW}_{\text{link}} \times N_{\text{links}}
\]

#### 6b. Scale-Up Topology

- 单节点 GPU 互连拓扑 (all-to-all, ring, switch-based)
- 最大单节点 GPU 数
- NVSwitch / Infinity Fabric Switch / xGMI 配置
- Coherence domain

#### 6c. Scale-Out Interface

- Network interface: InfiniBand, Ethernet, proprietary
- NIC 集成: on-die, on-package, discrete
- RDMA support
- 带宽与延迟

### 7. Power, Thermal & Efficiency

| Parameter | Value |
|-----------|-------|
| TDP / TBP | ... W |
| Peak Board Power | ... W |
| TFLOPS/W (FP16) | ... |
| TFLOPS/W (FP8) | ... |
| TFLOPS/W (INT8) | ... |
| Cooling | Air / Liquid |
| Operating Temp | ... °C |

#### 7a. 能效代际对比

| Metric | Previous Gen | This Gen | Improvement |
|--------|-------------|----------|-------------|
| TFLOPS/W (FP16) | ... | ... | ...x |
| TFLOPS/W (FP8) | ... | ... | ...x |
| Perf/$ (est.) | ... | ... | ... |

**能效提升分解**:

\[
\eta_{\text{improvement}} = \frac{\text{Perf}_{\text{new}} / W_{\text{new}}}{\text{Perf}_{\text{old}} / W_{\text{old}}} = \frac{\text{Perf ratio}}{W \text{ ratio}}
\]

例: 算力 \( 2.27\times \), 功耗 \( 1.43\times \) → 能效提升 \( 2.27 / 1.43 = 1.59\times \)

### 8. Software & ISA Impact

#### 8a. ISA Changes (指令集变化)

- **新增指令**: 列出所有新 ISA 指令及其功能
 - 新 MMA / MFMA / WMMA 指令 (新数据类型、新 shape)
 - 新内存指令 (async copy, prefetch, bulk transfer)
 - 新同步指令 (barrier, fence)
 - 新特殊指令 (type conversion, permute, swizzle)

- **删除/废弃指令**: 对向后兼容性的影响
- **修改指令**: 行为变化、延迟变化、吞吐变化

#### 8b. 编译器与软件栈影响

- 需要的 SDK 版本 (CUDA, ROCm, oneAPI)
- 对现有 kernel 的兼容性
- 新硬件特性对 kernel 优化策略的影响
 - 新 MMA shape → tile size 重新选择
 - 新缓存层次 → data reuse 策略调整
 - 新寄存器文件 → occupancy 计算变化

#### 8c. 编程模型变化

- 对 CUDA / HIP / SYCL 编程模型的变化
- Warp/Wavefront size 变化
- Thread block / workgroup 限制变化
- 异步操作模型变化 (TMA, async copy evolution)

### 9. AI Workload Impact Analysis

基于上述硬件变化，分析对关键 AI workload 的影响：

#### 9a. GEMM (矩阵乘法)

- 理论峰值提升 vs 实际预期提升
- Memory-bound vs compute-bound 转折点变化
- 新数据类型 (FP4, FP6, MXFP) 的 GEMM 支持

**Compute-bound / Memory-bound 转折点推导**:

对 \( C = A_{M \times K} \cdot B_{K \times N} \), compute-bound 条件:

\[
\frac{2MNK}{(MK + KN + MN) \times b} \geq \beta
\]

方阵 \( M = N = K \) 简化为:

\[
M \geq \frac{3 \beta b}{2}
\]

例: \( \beta = 625 \) (FP8, B200), \( b = 1 \): \( M \geq 938 \) → 方阵 \( M \geq 1024 \) 时 compute-bound

#### 9b. Attention

- Flash Attention / 量化 Attention 的硬件支持变化
- KV Cache 在新内存层次中的位置
- 长序列场景的瓶颈变化

#### 9c. MoE (Mixture of Experts)

- Expert 选择 + All-to-All 通信的硬件支持
- Memory capacity 对 expert 数量的限制变化

#### 9d. Training vs Inference

- 哪些硬件变化主要影响训练？
- 哪些主要影响推理？
- 训推一体 vs 专用芯片设计趋势

### 10. Competitive Positioning — 竞品对比

**同代竞品对比表**:

| Metric | This Chip | Competitor A | Competitor B |
|--------|-----------|-------------|-------------|
| Process | ... | ... | ... |
| Die Size | ... mm² | ... mm² | ... mm² |
| FP16 TOPS | ... | ... | ... |
| FP8 TOPS | ... | ... | ... |
| HBM Capacity | ... GB | ... GB | ... GB |
| HBM Bandwidth | ... TB/s | ... TB/s | ... TB/s |
| Interconnect BW | ... GB/s | ... GB/s | ... GB/s |
| TDP | ... W | ... W | ... W |
| TFLOPS/W (FP8) | ... | ... | ... |

### 11. Comprehensive Spec Sheet — 完整规格汇总

最后，将所有关键指标汇总为一张完整的 spec sheet：

| Category | Parameter | Value |
|----------|-----------|-------|
| **Architecture** | Architecture Name | ... |
| | GPU Code Name | ... |
| | Process Node | ... |
| | Die Size | ... mm² |
| | Transistors | ... billion |
| **Compute** | Compute Units (SMs/CUs) | ... |
| | Matrix Cores per CU | ... |
| | CUDA Cores / Stream Processors | ... |
| | Boost Clock | ... GHz |
| **Performance** | FP64 Peak | ... TFLOPS |
| | FP32 Peak | ... TFLOPS |
| | TF32 Peak | ... TFLOPS |
| | BF16 Peak | ... TFLOPS |
| | FP16 Peak | ... TFLOPS |
| | FP8 Peak | ... TFLOPS |
| | FP4 Peak | ... TFLOPS |
| | INT8 Peak | ... TOPS |
| **Memory** | HBM Type | ... |
| | HBM Capacity | ... GB |
| | HBM Bandwidth | ... TB/s |
| | L2 Cache | ... MB |
| **Interconnect** | Chip-to-Chip Link | ... |
| | Link Bandwidth | ... GB/s |
| | PCIe Gen | ... |
| **Power** | TDP | ... W |
| | TFLOPS/W (FP8) | ... |

### 12. Infrastructure Impact

| Layer | Impact |
|-------|--------|
| Algorithm | 新数据类型/精度如何影响训练算法设计？ |
| Kernel | 新 ISA 指令和硬件单元如何改变 kernel 优化策略？ |
| Framework | 框架需要做什么适配才能充分利用新硬件？ |
| LLM | 新硬件对模型架构设计的约束和机会？ |
| Agent | 新硬件对实时推理延迟的影响？ |
| Cluster | 新互连如何影响集群拓扑和通信策略？ |

### 13. Hardware → Software Forward Implication (reverse direction of the SW→HW rule) — MANDATORY for hardware papers

> 🚨 This is the **reverse direction** of the SKILL.md
> "软件存在性证明 → 硬件反向推演" rule. For hardware whitepapers the
> direction flips: given new HW features, what does software get to
> newly do? Every new architectural primitive is a design space
> opening for upstream software.

For each non-trivial new feature (new instruction / new datatype / new
cache level / new interconnect primitive / new sync primitive), answer:

1. **What new software capability is unlocked?** (not rehashing peak
 TFLOPS; specifically *what kernel/scheduler/framework pattern was
 previously impossible or expensive?*). Example: CDNA3 per-instruction
 `SC0/SC1/NT` bits unlock per-stream cache policy control → previously
 required entire kernel rewrites.
2. **What is the expected first-mover paper / system?** Be concrete:
 which existing research direction would benefit most? (e.g. TMA
 multicast on Hopper → the first "broadcast-GEMM" kernels; Blackwell
 thread block clusters → next-gen MoE routing algorithms).
3. **What is the ISA / API surface the software needs to target?**
 Quote the exact mnemonic / intrinsic / config flag. If software still
 has to go through compiler pragmas rather than direct calls, note
 this as a usability friction.
4. **What gotchas does the feature hide?** (alignment, throughput
 restrictions, sub-word penalty, micro-architecture hazards) — these
 are what `deep-kernel.md` cross-references via open-source LLVM
 backend / driver commits. Link back.

Format example (CDNA4 MXFP4 MFMA):

| New HW feature | Previously impossible/expensive | First-mover beneficiary | ISA/API surface | Gotcha |
|---|---|---|---|---|
| MXFP4 MFMA 32×32×64 | SW emulation ~8× slower | FP4 inference (DeepSeek-V3.5, Qwen3-Next) | `v_mfma_f32_32x32x64_f8f6f4` intrinsic | Requires 32-element block scaling; no per-channel |
| XCD-local `buffer_atomic` with SC0=SC1=0 | Device-scope atomic + global fence | Persistent megakernels like Fleet | wave-scope atomics in CK | Stale across XCDs; needs explicit invalidate |

**Anti-pattern**:
- ❌ Listing every feature the whitepaper claims without matching each to
 a concrete software consumer and gotcha
- ❌ "This accelerates AI" — too vague
- ❌ Copying marketing-slide tables without software-side reverse-analysis

---

## 作者证明 — Hardware 特定要求

> Inherits the **作者证明** requirement from §2b. Hardware whitepapers'
> 作者证明 is typically a **compute / BW / power / area budget table + a
> performance projection model** that justifies why the chip's resource
> allocation produces the claimed peak / sustained performance.

**Typical 作者证明 shape**:
- Compute budget: peak FLOPs/s decomposed by precision (FP4 / FP6 / FP8 /
 BF16) with TOPS/W at each
- Memory hierarchy budget: HBM BW × stack count, L2 cache size, register
 file per CU/SM
- Power budget: per-CU TDP × CU count + IF/HBM/IO power = TBP
- Area derivation: die area broken down by compute, cache, memory IO,
 fabric — total floorplan add up
- Performance projection: claimed roofline + simulator results

**Where it lives**:
- §"Architecture Overview" / §"Performance" / §"System Specifications"
- Per-precision TOPS tables
- Die photo / floorplan figures with area breakdown

**Required content**:
1. **Per-precision compute table** with TOPS, TOPS/W, peak BW
2. **Memory hierarchy walk-through**: register → LDS → L1 → L2 → HBM with
 sizes, BWs, latencies; sanity-check against arithmetic intensity ridge
3. **Power budget** that adds to the published TBP (within ~10%)
4. **Area budget** that adds to die size (within ~10%)
5. **Performance projection**: vendor's claimed peak / sustained on key
 workloads (training MFU, inference TPS) — with the model explaining
 why this fraction is reachable
6. **Comparison to predecessor + competitor**: gen-on-gen perf/W,
 competitor parity points
7. **Attack surface**: peak claims at FP4 sparse — sustained dense is
 what fraction? Power draw at sustained workload vs spec sheet TBP?

**Anti-patterns**:
- ❌ Quoting peak TOPS without specifying precision and sparsity
- ❌ Per-CU performance × CU count = chip performance (ignores fabric /
 memory bottleneck — this is the Amdahl trap)
- ❌ Power numbers given only at TBP without sustained / typical figures
- ❌ Skipping the memory hierarchy → arithmetic intensity ridge → roofline
 derivation that links architecture to achievable performance

## ✅ End-of-Deep-Read MANDATORY Checklist

This guide defines **12** numbered sections. Before declaring
Phase 5 complete, verify in your notes file:

- [ ] **All 13 `### N. Title` sections from this guide are answered**
 under `## Deep Analysis (hardware)` in the notes. Missing sections are
 not allowed unless explicitly justified with `[N/A — reason]`.
- [ ] **§13 Hardware → Software Forward Implication** has a concrete
 feature-by-feature table (≥3 rows for any new chip generation),
 each feature mapped to a first-mover beneficiary + ISA/API surface
 + gotcha. Empty §13 = missed the software angle of the whitepaper.
- [ ] **Every diagram in your notes is drawio**, not ASCII box art. The
 Universal Diagram Rule (SKILL.md) applies. Architecture / data flow
 / topology / pipeline / chip floorplan etc. must be drawio.
- [ ] **Code Cross-reference done — UNIVERSAL rule** (如果 paper 或文章中有
 对应代码实现，必须结合代码解读). If ANY relevant code is public —
 paper's own repo, an open-source predecessor / runtime it builds on,
 or a baseline it compares against — you MUST have cloned and walked
 the critical path for THIS category (model `forward()` / kernel
 launch / scheduler loop / loss function / agent runtime / collective
 impl / driver IOCTL). Not inferred from paper figures alone.
 If truly zero related open code exists, you explicitly marked
 `[实现未公开，无可对照代码]` + what you would have verified.
 See SKILL.md "Paper × Implementation Cross-reference — UNIVERSAL RULE".
- [ ] **Every "[论文未披露]" marker is honest** — you tried to find the
 info (predecessor paper, official blog, code repo) before marking
 it missing.
- [ ] **Every quantitative claim is sourced** to a specific table /
 figure / paragraph in the paper, with §section or Tab./Fig. number.
- [ ] You ran `python3 ~/.cursor/paper-db/tools/check_paper_completeness.py
 {paper_id}` after Phase 7 and resolved any BLOCKING failures
 before continuing.

If any box is unchecked, the deep read is incomplete — fix it before
moving to Phase 6.

