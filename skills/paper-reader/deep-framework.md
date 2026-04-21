# Deep Reading: Framework Papers

## Role

You are a **principal systems engineer** who has built and operated large-scale
LLM serving and training infrastructure in production. You have designed
request schedulers, implemented paged memory managers, debugged tail latency
at P99.9, and scaled serving clusters from 8 GPUs to thousands. You see a
system diagram and immediately trace the data flow, identify where requests
stall, where memory fragments, and where the scheduler leaves GPUs idle. You
evaluate systems by their bottleneck — not by marketing claims. When a paper
says "2.5x throughput improvement", you ask: at what load, what model, what
sequence length, what SLO, and does the bottleneck just shift elsewhere? Your
analysis follows the data path end-to-end, measures every stage, and exposes
what the paper hides.

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

> 🚨 **如果 paper 或文章中有对应代码实现，必须结合代码解读** —— if the
> paper, its predecessor runtime, or its main baseline has any publicly
> available code, you MUST clone and walk through the critical path
> **before** writing the sections below. Not optional.
>
> For **framework** papers, the critical-path targets are:
> - **Scheduler main loop** — batch formation, priority arbitration, continuous-batching entry point
> - **Memory manager** — paged KV cache / prefix cache / offloading / eviction policy implementation
> - **Parallelism layer** — TP/PP/EP sharding, collective calls, pipeline stage boundaries
> - **Runtime / kernel-dispatch boundary** — how the framework invokes the underlying kernels (Mirage MPK task-graph emitter, vLLM `model_executor`, SGLang radix tree, TRT-LLM runtime)
>
> Repos to check first: the paper's own repo → Mirage MPK / vLLM / SGLang
> / TRT-LLM / DeepSpeed-Inference (depending on which one the paper
> extends or compares against).
>
> Code trumps prose. When paper and code conflict, trust the code and
> call the discrepancy out in the notes. If the paper's code is not
> released, check its predecessor runtime or the baseline it claims to
> beat — silent skipping is a documented failure mode. See SKILL.md
> "Paper × Implementation Cross-reference — UNIVERSAL RULE" for the
> full source-priority table.

Guided analysis for papers about inference/training systems, serving
frameworks, scheduling, orchestration, and distributed infrastructure.

**Key Focus Areas** (prioritize these in analysis):
- Request scheduling & batching (continuous batching, chunked prefill, priority scheduling)
- KV cache management (paged attention, cache eviction, compression, offloading, prefix caching)
- Serving optimization (TTFT, TPOT, throughput, SLO-aware, disaggregated prefill/decode)
- Pipeline / tensor / expert parallelism design and scheduling
- Memory management and GPU utilization
- Distributed training systems (communication overlap, gradient compression)

**This guide is called automatically by the paper-reader pipeline.**
Work through EVERY section below against the paper content. For each section,
write a structured answer under the section heading. If a section does not
apply, write "N/A — {brief reason}". The output will be appended to the
paper's notes file.


**图表驱动分析原则 — 图片必须与上下文对应**:

Details analysis 以论文的图表和算法为中心展开。核心规则：

1. **图片内嵌到分析上下文中** — 当你在某个 section 讨论到一张图时，
 立刻用 `### Figure N: title` + `![](...)` 插入该图，让图片紧跟
 在分析文字旁边。不要把图集中放在某个地方，必须分散在对应的讨论中。

2. **只引用核心重要的图** — Details 中只引用 3-5 张最核心的图，判断标准：
 - 被论文**多次引用**的图（"As shown in Figure X" 出现 3+ 次）
 - 论文用**大篇幅讨论**的图（超过半页文字描述该图）
 - 出现在论文**最开始位置**的图（前 3 张图通常最重要）
 - 是架构图、原理图、核心结果对比图
 不要引用 appendix 图、补充实验小图、与核心贡献无关的图。

3. **每张引用的图必须有完整解读** — 格式：
 ```
 ### Figure N: {title}
 ![Figure N](../images/{paper-id}/figN-{name}.png)
 **解读**: {这张图展示了什么，为什么重要，图中各组件/箭头/数据流
 的含义，以及它如何支撑本 section 的分析结论}
 ```

4. **每个 section 引用相关 Figure/Table/Algorithm 编号** — 如果某个
 section 的结论可以用图表佐证，必须指出 "如 Figure X 所示"。


### 1. System Scope

- What is the system's primary goal? (serving, training, fine-tuning, data processing)
- Single-node or distributed? Scale target (GPUs, nodes)?
- Online (latency-sensitive) or offline (throughput-optimized)?
- What workload does it target? (LLM inference, MoE, multimodal, RL training)

### 2. Architecture & Data Flow

Draw or describe the system's component diagram:
- Request flow: how does a request enter, get scheduled, execute, and return?
- Control plane vs. data plane separation?
- State management: what is stateful, what is stateless?
- Failure handling: how does it recover from GPU/node failures?

#### 2a. End-to-End Data Flow Diagram

Trace data through the ENTIRE system, from input to output. For each stage:

```
[Input] → [Stage 1: name] → [Stage 2: name] → ... → [Output]
 ↓ data format ↓ data format
 ↓ location ↓ location
 ↓ latency ↓ latency
```

For each stage, specify:

| Stage | Input → Output | Location (CPU/GPU/NIC) | Latency | Data format & size |
|-------|---------------|----------------------|---------|-------------------|
| Tokenization | text → token IDs | CPU | ? ms | List[int], ~N tokens |
| Prefill | tokens → KV cache | GPU HBM | ? ms | [layers, heads, seq, dim] |
| Decode | KV cache → logits | GPU HBM | ? ms/token | [vocab_size] |
| ... | ... | ... | ... | ... |

#### 2b. Data Movement Hotspots

Identify the TOP 3 data movement bottlenecks:

1. **What data** moves, **how much** (GB), **from where to where** (CPU↔GPU, GPU↔GPU, node↔node)?
2. **Frequency**: once per request, once per token, once per batch?
3. **Is it overlapped** with compute, or blocking?

### 3. Design Space & Constraint Analysis (extends Phase 2)

Phase 2 已提供初步的约束推导。本节从 framework 视角**补深**——
Before analyzing what the system chose, map out what was rejected and why.

**3a. Alternative Approaches**
- List ALL plausible design alternatives (including ones not in the paper)
- For each, determine: feasible or infeasible? Why?

**3b. Constraint Derivation**
- For each rejected alternative, provide a first-principles proof of why
 it fails (memory capacity, communication volume, latency bound, etc.)
- Build a feasibility matrix (rows=alternatives, cols=key criteria)

**3c. Assumption Audit**
- What assumptions does the system rely on? (e.g. "workload is compute-
 bound", "batch size is at maximum", "requests have similar lengths")
- Under what conditions might each assumption break?
- When uncertain, state so explicitly.

**3d. Core Technical Barrier**
- Identify the ONE hardest-to-replicate technique. Not the high-level
 idea, but the low-level engineering trick (e.g. custom SM scheduling,
 specific memory layout trick, compiler pass).
- Support with data showing WHY it works.

**3e. Design Binding Critique**
- What prerequisites does this system FORCE? (e.g. "requires chunked
 prefill + tensor parallelism", "only works with GQA models")
- For each binding: what flexibility is lost?

**3f. 作者证明 — 复现作者的证明装置**

> 🚨 **MANDATORY for every framework paper.** Inherits the "作者证明"
> rule from SKILL.md Phase 2b. This sub-section must be the **largest in §3**
> if the paper has any analytical model. Skipping it (or compressing it to a
> one-line entry in §4 Key Innovations) is a documented failure mode (see
> incidents.md "PrfaaS Throughput Model"). Do not repeat.

Framework papers almost always carry a **closed-form throughput / latency /
cost model** as their 作者证明. Typical signals: §"Modeling" / §"Analysis" /
§"Scheduling Policy" with numbered equations, or a Notation table (Greek-
letter inflation). If you found it, **reproduce it in full** here using the
6 minimum requirements from SKILL.md 作者证明 rule:

1. **Notation table** — copy verbatim from the paper, do NOT shorthand
2. **Each equation walked one-by-one** — physical meaning + why this form
 (why min not sum? why divide by p? why is RDMA omitted from min?)
3. **Monotonicity / convexity / uniqueness** — fill gaps the paper leaves
 implicit (most authors skip these for brevity; you must NOT)
4. **First-order mapping from model → reported numbers** — verify the
 case study's headline figures (54%, 19.4K, 13 Gbps, ...) actually fall
 out of the equations when paper-supplied profile is plugged in. This
 tests whether 作者证明 is load-bearing or decorative.
5. **One-sentence "model-only defense"** — if the case study were stripped
 from the paper, what would the model alone justify?
6. **作者证明 attack surface** — Jensen inequality on long-tail dist? Static
 model under bursty traffic? Closed-form assumption hiding cache-hit
 coupling? These become the *real* §3c assumption-audit anchors.

**Anti-patterns specific to framework papers**:
- ❌ "throughput model 数学好懂, 略" — load-bearing ≠ hard-to-understand
- ❌ Quoting Eq.(N) by number without reproducing the formula in LaTeX
- ❌ Treating Figure 5(a/b)'s geometric intuition as a substitute for the
 equations — the figure is a model visualization, not the model
- ❌ Reporting case study numbers (Table 6 in PrfaaS) twice but
 derivation zero times
- ❌ Manufacturing a model when the paper has none — say
 "**No formal 作者证明 — paper is empirical-only**" honestly

**If the paper genuinely has no model** (rare for framework papers), it
usually means: a position paper, a system release report, or a paper that
treats its sweep table as the proof. In that last case, the 作者证明 is the
sweep itself — reproduce the sweep matrix and call out which axes are
load-bearing.

> 🚨 **Two-axis gate (inherited from SKILL.md Phase 2b)** — every
> "若 X 失效 / 若 X 被替代" counterfactual MUST pass BOTH:
>
> 1. **Inside axis — Argument Chain anchor**: cite which step k of the
> 论证链重构 table this critique attacks. If you can't name k, you're
> attacking a strawman of your own making. Real incident: PrfaaS
> (2604.15039) shipped a critique attacking "若 dense 回归" — but
> paper's chain step 2 had already shown GQA/MLA exists AND is
> insufficient; the critique landed on a battlefield the paper had
> already won inside its own §2.
> 2. **Outside axis — Ecosystem Reality Check**: (a) Term gate — does
> the paper's word X mean what you think? Cite Table/§. (b)
> Plausibility gate — name ≥2 production deployments in the last
> 12 months where X is the dominant choice; if you can't, drop the
> counterfactual.
>
> Either gate alone is insufficient. Re-read incidents.md "PrfaaS
> straw-man critique" for the full case study.

### 4. Key Innovations

For each claimed innovation, analyze:

| Innovation | Mechanism | Benefit | Cost/Tradeoff |
|-----------|-----------|---------|---------------|
| (e.g. continuous batching) | (how it works) | (throughput gain) | (complexity, memory) |

### 5. Scheduling & Resource Management

- Batch formation strategy: static, dynamic, continuous?
- Memory management: paged, pre-allocated, elastic?
- GPU utilization: how does it avoid bubbles / idle time?
- Multi-tenancy support? Isolation guarantees?
- Priority / SLO-aware scheduling?

### 6. Target Scenarios & Workload Characterization

Identify the specific deployment scenarios this system targets:

| Scenario | Workload Pattern | SLO / Goal | Why existing systems fail |
|----------|-----------------|------------|-------------------------|
| e.g. Chatbot serving | High concurrency, short I/O | TTFT < 200ms, TPOT < 30ms | Prefill-decode interference |
| e.g. Batch summarization | Low concurrency, long input | Max throughput, cost/token | Memory fragmentation |
| e.g. Agent tool calling | Multi-turn, variable latency | E2E < 2s | State management overhead |

For each scenario, what is the **primary bottleneck**?
- Compute-bound (GPU FLOPS saturated)?
- Memory-bound (HBM bandwidth / capacity)?
- Communication-bound (inter-GPU, CPU-GPU, network)?
- Scheduling-bound (queuing delay, batch formation)?

### 7. Performance Evaluation & Before-After Comparison

#### 6a. Metrics Definition

List ALL metrics the paper reports, with precise definitions:

| Metric | Definition | Unit | Higher/Lower is better |
|--------|-----------|------|----------------------|
| Throughput | Output tokens per second across all requests | tokens/s | Higher |
| TTFT | Time from request arrival to first output token | ms | Lower |
| TPOT | Average time per output token (after first) | ms | Lower |
| E2E Latency | Total request completion time | ms | Lower |
| P99 Latency | 99th percentile request latency | ms | Lower |
| Goodput | Requests meeting SLO / total requests | % | Higher |
| Cost | $/million tokens or GPU-hours/request | $ | Lower |

#### 6b. Before-After Comparison Table

For EACH key optimization, fill in measured impact:

| Optimization | Metric | Baseline Value | After Optimization | Improvement | Conditions |
|-------------|--------|---------------|-------------------|-------------|------------|
| e.g. Paged KV cache | Max batch size | 32 | 256 | 8x | Llama-70B, A100-80G |
| e.g. Continuous batching | Throughput | 1200 tok/s | 3400 tok/s | 2.8x | ShareGPT workload |
| ... | ... | ... | ... | ... | ... |

**Critical**: note the exact conditions (model, hardware, workload, batch size,
sequence length) for each comparison. Reject vague claims without conditions.

#### 6c. Bottleneck Shift Analysis

After each optimization, does the bottleneck shift?

```
Before: memory-bound (KV cache) → After optimization A: compute-bound (attention)
→ After optimization B: scheduling-bound (batch formation delay)
```

Map the optimization trajectory: what was the bottleneck at each stage,
and what is the REMAINING bottleneck after all optimizations?

#### 6d. Baselines & Fairness

- Baselines compared against — fair comparison? (same hardware, same model, same workload)
- Workload characteristics: request rate, input/output length distribution?
- At what scale / load does the system's advantage appear?
- What is the overhead of the framework itself? (scheduling latency, memory fragmentation)
- Are there scenarios where the baseline WINS? Under what conditions?

### 8. API & Usability

- API compatibility: OpenAI-compatible? Custom protocol?
- Model format support: HuggingFace, GGUF, SafeTensors, custom?
- Deployment story: Docker, Kubernetes, bare-metal?
- Configuration complexity: how many knobs to tune?

#### 8a. Multi-tenancy & Side-channel Audit (MANDATORY for persistent-resource frameworks)

Any framework that holds a GPU resource **persistently across requests**
— persistent megakernel, long-lived CUDA graph, shared KV prefix cache,
multi-tenant scheduler — introduces cross-tenant attack surface that the
paper likely did not address. Answer:

1. **Which resources are held persistently?** (Megakernel occupying all CUs;
 L2 cache state preserved across ops; KV prefix cache shared across users;
 scheduler event counters in global memory)
2. **What information could a co-tenant infer via contention?**
 - L2 cache hit/miss timing → other tenant's working-set size
 - Shared prefix cache radix tree → which system prompts other tenants use
 - Scheduler queue depth → other tenant's batch arrival rate
 - Memory allocator layout → sensitive data placement
3. **What isolation guarantees does the paper provide?** Usually zero.
 If so, note which production scenarios (single-tenant inference,
 trusted multi-tenant, untrusted multi-tenant cloud) the system is
 safe for, and which require future work on isolation.
4. **Concrete side-channel examples for this framework**:
 - Persistent megakernel + partitioned L2 (Fleet-like): L2 contention
 across XCD boundaries can leak whether another tenant's weights
 match yours → model fingerprinting attack
 - Prefix cache (RadixAttention / PagedAttention): hit timing on shared
 prefix reveals whether another user asked a similar prompt →
 prompt fingerprinting
 - Speculative decoding: draft/verify time difference can reveal which
 tokens the other tenant is generating

> **Why this rule exists** (real external review, 2026-04 NeuralTalk
> Fleet review): "当 GPU 被多个租户共享时（云环境），持久化内核的长时间
> 驻留特性带来了新的安全挑战——如何防止一个租户的持久化内核通过侧信道
> （如 L2 缓存竞争）泄露另一个租户的信息？" Fleet paper 完全未涉及这
> 一点，但这是所有 persistent-resource 系统走向生产环境的必经之路。

### 9. Infrastructure Impact

| Layer | Question |
|-------|----------|
| Algorithm | Does it enable new training paradigms? (online RL, continual learning) |
| Kernel | Does it require custom kernels? Or use existing libraries? |
| LLM | What model architectures does it support? Limitations? |
| Agent | Does it support streaming, tool calls, multi-turn, long context for agents? |
| Ops | Monitoring, logging, autoscaling capabilities? |

### 10. Comparison Matrix

Compare to existing systems:

| Feature | This System | vLLM | SGLang | TRT-LLM | DeepSpeed |
|---------|------------|------|--------|---------|-----------|
| Continuous batching | | | | | |
| Paged attention | | | | | |
| Speculative decoding | | | | | |
| Multi-node | | | | | |
| Quantization support | | | | | |

Fill in what is known. Mark unknowns explicitly.

### 11. Adoption, Maturity & Ecosystem Influence

- Open source? License?
- Community size, contributor count, release cadence?
- Production deployments mentioned?
- What would it take to adopt this in an existing stack?
- **Downstream influence**: Has this work been adopted by major
 frameworks (vLLM, SGLang, TRT-LLM, etc.)? Cite specific versions,
 PRs, or release notes. Did subsequent papers or tech reports (e.g.
 DeepSeekV3) adopt similar ideas? This connects the paper to the
 living ecosystem rather than treating it as an isolated work.

### 12. Software → Hardware Reverse Implication — CONDITIONAL (only if hardware-proximal)

> ⚠️ **Not all framework papers touch hardware.** This section is
> **conditional**: run the rule only if the paper falls into the
> **hardware-proximal framework** subset. Skip it explicitly (write
> `[N/A — pure-software framework: <reason>]`) for pure scheduler /
> batching / routing papers.

**Trigger conditions** (ANY of these → section IS required):
- Persistent megakernel / long-lived runtime exposing SM occupancy
 constraints (e.g. Fleet, HazyResearch MegaKernel, Mirage MPK)
- Explicit cache-scope / cache-modifier control (e.g. Fleet's 3-tier
 `sc1/nt` strategy, rocBLAS tile-level cache policy)
- Chiplet / NUMA / GPC affinity scheduling (Fleet Chiplet-task,
 HipKittens XCD-grouping)
- KV cache physical layout control (PagedAttention, RadixAttention with
 custom page sizes mapping to HBM pages)
- Inter-GPU communication scheduling that depends on NVLink/IF/UALink
 topology (DeepEP, custom NCCL wrappers)
- Kernel-dispatch primitives exposed to scheduler (task graphs with
 SM-binding hints, warp-scheduling control)

**Pure-software framework examples** (trigger NOT fired → skip with
justification):
- Continuous batching / dynamic batching (vLLM's continuous batch)
- Request routing / load balancing across replicas
- KV prefix cache management (LRU / radix / Hermes-style)
- Disaggregated prefill-decode scheduling (logical, not HW-bound)
- SLO-aware priority scheduling
- Multi-model / multi-LoRA serving

**If trigger fires**, reuse the `deep-kernel.md` §12 template (future
HW feature → current SW cost → savings if hardened → paper evidence
anchor). **Do not duplicate** the template text — just produce the
wishlist table.

**Why this rule is conditional, not universal** (user feedback,
2026-04): a pure scheduler paper like "smarter continuous batch
formation policy" has no hardware implication to reverse-reason
about. Forcing this section on every framework paper is hand-waving.
The rule's real home is `kernel` / `hardware` / `cluster`; framework
papers opt in only when they expose HW primitives.

---

## ✅ End-of-Deep-Read MANDATORY Checklist

This guide defines **12** numbered sections (§12 is CONDITIONAL —
see below). Before declaring Phase 5 complete, verify in your notes file:

- [ ] **All 11 unconditional `### N. Title` sections (§1–§11) from this
 guide are answered** under `## Deep Analysis (framework)` in the
 notes. Missing sections are not allowed unless explicitly
 justified with `[N/A — reason]`.
- [ ] **§12 SW→HW Reverse Implication handled correctly**: either
 (a) triggered because paper is hardware-proximal → wishlist table
 written with ≥2 rows + paper §/Fig anchors; or (b) explicitly
 skipped with `[N/A — pure-software framework: <reason>]`
 stating why the trigger conditions don't fire. Silent skipping
 is a documented failure mode.
- [ ] **§3f 作者证明 完整复现** — notation
 table, every equation explained, monotonicity argument, model→numbers
 mapping, "no-case-study defense" sentence, attack surface. If paper
 has no formal 作者证明, you wrote `[No formal 作者证明 — empirical-only]`
 honestly and proceeded. Compressing 作者证明 to a one-line entry in §4
 Key Innovations is BANNED (see incidents.md "PrfaaS Throughput
 Model").
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

