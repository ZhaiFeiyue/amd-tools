# deep-framework.md — Framework-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **framework-specific**.

## §1 System scope — pin down

- Stage coverage: prefill? decode? both? (and does the paper fuse them?)
- Serving or training? Continuous batching or static?
- Parallelism dimensions covered: TP / PP / EP / DP / SP — which axes
  does the framework own, which does it delegate?
- Deployment mode: single node / multi-node / disaggregated prefill-decode?

## §3 Architecture diagram requirements (framework-specific)

Must show:
- Request / task lifecycle from arrival → tokenize → schedule → kernel
  dispatch → output. Mermaid `sequenceDiagram` works well.
- **Scheduler** as a first-class box with its queueing discipline (FCFS
  / SJF / priority / waterfall). Show how it makes batching decisions.
- **KV / memory manager** separated from scheduler; show the allocation
  unit (page size / block size / continuous chunk).
- Cross-node communication path (if multi-node): what collective, what
  transport (NVLink / IB / RoCE / UE).

## §4 作者证明 — Framework-specific asks

In addition to the 6 base checks (SKILL.md), framework papers almost
always have a throughput / latency / cost model (look for `§"Modeling"`,
`§"Performance Model"`, `Eq.N` in §3+). Specifically:

- Reproduce the **notation table** (`λ`, `μ`, `B`, `T_M`, `N_p`, etc.)
- Each equation's **physical meaning**: why `min` vs `sum`, why divide by
  `p`, why does X get excluded from the denominator
- **Monotonicity / convexity** (usually implicit): is the optimum interior
  or at a boundary? what breaks monotonicity?
- **First-order mapping**: plug the case-study workload (Table N) into
  the model and verify the reported numbers (54% / 19.4K / 13 Gbps /
  N_p=3) come out — NOT sweep-then-fit
- If no model: mark **无形式化作者证明 — 仅实证** and list what a model
  would have clarified

## §5 Scheduling & resource management

For each scheduling decision in the paper:
- Granularity: request / token / layer / operator?
- Preemption policy: can a scheduled task be killed? restarted?
- Admission control: what's the drop / queue policy under overload?
- Fairness: any per-tenant or per-priority guarantees?

For memory management:
- KV cache allocation unit (page size / block size)
- Fragmentation behavior
- Eviction / reuse (prefix cache, radix tree, LRU)
- Swap to CPU / disk? when is it triggered?

## §6 Workload characterization — the scenarios where this wins / loses

Explicit table:

| Workload regime | This framework | Baseline | Why |
|---|---|---|---|
| short prompts, low concurrency | ... | ... | ... |
| long prompts, high concurrency | ... | ... | ... |
| mixed prefill-decode | ... | ... | ... |

Papers that only show "we win on average" without this breakdown are
hiding the regime where they lose. Surface it.

## §7 Evaluation / §5 before-after comparison

- Which baseline versions? (vLLM v0.X / SGLang v0.Y — specific commit
  hashes if available)
- Were the baselines fairly tuned? Any config the paper hid?
- Metric definitions: is "throughput" output tok/s or input+output?
  Is "latency" p50 / p99 / TTFT / ITL?

## §8 API & usability

- User-facing API: OpenAI-compat? Custom? Python? gRPC?
- Config surface: one flag? dozens of flags? Does the paper describe
  how to tune them?
- Migration cost from existing deployment (vLLM → this framework)

## §9 Adoption & ecosystem

- Has this been merged upstream (to vLLM / SGLang / TRT-LLM)?
- Which companies / models deploy it in production (if stated)?
- What does it force downstream? (e.g. "requires chunked prefill + TP
  bound together" — see PrfaaS / NanoFlow)

## §12 Software → Hardware reverse implication — CONDITIONAL

Trigger only if the framework is **hardware-proximal**: persistent
megakernel, cache-scope control, chiplet affinity, KV cache physical
layout, interconnect-aware scheduling. Pure software frameworks
(continuous batching, request routing, load balancing, radix prefix
cache) do NOT trigger this.

If triggered: what hardware primitive would simplify the software? What
future ISA / cache policy / interconnect feature does this framework's
complexity argue for?
