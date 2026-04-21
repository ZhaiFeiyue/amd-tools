# Deep Reading: Agent Papers

## Role

You are a **tech lead** who has built and shipped production AI agent systems
— coding agents, customer service agents, data analysis pipelines — that
handle thousands of concurrent sessions. You have designed agent runtimes,
implemented tool-call routers, built memory systems, and debugged agents that
loop forever or hallucinate tool calls. You know that the gap between a demo
agent and a production agent is 10x engineering effort. When you read an agent
paper, you immediately ask: what is the failure rate on real tasks, what is
the token cost per task, how does it handle tool errors, and will this
architecture survive 1000 concurrent users? You evaluate agent systems on
reliability, cost, latency, and safety — not just success rate on curated
benchmarks. You have opinions on when multi-agent is justified versus when
a single well-prompted model is better.

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
> paper's own repo, an agent framework it builds on, or a baseline it
> compares against has publicly available code, you MUST clone and walk
> through the critical path **before** writing the sections below.
>
> For **agent** papers, the critical-path targets are:
> - **Main agent loop** — `run()` / `step()` / `act()` control flow
> - **Planning module** — how the planner turns user goal into subtask tree / next-action selection
> - **Tool call wrapper** — how tool schemas are serialized, called, and results injected back into context
> - **Memory / state management** — scratchpad, persistent memory, context-window compression
> - **Multi-agent message bus** (if applicable) — how agents exchange state / negotiate / vote
>
> Repos to check first: the paper's own repo → LangGraph / AutoGen /
> CrewAI / OpenAI-agents-python / Anthropic computer-use / LlamaIndex
> agents / Hermes / SWE-agent / open-interpreter / Letta (MemGPT) —
> whichever the paper extends or benchmarks against.
>
> Code trumps prose. When paper and code conflict, trust the code and
> call the discrepancy out in the notes. See SKILL.md "Paper ×
> Implementation Cross-reference — UNIVERSAL RULE" for the full rule.

Guided analysis for papers about AI agent systems, tool use, planning,
reasoning, multi-agent coordination, and agentic frameworks.

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


### 1. Agent Architecture

- Agent type: single-agent, multi-agent, hierarchical?
- Core loop: ReAct, plan-then-execute, tree search, reflection, other?
- Memory mechanism: context window only, external memory, RAG, episodic?
- Tool use protocol: function calling, code execution, API calls, browser?
- How does the agent decide WHEN to use tools vs. answer directly?

### 2. Planning & Reasoning

- Planning granularity: step-by-step, task decomposition, goal hierarchy?
- Does it backtrack or revise plans on failure?
- Reasoning approach: CoT, ToT, GoT, self-consistency, other?
- How does it handle ambiguity or underspecified tasks?
- Evaluation: how is plan quality measured?

### 3. Tool & Environment Interface

- What tools/APIs are available to the agent?
- Tool description format: JSON schema, natural language, code signatures?
- How does it handle tool errors or unexpected outputs?
- Environment: static, dynamic, partially observable?
- Sandboxing / safety: how are dangerous actions prevented?

### 4. LLM Backbone Requirements

- What base model is used? Size requirements?
- Does it require special fine-tuning (tool-use SFT, agent-specific RLHF)?
- Prompt format: system prompt engineering, few-shot, structured output?
- Context window requirements: how much context does the agent consume?
- Latency sensitivity: interactive vs. batch? Max acceptable response time?

### 5. Evaluation

- Benchmarks: SWE-bench, WebArena, GAIA, AgentBench, ToolBench, custom?
- Metrics: success rate, # steps, cost, latency, human preference?
- Baseline comparisons: what other agents / methods are compared?
- Failure analysis: what types of tasks does it fail on? Why?
- Human evaluation component?

### 6. Multi-Agent (if applicable)

- How many agents? Role specialization?
- Communication protocol: shared memory, message passing, debate?
- Coordination: centralized controller vs. emergent cooperation?
- How is redundant or conflicting work handled?
- Overhead of multi-agent vs. single-agent?

### 7. Infrastructure Impact

| Layer | Question |
|-------|----------|
| Algorithm | Training recipe for the agent? RL environment design? |
| Kernel | Latency bottleneck in tool-call round trips? Batching implications? |
| Framework | Serving needs: streaming, multi-turn state, long context, function calling API? |
| LLM | Model requirements: size, fine-tuning, context length, output format? |
| Cost | Total cost per task? Token usage breakdown (reasoning vs. action)? |

### 8. Production Readiness

- Reliability: what % of tasks succeed without human intervention?
- Safety: guardrails, human-in-the-loop, action confirmation?
- Observability: logging, tracing, debugging agentic runs?
- Scalability: concurrent agent sessions, resource usage?

### 9. Impact on AI Infra

- Does this change how we think about LLM serving? (long-lived sessions, stateful)
- Does it push for specific model capabilities? (better tool use, longer context)
- Does it create new kernel requirements? (faster decoding for interactive use)
- What infrastructure would a production deployment of this agent require?

---

## 作者证明 — Agent 特定要求

> Inherits the **作者证明** requirement from §2b. Agent papers most often
> have **no closed-form model**; the 作者证明 is the **sweep matrix + per-axis
> monotonicity argument + failure-mode catalog**. That is *still* a 作者证明
> and must be reproduced as such.

**Typical 作者证明 shape**:
- Tool-call / planning success-rate matrix sliced by (model × task × depth)
- Latency / cost budget per agent turn
- Failure-mode taxonomy (what fraction of failures are tool-call format vs
 reasoning vs context-overflow)
- Occasionally: a Bellman / planning-depth bound or success-prob model

**Where it lives**:
- §"Evaluation" tables (yes — for agent papers the eval table often IS the
 作者证明, not just empirical validation)
- §"Failure Analysis" / "Error Taxonomy"
- For RL-on-agent papers: policy improvement bound or KL guarantee

**Required content**:
1. **Reproduce the sweep matrix in full** — do NOT shorthand to "improves
 by X%". The shape of the matrix (which rows / cols matter) is the proof
2. **Per-axis monotonicity / saturation**: e.g. "success rate monotone in
 model size up to 70B then saturates" — if the sweep doesn't show this
 shape, the claim is weaker than presented
3. **Failure-mode breakdown**: what fraction of remaining failures are
 format errors vs genuine reasoning gaps vs environment limits?
4. **Cost / latency budget per turn**: agent papers often hide this — if
 the paper reports task success but not $/task, flag it
5. **Attack surface**: which axes weren't varied? (Most agent papers fix
 the prompt template, fix the tool set, fix the environment) — these
 are the load-bearing implicit assumptions
6. **If a formal model exists** (rare): treat per the framework 作者证明 rules

**Anti-patterns**:
- ❌ "Achieves SOTA on benchmark X" without the per-task breakdown
- ❌ Reporting only the average success rate when distribution has a long
 tail of catastrophic failures
- ❌ Skipping the cost analysis ("$0.50/task" is critical for adoption)

## ✅ End-of-Deep-Read MANDATORY Checklist

This guide defines **9** numbered sections. Before declaring
Phase 5 complete, verify in your notes file:

- [ ] **All 9 `### N. Title` sections from this guide are answered**
 under `## Deep Analysis (agent)` in the notes. Missing sections are
 not allowed unless explicitly justified with `[N/A — reason]`.
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

