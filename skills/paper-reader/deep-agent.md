# deep-agent.md — Agent-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **agent-specific**.

Covers: agent runtimes, tool-use systems, planning, multi-agent, LLM-as-OS.

## §1 Agent scope

- Task class: open-ended (browser / coding / research) vs closed
  (function calling / single-turn RAG)?
- Interaction pattern: single-shot / multi-turn / continuous?
- Autonomy: human-in-the-loop, fully autonomous, supervised?

## §3 Architecture — the agent loop

Must draw as a Mermaid `stateDiagram-v2` or `sequenceDiagram`:

- State of one turn: observation → plan → act → verify → reflect
- Memory model: short-term (context window), long-term (vector DB /
  file), episodic (trajectory log)
- Tool invocation: call syntax, argument schema, result handling
- Error recovery: when a tool fails / verification fails, what state
  does the agent fall back to?

## §3 Planning & reasoning

- Planning style: ReAct (think-then-act loop) / tree search (ToT / LATS)
  / predetermined workflow / graph?
- Decomposition: top-down (task → subtasks) or bottom-up (act → assemble)?
- Budget: max steps, max tokens, max tool calls per task
- Backtracking: can the agent undo an action? revisit a branch?

## §4 作者证明 — Agent-specific asks

In addition to the 6 base checks (SKILL.md):

- **Success-rate model** (usually empirical): sweep matrix over
  `(task difficulty, planning depth, tool set, backbone model)` —
  reproduce the sweep, note monotonicity along each axis
- **Latency budget per turn**: tool call + LLM inference + memory read.
  Does the paper claim "interactive latency"? Verify from the sweep.
- **Failure mode classification**: how many failure classes does the
  paper identify? What's the dominant one? Is the method targeted at
  fixing that class?
- Almost no agent paper has formal convergence / success guarantees —
  mark **无形式化作者证明 — 仅实证** is the common case. State what
  metric *could* have been bounded.

## §5 Tool & environment interface

- Tool catalog: name, description, argument schema (JSON schema? natural
  language?)
- Side effects: which tools are read-only vs write, reversible vs not
- Error surface: what does a broken tool look like to the agent? (exception
  stringified? retry wrapper? LLM sees none?)
- Environment contract: is the env assumed deterministic? stateless between
  turns?

## §7 LLM backbone requirements

- Minimum model size for the method to work (ablation with smaller
  backbones?)
- Required capabilities: long-context? tool-call format? structured
  output? chain-of-thought?
- Sensitivity to backbone: does the same method work with Llama / Qwen /
  Claude, or is it Claude-specific?
- Serving cost: tokens per task × tasks per day × cost per million tokens

## §8 Evaluation

- Benchmarks used: SWE-bench / WebArena / τ-bench / AgentBench / tau2
  / custom? Which version?
- Task difficulty distribution (easy / medium / hard breakdown)
- Metric: pass@1? pass@k? human eval? cost-adjusted?
- Baselines: ReAct? Reflexion? CodeAct? the tested frontier model alone?

## §9 Multi-agent (if applicable)

- Topology: pipeline / hub-and-spoke / peer mesh / hierarchical
- Coordination: shared memory / message passing / common blackboard
- Role specialization: are agents differently prompted? different models?
- Failure isolation: does one agent's crash take down the system?

## §9 Production readiness

- Sandboxing: how is tool execution isolated? (Docker / VM / Firecracker /
  process?)
- Secrets & auth: how does the agent get API keys? what's the blast
  radius if compromised?
- Observability: trace logging, reproducibility of non-deterministic runs
- Cost / rate limit / concurrency controls
