# deep-llm.md — LLM-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **LLM-specific**
and must be added on top.

## §3 Architecture diagram — code source priority

Before drawing the architecture, locate the model's reference code. Use
at least **2 independent sources** for cross-validation.

| Priority | Source | Lookup command |
|---|---|---|
| 1 | Paper's official repo | `gh search repos "<model>" --limit 5` → `git clone --depth 1` |
| 2 | HF `transformers/models/<name>/modeling_<name>.py` | `curl -sL "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/<name>/modeling_<name>.py"` |
| 3 | HF Model Card `modeling_*.py` (trust_remote_code) | `curl -sL "https://huggingface.co/api/models/<org>/<model>" \| jq '.siblings[].rfilename' \| grep modeling` |
| 4 | vLLM `vllm/model_executor/models/<name>.py` | `ls /apps/feiyue/upstream/vllm/vllm/model_executor/models/ \| grep -i <name>` |
| 5 | SGLang `python/sglang/srt/models/<name>.py` | `ls /apps/feiyue/upstream/sglang/python/sglang/srt/models/ \| grep -i <name>` |
| 6 | HF `config.json` | `curl -sL "https://huggingface.co/<org>/<model>/raw/main/config.json"` |

Every architecture claim in §3 / §7 must cite `file:line`. "From the
code" without anchor is rejected.

**Diagrams-over-code**: do not paste `class Foo(nn.Module)` or `def forward`
blocks into notes. Architecture → drawio/Mermaid. Code blocks only for:
identifier regex patterns, config.json → diagram position mapping table,
or short Algorithm pseudocode.

## §3 Must-draw per-block expansions

Every LLM note must draw the following as Mermaid (or drawio if multi-page):

- Top-level data flow: tokenizer → embed → N×decoder → norm → lm_head
- **Attention variant** (MHA / GQA / MQA / MLA / sliding window): show
  q/k/v projections, head counts, RoPE application, KV shape — with
  concrete dims from config.json
- **FFN variant** (SwiGLU / GeGLU / MoE): if MoE, show routing, expert
  count (routed + shared), top-k, aux loss
- **Normalization + position encoding** layout (pre-norm vs sandwich,
  RoPE base, long-context rescaling)

## §4 作者证明 — LLM-specific asks

In addition to the 6 base checks (SKILL.md):

- **Scaling-law fit** if paper is a model release: which scaling law did
  the authors fit to (Chinchilla / DeepSeek-style)? Reproduce the fit
  exponent and discuss residuals.
- **Parameter breakdown**: reproduce the per-module param count table.
  Verify total matches the reported model size.
- **Capacity budget**: compute attention KV bytes/token and FFN param
  count from first principles; compare with config.json.

## §5 Training recipe itemization (MANDATORY — equal weight to architecture)

Never summarize training as "trained on 2T tokens with SFT+RLHF".
Produce a multi-stage table:

| Stage | Goal | Data (tokens + mix) | LR schedule | Context | Techniques |
|---|---|---|---|---|---|

Must cover: pre-training stages, mid-training (context extension /
annealing), SFT, post-training (DPO / GRPO / KTO / distillation),
quantization-aware training if present. For each field not in the paper,
write `[论文未披露]` — never skip the row.

Also list the **single hardest-to-replicate training trick** (data mix
secret / custom loss / stability hack). Every LLM paper has one.

## §7 Quantization checklist (if paper discusses quantization)

- **Weight quantization**: method (GPTQ/AWQ/SmoothQuant), bits, group size
- **Activation quantization**: scheme (per-tensor / per-token / per-channel),
  bit width, calibration set
- **KV cache quantization**: K format vs V format (often different),
  per-channel vs per-token, accumulation precision
- **Mixed precision**: which tensors stay FP16/BF16
- **Accuracy-efficiency tradeoff**: MMLU / GSM8K / HumanEval drop %

## §8 Serving deployment considerations

- Minimum serving hardware (GPU count + VRAM) by precision (FP16 / FP8 / FP4)
- KV cache bytes per token at each precision → max concurrency at 128K
- Prefill vs decode bottleneck at different batch sizes
- Continuous batching / prefix caching friendliness (static vs dynamic shape)

## §9 Open questions — LLM-specific angles

Beyond generic open questions, include:
- At what scale does this design's advantage saturate / invert?
- Can the recipe be transferred to a different modality (vision / audio)?
- What is the hardware affinity (does FP4 help or hurt this arch)?
