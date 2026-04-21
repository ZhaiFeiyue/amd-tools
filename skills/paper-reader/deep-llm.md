# Deep Reading: LLM Papers

## Role

You are a **staff ML engineer** who has deployed dozens of LLMs into production
across every size class from 1B to 400B+. You have quantized models to FP4,
optimized KV caches, implemented speculative decoding, and made serving cost
decisions that saved millions in GPU-hours. You read a model paper and
immediately estimate: KV cache size per token, minimum serving hardware,
memory-bandwidth vs compute bottleneck at different batch sizes, and whether
the claimed benchmark numbers will hold up on real-world traffic. You know the
difference between "demo quality" and "production quality". When a paper
reports MMLU scores, you check the eval harness version. When it claims long
context, you ask about NIAH results at different depths. You evaluate models
not just on accuracy, but on the full cost-quality-latency Pareto front.

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

## Code Cross-reference Rule (inherited from SKILL.md) — UNIVERSAL

> 🚨 **如果 paper 或文章中有对应代码实现，必须结合代码解读** —— for LLM
> architecture / scaling / quantization / multimodal papers this
> requirement is particularly strict. §1a below is the long-form template
> for model-architecture cross-reference; it must be executed whenever
> any of: the paper's own repo, HF `transformers` `modeling_<name>.py`,
> HF Model Card `modeling_xxx.py`, vLLM / SGLang `model_executor` port,
> or a predecessor model's modeling file is publicly available.
>
> Quantization / scaling-law / tokenizer papers that don't ship a new
> model must still cross-reference the **reference quantizer** (`autoawq`,
> `gptqmodel`, `bitsandbytes`, `llmcompressor`) or the **training
> framework** (Megatron-LM / DeepSpeed / TRL / ColossalAI) that implements
> the recipe. See SKILL.md "Paper × Implementation Cross-reference —
> UNIVERSAL RULE" for the full source-priority table.

Guided analysis for papers about large language model architecture, scaling
laws, quantization, attention mechanisms, and multimodal model design.

**Key Focus Areas** (prioritize these in analysis):
- **Model architecture design** (MANDATORY, priority 1): attention variants (MHA/GQA/MQA/MLA),
 FFN design (SwiGLU/GeGLU), normalization, position encoding, MoE design (routing, experts,
 load balance)
- **Architecture diagram** (MANDATORY): every LLM paper must have a detailed architecture
 diagram in the notes (see Section 1 below). For papers with complex multi-module pipelines,
 prefer embedding an interactive drawio file via `{{drawio:filename.drawio}}` directive.
- **Training methodology** (MANDATORY, priority 1 — equal to architecture): how was the model
 trained? how many tokens (per stage)? what data mix? what techniques (parallelism strategy,
 precision, optimizer, stability tricks, post-training recipe)? **NEVER skim training with
 "trained on XT tokens"** — every stage, every technique must be itemized. See Section 2.
- **Quantization**: weight quantization (GPTQ, AWQ, SqueezeLLM), activation quantization
 (FP8, INT8, FP4), KV cache quantization, mixed-precision inference, accuracy-efficiency tradeoff
- **Architecture decisions vs. competitors**: what each vendor changed and why, ablation evidence
- **Scaling laws**: compute-optimal training, parameter-data tradeoff
- **Tokenizer & vocabulary design**: BPE/SentencePiece, multilingual coverage

**This guide is called automatically by the paper-reader pipeline.**
Work through EVERY section below against the paper content. For each section,
write a structured answer under the section heading. If a section does not
apply, write "N/A — {brief reason}". The output will be appended to the
paper's notes file.

---

## LLM-Specific Extensions to Phase 2

The inherited Phase 2 analysis covers general 约束推导 and 核心壁垒.
For LLM papers, **extend** it with these LLM-specific angles:

- **Architecture constraint derivation**: for each key architectural choice
 (attention type, FFN design, position encoding, MoE routing), explain
 why alternatives were rejected using dimension analysis, hardware
 constraints, or ablation data. Build a feasibility matrix when 3+
 alternatives exist.
- **Training recipe barrier**: identify the ONE training trick that is
 hardest to replicate (data mix, lr schedule, stability hack, etc.)
- **Serving-aware assumption audit**: does the architecture assume
 specific serving conditions (batch size, sequence length, hardware)?
 What breaks if these change?
- **Framework perspective** (when applicable): if the paper involves
 streaming, real-time interaction, or novel serving patterns, trace
 end-to-end latency and identify bottleneck shifts at different
 concurrency levels.

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


### 1. Architecture (核心重点 — 必须包含架构图)

**架构分析是 LLM 论文 deep reading 的第一优先级。** 必须满足以下全部要求：

#### 1.0 Mandatory preflight — 找到 model source code (READ FIRST, before §1a)

> 🚨 **最常被忽略的一步**. LLM agent 在第一次通读 skill 时, 常常把
> §1a 的 "参考实现来源优先级表" 看成了 "informative reference", 开始
> 动手写架构时才意识到没查代码。为避免这种情况, 把源代码 lookup 提到
> §1 的第 0 步, 任何架构文字产出之前必须先完成这一步。

**在写任何架构内容之前, 按以下优先级逐项尝试定位模型实现, 并把找到的
文件路径记录到笔记的 §1a "Implementation references" 表格里**：

| 优先级 | 来源 | 一键定位命令 |
|---|---|---|
| **1. 论文官方 repo** | paper 的 "Code" 字段 / README 里的 GitHub 链接 | `gh search repos "<model_name>" --limit 5` → `git clone --depth 1 <url> /tmp/<name>` |
| **2. HuggingFace `transformers` 主仓库** | `transformers/models/<name>/modeling_<name>.py` | `curl -sL "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/<name>/modeling_<name>.py" \| head` |
| **3. HuggingFace Model Card `trust_remote_code`** | `https://huggingface.co/<org>/<model>/tree/main` 的 `.py` | `curl -sL "https://huggingface.co/api/models/<org>/<model>" \| jq '.siblings[].rfilename' \| grep -i modeling` → `curl -sL "https://huggingface.co/<org>/<model>/raw/main/<modeling_file>"` |
| **4. vLLM 推理实现** | `vllm/vllm/model_executor/models/<name>.py` | `ls /apps/feiyue/upstream/vllm/vllm/model_executor/models/ \| grep -i <name>` |
| **5. SGLang 推理实现** | `sglang/python/sglang/srt/models/<name>.py` | `ls /apps/feiyue/upstream/sglang/python/sglang/srt/models/ \| grep -i <name>` |
| **6. HF `config.json`** | `https://huggingface.co/<org>/<model>/raw/main/config.json` | `curl -sL "..." \| jq .` — 读参数规格, 不是结构, 但**每个数字**都要进 drawio |

**硬要求**:

- 至少 **2 个独立来源** (通常是 HF transformers + vLLM 或 SGLang, 或官方
 repo + HF config.json), 用于**交叉验证**——单一来源可能过时或有错。
- 每个架构声明必须携带 `file:line` 引用 (e.g.
 `sglang/.../kimi_k25.py:678`, `vllm/.../kimi_k25_vit.py:183`,
 `modeling_deepseek_v3.py:L234`)。不接受 "from the code" 这种没有锚点
 的引用。
- 如果真的找不到任何代码（rare, 通常意味着 paper 很新且 trust_remote_code
 还没上传）, 在 §1a 明确标注 `[代码未公开, 仅基于 paper + config.json
 推测]`——这是**警示信号**而非 excuse。

#### 1.05 Diagrams-over-code principle — 首选 drawio, 代码只引锚点

> 🚫 **不允许在 notes 里贴大段 Python / C++ 代码作为架构说明**.
> 架构属于 drawio 的工作, 不是 ```python``` 块的工作。
> 真实事故, 2026-04-21 K2.6 第二轮: 第一版 §1 贴了 4 个大代码块
> (KimiK25ForConditionalGeneration.__init__ / DeepseekV2DecoderLayer /
> MoonVision3dPatchEmbed / K2VLMultiModalProjector), 用户 catch "你
> 直接给代码不是画图"。代码能表达结构, 但**远不如 drawio 可缩放可交互**;
> 而且贴代码让 notes 变成 "代码转载", 失去增量价值。

**Allowed in notes §1**:
- ✅ **file:line 表格** — 把代码锚点做成一张表, 读者在本地编辑器打开
- ✅ **config.json 字段 → drawio 位置映射表** — 每条常量指向它在架构图哪个盒子
- ✅ **关键常量 / 易错点清单** — 用自然语言列出 paper 容易让你看错的地方
- ✅ **短的 identifier 引用** — `re:.*mlp\.(gate|up|down)_proj.*`、`q_a_proj` 之类
- ✅ **对照表** (paper 说法 vs 代码真实情况)

**Forbidden in notes §1**:
- ❌ 贴多行 `class Foo(nn.Module):` 定义
- ❌ 贴多行 `def forward(...):` 函数体
- ❌ 用代码块复现 config.json 字段 (应该用 markdown 表格)
- ❌ 把 drawio 本来能画的 "MLA 内部 Q 路径/KV 路径" 用代码块写
- ❌ 超过 5 行的连续代码块 (除非是 Algorithm 伪代码)

**Rule of thumb**: 读 paper 的人应该从 drawio 得到 "是什么样", 从 file:line
得到 "在哪找实现", 从 notes 文字得到 "为什么这样 + 容易看错在哪"。
三者各司其职, **不允许任一项抢另一项的工作**。

#### 1.1 Architecture diagrams — Mermaid first, drawio for ladder (self-containment)

> 🚫 **No Backbone-Reuse Exemption** (同 SKILL.md 通用规则, LLM 类别
> 尤其严格). 见 `incidents.md` 2026-04-21 条目。
>
> **Tool selection** (updated 2026-04-21E, Mermaid-first policy):
> default to **Mermaid** inline in notes for each view; only escalate
> to drawio when you need **multi-page tab switching** (≥ 3 coordinated
> views that benefit from ladder navigation) or **precise geometry**
> (GPU die / chip floorplan). See
> `~/.cursor/skills/paper-reader/diagram-tool-choice.md` for full
> selection matrix + templates.

架构图必须**自包含**整个模型, 不允许指向其他 paper 的架构。哪怕两篇
paper 架构字字相同:

- ✅ **允许**: 把 M 的 Mermaid / drawio 复制过来作为 N 的起点, 然后在
 N 自己的 notes / .drawio 里独立承载 (commit 在 N 的 paper-id 下)
- ❌ **禁止**: 在 N 的 notes 里写 "架构同 M, 见 [M notes 互动图]"
 而 N 自己没有架构图

**N 的架构图最少要求 (llm 类别硬要求)**:

N 的 notes **必须包含**以下架构视角 (Mermaid 块 inline 或 drawio 多页
均可, 优先 Mermaid):

1. **架构 1 — Top-level pipeline**: 输入 (input_ids / pixel_values /
 audio) → 各 encoder → 合流 → LM → 输出 (logits / codebooks /
 tool_calls)。每条数据流标 tensor shape。
2. **架构 2 — Decoder (或 Encoder) block internals**: 展开到 proj 级别
 —— 每一个 Linear 标 `(in_dim → out_dim, parallelism)`, 每一个 norm
 标 `RMSNorm(dim, eps)`, RoPE / QK-Norm / 残差显式画出。MLA 分 Q/KV
 两路 (各自一个 subgraph), MoE 分 gate / routed / shared / combine。
3. **架构 3 — 非 transformer 子系统** (if any): vision tower / speech
 tower / adapter / MTP head / codec。纯文本模型此项可省, 但 §1a 中
 必须明确说明 "pure text model, no sub-module diagram needed"。
4. **架构 4+ (additive, 可选)**: 训练流水线、量化布局、release delta、
 scaling law 拟合图等。

**Mermaid 优先** — 以下场景都应该用 Mermaid 首选而非 drawio:

- 顶层 pipeline (< 15 节点)
- Decoder block proj-level (通常 15-25 节点 + 2-3 subgraph, Mermaid 够用)
- Vision / audio / speech tower (10-15 节点)
- 量化布局 (3 个子图 + 色块, classDef 色码精准)
- 训练流水线 (stages + branches)

**drawio fallback** 仅在下列情况合理:

- 需要 ≥ 3 coordinated views 做 tab switch (例如 Qwen3-Omni 的 7 页
 top-level + audio + vision + thinker + talker + code predictor +
 code2wav — 这种 hierarchical ladder drawio 明显赢)
- Chip / die floorplan / SM or CU geometry (Mermaid 没有精确坐标)
- 你 fork 了一个前代 paper 的 drawio 继续演化

**Shortcut phrases that MUST NOT appear in the notes**:

- ❌ "架构不画了——与 X 完全相同, 见 [X notes 互动图]"
- ❌ "same backbone as X, refer there"
- ❌ "继承 K2 的架构, 不再重复"
- ❌ "drawio 中已画, 文字不展开" (如果连 drawio 都没画就更不行)

出现以上短语的笔记会在 compliance check 中被标记为 **backbone-reuse
shortcut violation**, 必须改写。

> **反面教训 1** (Kimi K2.6, 2026-04-21): 第一版笔记看到 config.json
> 与 K2.5 字字相同, 直接写 "架构不画了, 见 K2.5 notes", drawio 只画了
> 4 页 release 级别的 delta, 没有一页是架构。User 直接 catch.
> 详见 `incidents.md` 2026-04-21A 条目。
>
> **反面教训 2** (Kimi K2.6, 2026-04-21E): 修好 backbone-reuse 后选
> drawio 把 architecture 画了 4 页, 但 57 KB 的 drawio 在 reader HTML
> embed 里静默渲染失败 (incidents.md 2026-04-21B); 用户"drawio 非常
> 不稳定"直接催生 Mermaid-first 切换。现在 K2.6 的 §1 架构 1-4 全部
> 是 Mermaid inline block, drawio 保留作为多页 ladder 的备份
> (`<details>` 折叠)。

#### 1a. Cross-reference: Paper × Implementation (MANDATORY 前置步骤)

**架构分析不能只读 paper。** Paper 常常**简化、略写、甚至画错**架构
（比如 Qwen3-Omni paper 把 "Code Predictor" 叫 "MTP"，把 15 次 AR 迭代
说成 "一次 MTP 输出所有 residual codebooks"，只看 paper 完全会理解错）。
**必须把 paper 和参考实现对照**，以参考实现为准还原真实架构。

##### 参考实现来源的优先级（从高到低）

| 优先级 | 来源 | 获取方式 | 可信度理由 |
|---|---|---|---|
| **1. 论文作者放出的官方仓库** | paper 的 "Code" 字段 / README 里的 GitHub 链接 | `git clone <repo> --depth 1` | 作者自己的实现，最权威 |
| **2. HuggingFace `transformers` 最新源码** | `transformers/models/<name>/modeling_<name>.py` | `pip show transformers` 找路径；或 `curl https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/<name>/modeling_<name>.py` | 社区 PR 已 review 过，基本不会错；**最精准的层级展开**；有完整的 `Config` 类可以查 dim/heads/experts |
| **3. HuggingFace Model Card 的 `modeling_xxx.py`** | `https://huggingface.co/<org>/<model>/tree/main` 里的 `.py` 文件 | `huggingface_hub.snapshot_download` 或 `curl https://huggingface.co/<org>/<model>/raw/main/modeling_<name>.py` | "trust_remote_code" 的文件，作者直接上传；常常**比 transformers 主仓库更新**（尤其新模型） |
| **4. 推理框架的 modeling 实现** | `sglang/python/sglang/srt/models/<name>.py`、`vllm/vllm/model_executor/models/<name>.py` | `git clone` 或 raw URL | 推理侧实现：tensor-parallel 布局、KV cache shape、sampling 细节比 HF 更清晰；但可能少训练路径代码 |

##### 查找步骤 (按模型名搜索)

```bash
# 1. 先尝试论文作者放的官方 repo (扫 arXiv abstract 和 README)
gh search repos "<model_name>" --owner="<likely-org>" --limit 5

# 2. HF transformers 主仓库
rg -l "<ModelName>PreTrainedModel" ~/.cache/huggingface/ 2>/dev/null
# 或直接:
curl -sL "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/<name>/modeling_<name>.py" | head

# 3. HF Model Card (新模型通常 transformers 还没 merge)
curl -sL "https://huggingface.co/api/models/<org>/<model>" | jq '.siblings[].rfilename' | grep -i modeling

# 4. SGLang / vLLM 实现
find /apps/feiyue/upstream/sglang/python/sglang/srt/models/ -iname "*<name>*"
find /apps/feiyue/upstream/vllm/vllm/model_executor/models/ -iname "*<name>*"
```

##### 对照解读的执行清单

把 paper 的架构描述和实现并排读，**重点校验**以下 6 类信息（paper 最常
错或略写的地方）：

| 核验项 | Paper 常见写法 | 代码里真实情况 |
|---|---|---|
| **每个子模块的层数** | "~20 layers" | `num_hidden_layers` 精确值 |
| **hidden_size / intermediate_size / head_dim** | 通常省略 | `config.hidden_size`, `config.intermediate_size`, `hidden_size // num_attention_heads` |
| **GQA 比例** | "use GQA" | `num_attention_heads` vs `num_key_value_heads` |
| **MoE 细节** | "MoE" | `num_experts`, `num_experts_per_tok`, `shared_expert_intermediate_size`, `moe_intermediate_size` |
| **是否有 QK-Norm / 归一化变体** | 常不提 | `q_norm`, `k_norm`, `RMSNorm` 的应用位置 |
| **Router 实现细节** | "top-k routing" | `softmax` 在 top-k 前还是后？aux-loss 权重？shared expert 的 gating function？|
| **Position encoding 细节** | "RoPE" | `rope_theta`, `rope_scaling` 类型 (YaRN / linear / NTK-aware)，3D/TM-RoPE 的 angle 分配 |
| **子模块之间的 tensor shape** | 省略 | 从 `forward()` 的 reshape/view 一眼可见 |

##### 图绘制流程（实现导向）

1. **先读实现**，画出每层 Transformer block 的"真实" forward pass
 （包括所有 reshape、broadcast、residual 走向）
2. **再看 paper figure**，对齐命名和高层抽象
3. **发现不一致时以实现为准**，但在笔记中明确指出 "paper 叫 X，实现里叫 Y"
 并解释哪个更准确
4. **记录 `config.json` 中所有关键 hyperparameter**（从 HF Model Card 下载），
 把数值塞进架构图的每个框里
5. **多模块系统优先用 drawio** (`{{drawio:filename.drawio}}` 指令嵌入到笔记)，
 不能用单张 PNG 表达完整流水线

> **反面教训**：Qwen3-Omni 解读的第一版笔记只读了 paper，把 "MTP module"
> 描述成 "一次性输出 residual codebooks 的轻量模块"。对照 vLLM/SGLang 的
> modeling 文件才发现实际是 **5 层 dense transformer + 15 次 AR 迭代 +
> 15 套独立 LM head/embedding**，两者是本质不同的架构。**必须读代码**。

#### 1b. Architecture Diagram (MANDATORY — 一律使用 drawio)

**规则：任何涉及模型结构、数据流、子模块拓扑、attention / FFN 内部展开、
pipeline 多段协作 的示意图，一律使用 drawio 绘制，不允许用 ASCII box art、
不允许只用 markdown 代码块画盒子。** ASCII 图在笔记页面（特别是 HTML 渲染后）
会出现对齐错位、换行断裂、无法缩放、无法点击等严重可读性问题。

##### 强制流程

1. **新建一个多页 drawio 文件** — 顶层流水线一页 + 每个子模块一页。
 文件放在 `/apps/feiyue/upstream/zhaifeiyue.github.io/assets/<paper-id>_arch.drawio`
2. **每个子模块页都要画到"每层内部"粒度** —
 - 顶层页：完整 forward pass 从 input 到 output，每个箭头标 tensor shape
 - 子模块页 (例如 Thinker)：
 * 整体 forward (embedding + N × decoder layer + norm + head)
 * 展开一层 decoder 的 **Attention 内部** (QKV proj / reshape /
 QK-Norm / RoPE / attn / O proj / residual)
 * 展开一层 decoder 的 **FFN / MoE 内部** (router / top-k / expert /
 shared expert / combine)
3. **每个组件框里写**：中文功能 + 英文名 + 关键维度（从 `config.json` 精确值）
4. **每条边上标 tensor shape** — e.g. `[seq, 28, 128]`, `Q:[seq,3584]`
5. **关键设计决策用粗体/特殊颜色标注** — "GQA 28Q/4KV"、"QK-Norm per head"、
 "128E top-8"、"expert hidden=768"
6. **在笔记中通过 `{{drawio:<file>.drawio#page=N&height=NNN}}` 嵌入** —
 每个 section 嵌一次，`page=N` 指定默认页（0-indexed），`height=NNN` 可选
7. **同一个 drawio 文件多处嵌入时 sync.sh 会自动 dedup** —
 XML 只内嵌一次，多处引用，页面大小不会膨胀

##### 例外（仅允许 ASCII 的场景）

以下**且只有以下**场景允许用 markdown 代码块里的 ASCII：
- **训练阶段流程图**（3-4 个步骤的线性 pipeline, e.g. `SFT → Distill → RL`）
- **线性管线的文字描述**（不涉及空间布局）
- **表格**（markdown table 直接胜出）
- **latency breakdown 路径**（简单的竖直箭头链）

**反例（严禁）**：
- ❌ 在 markdown 代码块里画 decoder block 的 attention 内部（ASCII 画
 QKV proj + heads + GQA broadcast 永远对不齐）
- ❌ 画多模态流水线的 input / encoder / backbone / decoder / vocoder 并排结构
- ❌ 画任何有"分支 / 汇聚 / 跨行连线"的拓扑

##### 工具选项

- drawio 在线编辑: https://app.diagrams.net/ (直接打开 .drawio 文件)
- drawio Desktop: https://github.com/jgraph/drawio-desktop
- 笔记中预览：`sync.sh` 会用官方 viewer-static.min.js 在网页里原生渲染，
 支持缩放、平移、页签切换、点击编辑

##### drawio 文件命名与组织

- **单模型**：`<paper-id>_arch.drawio`（e.g. `2509.17765_arch.drawio`）
- **多模型对比**：`<paper-id>_compare.drawio`
- 每个 drawio 文件**必须多页**，每页一个 diagram id，有意义的 name（页签会显示）
- 复杂的子模块内部可以进一步拆多页（例如 "4.1 Overview / 4.2 Attention / 4.3 FFN"
 同一 page 里三张子图，或分到 page 4/5/6）

##### 反面教训

Qwen3-Omni 解读的第一版笔记每个子模块都画了 ASCII 盒子图，HTML 渲染后
完全错位，而且无法缩放查看 "Linear(2048→4608)" 这种维度标注。改用
`qwen3_omni_model_arch.drawio` 的 7 页 drawio 后，读者可以：
- 滚轮缩放看任意 Linear/Conv 层的精确维度
- 顶部页签一键切换不同子模块
- 拖动平移看大图全局
- 点击工具栏一键在 diagrams.net 打开编辑

这就是为什么**架构图一律用 drawio**。
- 如果有多个协作子模块 (encoder / decoder / vocoder / ...)，用 drawio
 的多 page 分别展开每个子模块的内部结构

#### 1b. Architecture Details

- Base architecture: dense Transformer, MoE, hybrid (Mamba + Attention), other?
- Key architectural changes vs. standard Transformer:
 - Attention variant: MHA, GQA, MQA, MLA, linear attention, sliding window?
 - Position encoding: RoPE, ALiBi, learned, NTK-aware? What modifications?
 - Normalization: LayerNorm, RMSNorm, DeepNorm? Pre-norm or post-norm?
 - Activation: SwiGLU, GeGLU, ReLU?
 - Vocab size and tokenizer design choices?
- For MoE: number of experts, top-k routing, shared experts, load balancing?
- For multimodal: encoder architecture, adapter design, modality alignment strategy

#### 1c. Architecture Comparison Table

Compare key architecture decisions vs. 2-3 competitor models at similar scale:

| Component | This Model | Competitor A | Competitor B |
|-----------|-----------|-------------|-------------|
| Attention | | | |
| FFN | | | |
| Position Enc | | | |
| Normalization | | | |
| Vocab Size | | | |
| Context Length | | | |
| KV Cache / token | | | |

For each difference, explain **why** this paper chose differently (cite
ablation evidence if available).

### 2. Scale & Training (MANDATORY — 核心重点，等同于架构)

**训练分析是 LLM 论文的第二优先级，和架构同等重要。** 任何关于模型的 paper，
"怎么训练出来的" 和 "模型长什么样" 是两个必须展开的核心话题。缺失或潦草的
训练章节会让整篇解读失去价值。必须回答以下所有子问题，缺失的项明确标注
`[论文未披露]` 或 `[需反推]`，**绝不能一句"训练 2T tokens"草草带过**。

#### 2a. Scale (规模清单，必须是一张表)

| 维度 | 数值 | 备注 |
|---|---|---|
| 总参数量 | e.g. 30B | dense 按总参；MoE 分 total / active 两列 |
| 激活参数量 (MoE) | e.g. 3B | 单 token forward 实际激活的参数 |
| 层数 × d_model | e.g. 28 × 2048 | 可从架构表引过来 |
| Expert 数 × top-k | e.g. 128E top-8 | MoE 必填 |
| Context length (训练) | e.g. 8K → 32K | 区分预训练原生长度和扩展后长度 |
| Vocab size | e.g. 151 643 | |

#### 2b. Training Data (数据配方，必须具体到量级和比例)

- **Total tokens (预训练)**：按阶段拆分（e.g. S1 0.5T, S2 2T, S3 0.3T）
- **Data mix (按模态/类型的 token 数和百分比)**：
 - Text: e.g. 0.57T (web / code / math / books 各占多少)
 - Audio / Image / Video：逐个模态
 - **多语言分布**：前 N 大语言各多少 token
- **Quality filtering pipeline**：去重 / 分类器 / 毒性过滤 / 困惑度筛选 …
 (论文常在附录写，但是非常关键，必须挖出来)
- **Data scheduling**：课程学习？阶段比例是否变化？long-context 阶段的数据如何构造？
- **Synthetic data / distillation data**：有没有用更大模型生成的数据？多少？
- **Data contamination 检查**：和 eval set 的重复率控制在多少？

> 如果论文说"2T tokens"但没写 mix，至少要从论文 figure/table 中的饼图反推，
> 或者和同作者的前代模型对比给出量级估计。

#### 2c. Training Compute (算力账单)

- **总 GPU-hours** (或 H100-equivalent hours)
- **硬件**: 具体 GPU 型号、节点数、互联（NVLink/IB/RoCE）
- **Training efficiency**: MFU / HFU（Model/Hardware FLOPs Utilization），
 tokens/s/GPU，具体数字
- **Wall-clock time**：从第一个 token 到 release 花了多久？
- **成本估算**：用 $2/GPU-hr 反推的美元成本（哪怕论文没说）

#### 2d. Training Recipe (训练流程，按阶段清单)

按阶段写清楚**每一阶段的目标、数据、时长、关键 hyperparameter**。典型结构：

| Stage | Goal | Data (tokens) | LR schedule | Context | Key techniques |
|---|---|---|---|---|---|
| S1 Encoder Alignment | 让 encoder 对齐 LLM | e.g. 100B | cosine 2e-4 | 4K | LLM frozen, adapter-first |
| S2 General Pre-train | 主预训练 | 2T | cosine warmup | 8K | 全参数训练 |
| S3 Long Context | 扩展上下文 | 300B | constant 5e-5 | 32K | RoPE scaling, long-data upsample |
| SFT | 指令遵循 | 5M samples | linear decay | 32K | ChatML, multi-turn |
| Preference (DPO/GRPO/…) | 对齐偏好 | 500K pairs | 1e-6 | 8K | 具体算法 + reward model |
| Distillation | Weak-to-strong 知识蒸馏 | — | — | — | teacher model, KL loss |

**对每一阶段都要回答**:
1. 为什么需要这个阶段？(前一阶段的不足是什么)
2. 数据是怎么构造的？有没有特殊的筛选 / 合成？
3. 训练目标是什么？(next-token / MLM / DPO loss / custom loss)
4. 有没有该阶段特有的 trick？(e.g. chunked attention, gradient accumulation)

#### 2e. Training Techniques (技术清单，必须逐项列出)

LLM 训练是一整套 engineering systems 的组合。必须把论文里**提到的所有训练技术**
逐项列出并解释为什么用。典型类别：

**分布式并行**：
- Tensor Parallel (TP) / Pipeline Parallel (PP) / Expert Parallel (EP) / Context Parallel (CP)
- 具体配置 (e.g. TP=8, PP=4, DP=16, EP=8)
- 为什么这样切？(通信量 / 内存 / bubble 权衡)
- 用的框架：Megatron-LM, DeepSpeed, FSDP, 自研

**数值精度**：
- 训练精度：BF16 / FP16 / FP8 (E4M3 / E5M2)
- 哪些算子走低精度？哪些保 FP32 (e.g. master weights, optimizer states, norm)
- Loss scaling 策略

**优化器 & 调度**：
- Optimizer: AdamW / Lion / Muon / Sophia / Shampoo
- β₁ β₂ 具体值，weight decay，gradient clipping
- LR schedule: cosine / WSD / linear warmup 的具体比例和 peak LR
- Batch size schedule (有些论文会 rampup)

**稳定性技巧**：
- QK-Norm / Z-loss / muP / LayerScale / DeepNorm
- Router loss (for MoE) / load balancing loss 权重
- Gradient spike detection & recovery
- Sandwich norm / post-norm vs pre-norm 讨论

**数据/采样**：
- Sequence packing / in-batch mixing
- Sample weight by domain / temperature sampling
- Curriculum learning (simple→hard)

**Long-context 技术**：
- RoPE base scaling (θ), YaRN, NTK-aware
- Ring Attention / Blockwise Attention
- Document masking 策略

**后训练技术**：
- SFT 数据构造（任务配比、多轮、工具调用）
- RLHF / DPO / GRPO / GSPO / PPO / Online-policy 的具体 loss
- Reward model 设计（paired / pointwise / rubric-based / process-supervised）
- Strong-to-Weak / Weak-to-Strong distillation 的 teacher/student 设置

> **解读要求**：不要只列名字，要对**每一项技术**回答："为什么选这个而不是备选？"
> e.g. "选 Muon 而非 AdamW 因为 Muon 在 non-diagonal 梯度上收敛更快" (带引用)

#### 2f. Scaling Law & Training Ablations

- 论文有没有提供 scaling law 曲线？遵循 Chinchilla 还是有偏离？
- Data-to-params ratio 是多少？(e.g. 60 tokens/param vs Chinchilla 20)
- 关键消融：某个训练 trick 对 final performance 的贡献？
- 失败尝试：论文有没有诚实写出 "what didn't work"？（这类信息最珍贵）

#### 2g. Training Barrier (核心壁垒识别)

每篇 model paper 必须识别 **"真正难复现的那一件事"**：
- 是数据配方？(e.g. DeepSeek 的 code/math 数据清洗)
- 是训练技巧？(e.g. Qwen3-Omni 的 "adapter-first, encoder-second" 两步策略)
- 是基础设施？(e.g. 5D parallelism + FP8 训练栈)
- 是 post-training 管线？(e.g. 多阶段 distillation + RL)

这个 "核心壁垒" 必须在本章节有单独一段明确指出，而不是淹没在技术列表里。

### 3. Quantization & Compression

This is a key focus area for LLM papers. Analyze thoroughly:

- **Weight quantization**: what methods tested? (GPTQ, AWQ, SqueezeLLM, round-to-nearest)
 - Bit widths: FP8, INT8, INT4, FP4, NF4? Per-channel vs. per-group vs. per-tensor?
 - Group size and its impact on accuracy?
 - Accuracy degradation at each level (MMLU, perplexity, task-specific)?
- **Activation quantization**: FP8/INT8 for compute? Dynamic vs. static scaling?
- **KV cache quantization**: what bit width for KV cache? Impact on long-context quality?
- **Quantization-aware training** vs. **post-training quantization**?
- **Memory savings**: model size at FP16 vs. INT8 vs. INT4, KV cache reduction
- **Serving implications** (model-side, not system-side — system papers go to `framework`):
 - KV cache size per token (derive from dimensions if not stated)
 - Memory footprint at inference: model weights + KV cache at typical batch
 - Prefill vs. decode characteristics: compute-bound vs. memory-bound?
 - Context length support: native, extended via RoPE scaling?
 - Speculative decoding compatibility?

### 4. Performance

- Benchmark results: MMLU, HumanEval, GSM8K, MATH, etc.
- Comparison to models of similar size — fair? (same training data budget)
- Long-context evaluation: NIAH, LongBench, etc.?
- Efficiency metrics: tokens/second, latency, cost per token?
- Multilingual / code / reasoning breakdown?

### 5. Key Innovations

For each claimed innovation:
- What problem does it solve?
- How does it work mechanically?
- What is the measured impact? (ablation)
- Is it architecture-specific or generally applicable?

### 6. Infrastructure Impact

| Layer | Question |
|-------|----------|
| Algorithm | New training techniques introduced? Post-training recipe? |
| Kernel | Needs custom attention kernels? New GEMM shapes? Sparse kernels for MoE? |
| Framework | Serving framework requirements? Tensor/pipeline/expert parallelism needs? |
| Agent | Instruction following quality? Tool use capability? Long context for RAG? |
| Hardware | Memory bandwidth vs. compute requirements? Multi-node necessity? |

### 7. Deployment Considerations

- Minimum hardware to serve at reasonable quality (after quantization)?
- Recommended parallelism strategy (TP, PP, EP)?
- Open weights? License restrictions?
- Fine-tuning friendly? LoRA/QLoRA compatibility?

### 8. Position in Model Landscape

- How does it compare to current SOTA at same size class?
- What gap does it fill? (open-source parity, multilingual, code, long context)
- What is the next step this model enables?

---

## 作者证明 — LLM/Architecture 特定要求

> Inherits the **作者证明** requirement from §2b. LLM papers' 作者证明 is
> typically a **scaling-law fit, capacity derivation, or component-level
> param/FLOP breakdown** that proves the architectural choices are
> consistent with a quantitative principle (not just "we tried it and
> it worked").

**Typical 作者证明 shape**:
- Scaling-law fit: loss vs FLOPs / params / data, with fitted exponents
- Param breakdown: each module's contribution to total params, justifying
 hidden dim / heads / experts ratio choices
- Capacity / arithmetic intensity per token: explaining why MoE expert
 count, GQA group ratio, MLA latent dim were chosen
- For attention variants: KV-throughput vs context-length curves with a
 closed-form expression (see PrfaaS Eq.(1) for $\Phi_{\text{kv}}$)
- For quantization papers: information-theoretic bound + measured loss

**Where it lives**:
- "Scaling Laws" / "Architecture Search" / "Design Choices" §
- Param tables broken down by sub-module
- Compute budget table (training FLOPs ÷ params ÷ tokens)

**Required content**:
1. **Reproduce the fit equation** + R² + range of validity
2. **Component param/FLOP breakdown**: each design knob (dim, heads,
 experts, ratio) → param contribution → why this choice
3. **Comparison to canonical references**: Chinchilla (params:tokens≈1:20),
 DeepSeek-V3 (active vs total params), Llama-3 (head-dim ÷ KV-heads)
4. **Cross-reference implementation** (mandatory for llm category): does
 the paper's claimed dim/heads/experts match `config.json`?
5. **Empirical-only fallback**: if no scaling fit, reproduce the
 architecture-search ablation grid + flag which axes are decisive
6. **Attack surface**: scaling laws fit on which data range? Extrapolation
 risk? Quantization bound assumes which distribution?

**Anti-patterns**:
- ❌ Quoting "follows scaling laws" without the actual exponent
- ❌ Param breakdown that doesn't add up to the headline param count
- ❌ Skipping cross-ref to `modeling_*.py` because "the paper says so"
 (the paper often mis-labels — see Qwen3-Omni "MTP" incident in SKILL.md)

## ✅ End-of-Deep-Read MANDATORY Checklist

This guide defines **8** numbered sections. Before declaring
Phase 5 complete, verify in your notes file:

- [ ] **All 8 `### N. Title` sections from this guide are answered**
 under `## Deep Analysis (llm)` in the notes. Missing sections are
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

