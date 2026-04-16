---
name: paper-reader
description: >-
  Automated orchestration skill for reading AI infrastructure papers. Given a
  paper (arXiv ID, URL, blog, or local PDF), automatically runs the full
  pipeline: fetch content, summarize, classify, deep read with category-specific
  analysis, save all results, and trigger synthesis for trend/relationship
  updates. Use when the user gives a paper to read, shares an arXiv link, a blog
  post, or any AI-related article. Also triggers when user says "读paper",
  "帮我读", "read paper", "精读", "粗读", "解读", or similar phrases about
  reading/analyzing a paper.
---

# AI Infra Paper Reader — Automated Pipeline

This is an **orchestration skill**. When the user provides a paper or article,
execute ALL steps below automatically without waiting for confirmation between
steps. The user provides an input; you deliver the complete analysis.

```
Input (arXiv ID / URL / PDF path)
  │
  ├─ Phase 1: Fetch ──────────── download content, extract text
  ├─ Phase 2: Summarize ──────── core contribution, summary, findings
  ├─ Phase 3: Classify ───────── primary category + secondary tags
  ├─ Phase 4: Save Overview ──── write to papers.json + notes file
  ├─ Phase 5: Deep Read ──────── load category guide, run deep analysis
  ├─ Phase 6: Save Images ────── download key figures from the paper
  ├─ Phase 7: Save Deep Notes ── append deep analysis to notes, update DB
  ├─ Phase 8: Synthesis ──────── call paper-synthesis for incremental update
  ├─ Phase 9: Generate HTML ──── rebuild overview + category HTML pages
  ├─ Phase 10: Paper Reader ──── generate per-paper cognitive HTML reader
  └─ Phase 11: Publish ────────── sync to GitHub Pages & push
```

---

## Phase 1: Fetch Content

Detect input type and fetch accordingly.

**GitHub repository** (URL containing `github.com` pointing to a repo, NOT a paper PDF):
→ This is source code, NOT a paper. **Stop this pipeline** and switch to the
  `code-reader` skill instead. Read `~/.cursor/skills/code-reader/SKILL.md`
  and execute its pipeline.

**arXiv paper** (ID like `2402.03300` or URL containing `arxiv.org`):

```bash
curl -s "https://export.arxiv.org/api/query?id_list=ARXIV_ID" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
entry = root.find('a:entry', ns)
if entry is None: sys.exit('Paper not found')
title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
published = entry.find('a:published', ns).text[:10]
cats = ', '.join(c.get('term') for c in entry.findall('a:category', ns))
summary = entry.find('a:summary', ns).text.strip()
print(f'Title: {title}')
print(f'Authors: {authors}')
print(f'Published: {published}')
print(f'Categories: {cats}')
print(f'Abstract: {summary}')
"
```

Then read full content: `WebFetch` on `https://arxiv.org/html/ARXIV_ID` first.
If HTML unavailable, try `https://arxiv.org/pdf/ARXIV_ID`.

**Blog / article URL**: `WebFetch` directly on the URL.

**Local PDF**: `Read` tool on the file path.

**Extract from metadata**: title, authors, date, source URL. These go into the
DB entry later.

---

## Phase 2: Summarize (Summary)

### Role: 学术论文首席解读者

You are a patient mentor who transforms obscure academic language into
clear, logical, accessible Chinese explanations. When encountering
technical terms (e.g. Attention Mechanism, Transformer), **always explain
with a real-life analogy/example first**, then give the academic definition.
Use bold for emphasis, `>` blockquotes for concept explanations.

### 2a. Core Three Questions (核心三问)

Answer each in ≤3 sentences, cut straight to the point:

- **Q1: 这篇论文试图解决什么核心痛点/问题？** (What is the problem?)
- **Q2: 作者提出了什么新的"杀手锏"方法/架构？** (What is the method?)
  — State the core innovation and how it differs from conventional approaches.
- **Q3: 最终效果/结论如何？** (What are the results?)
  — Include 1-2 key data points proving effectiveness.

### 2b. Logic Flow (逻辑故事还原 — 约束推导型)

Reconstruct the author's thinking path using **constraint-driven derivation**,
NOT just method description. The reader should understand the shape of the
design space — which paths are blocked and why — before seeing the solution.

- **时代定位 (Era Positioning):** Where does this paper sit in the
  field's evolution? Is it picking low-hanging fruit or entering "deep
  water"? What prior optimizations have been exhausted? Example:
  "2024年LLM推理的low hanging fruits几乎被采摘殆尽，NanoFlow代表
  优化进入了深水区——从粗粒度的算子融合转向GPU资源的精细管理。"
  This gives readers a sense of WHY this paper matters NOW.
- **背景 (Context):** Why couldn't existing methods solve this well? Their deficiencies.
- **约束推导 (Constraint Derivation):** What design alternatives exist? For each
  rejected alternative, explain **why it fails** with mathematical or hardware
  proof. Use the "为何不可X？" format to expose constraints that narrow the
  design space. Example: "为何不可 per-channel 量化 K？因为 QK^T 的结果
  维度是 N×N，没有 d 维度来应用 per-channel 的 scale factor。"
  — Build a **feasibility matrix** when applicable (rows=alternatives, cols=criteria).
- **破局 (Insight):** What was the author's "aha moment"? The core intuition.
  — 👉 Insert a plain-language analogy to explain this intuition.
- **核心技术壁垒 (Core Technical Barrier):** Identify the ONE hardest-to-
  replicate technique that makes the whole method actually work in
  practice. This is not the high-level idea but the low-level engineering
  insight. Example: "NanoFlow的核心壁垒是custom execution unit scheduling
  ——限制kernel执行的SM个数。仅用108个SM中的35个(32%)，网络
  kernel即可实现92%峰值性能——这个非线性关系是整个方案能work的
  根本原因。" Every paper has one; find it and call it out explicitly.
- **质疑假设 (Challenge Assumptions):** What statistical or empirical
  assumptions does the method rely on? Are they validated by independent
  evidence? Under what conditions might they break? When uncertain,
  explicitly say so: "这部分笔者也不是十分确定" — honest uncertainty
  builds reader trust more than false confidence.
- **设计绑定批判 (Design Binding Critique):** What prerequisites does
  the method FORCE? List all forced dependencies (e.g. "NanoFlow
  强制绑定了chunked prefill和张量并行——这两个条件不再绑定时，
  很多设计可能会面临挑战"). Discuss what happens if these prerequisites
  are removed or changed.
- **拆解 (Deconstruction):** Concrete steps from input to output (numbered 1, 2, 3 list).
- **实践上下文 (Deployment Context):** Where does this method apply in
  real systems? (e.g. prefill vs decode, training vs inference, which
  GPU architectures benefit most, ecosystem integration)
- **生态影响追踪 (Ecosystem Influence):** Trace the paper's influence
  on downstream systems. Has it been adopted by vLLM, SGLang,
  TRT-LLM, or other frameworks? Did subsequent papers (e.g.
  DeepSeekV3) adopt similar ideas? This connects the paper to the
  living ecosystem rather than treating it as an isolated work.

### 2c. Key Figures & Tables (核心图表)

**论文解读以图表、算法为中心。** 图表是论文最浓缩的信息载体。

#### Figure Importance Ranking (按优先级选择)

按以下优先级选择 **3-5 张最重要的图**：

1. **架构图 / 系统总览图** — 论文的"灵魂图"，通常是 Figure 1 或 2
2. **核心原理图** — 解释方法如何工作的机制图（数据流、pipeline、调度策略）
3. **算法伪代码** — Algorithm 1, 2 等，必须完整提取并逐行解读
4. **关键对比实验结果图** — 主实验的 bar chart / line chart / heatmap
5. **消融实验图** — 证明各组件贡献的图
6. **Motivation 图** — 论文开头用来说明问题的图（bottleneck 分析、profiling）

判断重要性的信号：
- 论文用**大篇幅文字解释**的图（超过半页描述）
- 被论文**多次引用**的图（"As shown in Figure X"出现 3+ 次）
- **位置靠前**的图（前 3 张图通常最重要）
- **独占一整页或半页**的图

#### Critical Rules for Figures

1. **图片必须和上下文对应** — 每张图必须出现在其解释文字的正上方或正下方，
   用 `### Figure N: title` 标题包裹，图片和描述在同一区块内。
2. **只要重要的图** — 不要下载论文中所有的图，只下载符合上述优先级排序的
   3-5 张核心图。Appendix 中的图、补充实验的小图、与主要贡献无关的图，
   一律不要。
3. **没有对应描述的图片不要** — 如果你无法为某张图写出有意义的
   "What it shows / Why it matters"，说明它不够重要，不要下载。
4. **只下载你在笔记中引用的图** — 笔记中每个 `### Figure N:` 标题
   必须对应一张已下载的图片。反过来，每张下载的图必须在笔记中有对应的
   `### Figure N:` 标题和完整描述。孤立的图片（没有描述）会被丢弃。

#### Figure Output Format

For each key figure, use this EXACT format (图片和描述紧挨在一起):

```markdown
### Figure N: {title}
![Figure N: {title}](../images/{paper-id}/figN-{short-name}.png)
**What it shows**: {one-sentence plain-language description}
**Why it matters**: {what understanding this figure unlocks}
**Detailed description**: {3-5 sentences describing all components,
arrows, data flow, labels — enough for someone who hasn't seen the
figure to reconstruct it mentally}
```

Image download — ONLY download the figures you describe:
```bash
mkdir -p ~/.cursor/paper-db/images/{paper-id}
# arXiv HTML images: https://arxiv.org/html/{id}v{N}/x{N}.png
# Map figure order to x{N}.png (Figure 1 → x1.png, Figure 4 → x4.png)
# ONLY download figures that have a ### heading in your notes
curl -sL -o figN-{name}.png "https://arxiv.org/html/{id}v{version}/xN.png"
file figN-{name}.png  # verify PNG, not HTML error page
```

#### Algorithm / Pseudocode Output

For each Algorithm block in the paper:
- **Algorithm ID & title**: e.g. "Algorithm 1: Inter-PE Scheduling"
- **Complete pseudocode**: reproduce in markdown code block, exactly as in paper
- **逐行解读**: explain each key line in Chinese — what it does and why
- **复杂度分析**: time/space complexity if applicable
- **与代码的对应**: if source code is available, map pseudocode to actual implementation

#### Table Importance Ranking

按以下优先级选择 **3-5 张最重要的表**：

1. **主实验结果表** — 与 SOTA/baseline 的全面对比（通常是论文最大的表）
2. **消融实验表** — 各组件贡献的量化分析
3. **模型/系统配置表** — 参数规格、硬件配置
4. **成本/效率表** — training cost、GPU hours、throughput

#### Table Output Format

For each key table:
- **Table ID & title**: e.g. "Table 2: Pipeline bubble comparison"
- **What it compares**: what are the rows and columns?
- **Key takeaway**: the single most important number/trend
- **Reproduce the table** in markdown format with actual data.
  **Bold** the best result in each column.

### 2d. Key Details & Takeaway (关键细节与启示)

- **技术细节补充:** 1-2 most critical implementation details (e.g. special loss
  function, data processing trick) that should not be overlooked.
- **一句话总结:** One sentence summary suitable for presenting at a lab meeting
  tomorrow — the sharpest possible distillation.

### 2e. Structured Extraction (for DB storage)

Also produce these fields for Phase 4:

1. **Core contribution** — one sentence, the primary novelty
2. **Summary** — 2-3 paragraphs: motivation → method → results
3. **Key findings** — bullet list (3-6 items)
4. **Limitations** — what the paper does NOT address or gets wrong

---

## Phase 3: Classify

Assign exactly one **primary category** and 2-5 **secondary tags**.

| Category | Scope | Focus Areas | Examples |
|----------|-------|-------------|---------|
| `algorithm` | Training algorithms, optimization, loss functions, RL | Training recipe, optimizer design, RL methods, data strategies | GRPO, DPO, RLHF, Adam variants, MoE routing, data mixing |
| `kernel` | GPU kernels, hardware optimization, numerical formats | Kernel implementation, hardware mapping, numerical precision | FlashAttention, Triton kernels, CUTLASS, CK, FP4/FP8 GEMM |
| `framework` | Inference & training systems, scheduling, orchestration, serving | Request scheduling, KV cache management, pipeline/tensor/expert parallelism, memory management, batching, distributed training, serving optimization | vLLM, SGLang, TRT-LLM, DeepSpeed, Megatron, DualPipe |
| `llm` | Model architecture, quantization | Architecture design choices, attention variants, position encoding, MoE design, model quantization (FP8/INT4/FP4), weight compression, scaling laws | Llama, Qwen, DeepSeek, Gemma, MLA, GQA, GPTQ, AWQ |
| `agent` | Agentic systems, tool use, planning, multi-agent | Agent runtime, tool calling, orchestration, memory, multi-agent communication | ReAct, code agents, tool calling, A2A, MCP |
| `cluster` | Networking, cluster topology, interconnects, storage systems | Network architecture (IB, RoCE, NVLink, PCIe), cluster topology, collective communication, congestion control, storage systems, RDMA, load balancing, failure recovery | Rail-optimized, fat-tree, 3FS, NCCL, RCCL, Ultra Ethernet, DGX SuperPOD |
| `hardware` | GPU/accelerator architecture, whitepapers, ISA, chip specifications | Architecture design, compute unit internals, matrix/tensor cores, memory subsystems, new data types, ISA changes, die design, process node, power/thermal, interconnect fabric | NVIDIA Blackwell/Hopper/Ampere whitepaper, AMD CDNA4/CDNA3 whitepaper, Intel Gaudi, HBM specs, NVLink/IF specs |

Also assess **infrastructure impact** — for each of the OTHER categories,
one sentence on how this paper affects that layer. `"N/A"` if no connection.

---

## Phase 4: Save Overview

Write two things:

### 4a. Append to `~/.cursor/paper-db/papers.json`

```python
import json, os, datetime
DB = os.path.expanduser("~/.cursor/paper-db/papers.json")
with open(DB) as f:
    db = json.load(f)
entry = {
    "id": "PAPER_ID",
    "title": "...",
    "authors": ["..."],
    "date": "YYYY-MM",
    "source": "arXiv",          # or "blog", "article", "pdf"
    "url": "https://...",
    "category": "algorithm",    # primary
    "secondary_tags": ["..."],
    "core_contribution": "...",
    "summary": "...",
    "key_findings": ["..."],
    "limitations": ["..."],
    "infra_impact": {
        "algorithm": "...",
        "kernel": "...",
        "framework": "...",
        "llm": "...",
        "agent": "..."
    },
    "related_paper_ids": [],
    "open_questions": [],
    "read_depth": "deep",
    "date_read": datetime.date.today().isoformat()
}
existing = [i for i, p in enumerate(db["papers"]) if p["id"] == entry["id"]]
if existing:
    db["papers"][existing[0]] = entry
else:
    db["papers"].append(entry)
with open(DB, "w") as f:
    json.dump(db, f, indent=2, ensure_ascii=False)
```

### 4b. Create notes file `~/.cursor/paper-db/notes/{paper-id}.md`

```markdown
# {Title}
> {Authors} | {Date} | {Source URL}
> Category: {category} | Tags: {tags}
> Read: {date_read}

## Core Contribution
{one sentence}

## Summary
{2-3 paragraphs}

## Key Findings
- ...

## Key Figures
![Figure N: description](../images/{paper-id}/figN.png)
- **What it shows**: ...
- **Why it matters**: ...

## Key Tables

### Table N: {title}
| Col1 | Col2 | Col3 |
|------|------|------|
| ...  | ...  | ...  |
**Takeaway**: ...

## Limitations
- ...

## Infrastructure Impact
- Algorithm: ...
- Kernel: ...
- Framework: ...
- LLM: ...
- Agent: ...
```

**Do NOT stop here.** Proceed immediately to Phase 5.

---

## Phase 5: Deep Read (Automatic)

Based on the primary category from Phase 3, **read the corresponding guide
file** and work through ALL its analysis prompts against the paper content.

| Category | Read this file |
|----------|---------------|
| algorithm | [deep-algorithm.md](deep-algorithm.md) |
| kernel | [deep-kernel.md](deep-kernel.md) |
| framework | [deep-framework.md](deep-framework.md) |
| llm | [deep-llm.md](deep-llm.md) |
| agent | [deep-agent.md](deep-agent.md) |
| cluster | [deep-cluster.md](deep-cluster.md) |
| hardware | [deep-hardware.md](deep-hardware.md) |

**Procedure**:
1. Read the guide file using the `Read` tool
2. For EACH analysis section in the guide, write the analysis for this paper
3. Collect all analysis into a structured deep-read output

This step is NOT optional. Every paper gets a deep read.

During the deep read, identify **3-5 important figures** from the paper
(architecture diagrams, result charts, key illustrations). Record their
URLs or descriptions for Phase 6.

---

## Phase 6: Save Images

Download important figures identified during the deep read.

```bash
mkdir -p ~/.cursor/paper-db/images/{paper-id}
curl -s -o ~/.cursor/paper-db/images/{paper-id}/fig1.png "IMAGE_URL_1"
curl -s -o ~/.cursor/paper-db/images/{paper-id}/fig2.png "IMAGE_URL_2"
```

For arXiv HTML papers, image URLs are typically at:
`https://arxiv.org/html/{arxiv-id}v{N}/extracted/{hash}/figure.png`

In the notes file, reference them as:
`![Figure N: caption](../images/{paper-id}/figN.png)`

If images cannot be downloaded (e.g., blog with JS-rendered charts), note
the figure description in text and skip the download.

---

## Phase 7: Save Deep Notes

Append the deep analysis to the existing notes file from Phase 4:

```markdown
---

## Deep Analysis ({category})

{All sections from the category-specific guide, filled in}

## Key Figures

![Figure 1: description](../images/{paper-id}/fig1.png)
![Figure 2: description](../images/{paper-id}/fig2.png)
```

Also update `papers.json`:
- Set `"read_depth": "deep"`
- Populate `open_questions` from the deep analysis
- Update `related_paper_ids` if the deep analysis identified related work
  already in the database

---

## Phase 8: Synthesis Update (Automatic)

After saving, **read the paper-synthesis skill** at
`~/.cursor/skills/paper-synthesis/SKILL.md` and execute its
**"Incremental Update"** procedure. This:

1. Reads `papers.json` to find existing papers related to the new one
2. Updates `related_paper_ids` bidirectionally
3. Identifies new cross-category connections
4. Reports a brief synthesis update to the user

---

## Phase 9: Generate HTML (Automatic)

This phase is handled by `sync.sh` in the GitHub Pages repo. It:

1. Reads `papers.json` + all `notes/*.md` files
2. Generates `index.html` — paper list sorted by **publication date**,
   with **category filter buttons** (All / Framework / Kernel / etc.)
3. Generates per-paper `papers/{id}.html` with two sections:
   - **Summary** (top) — from notes content above `## Deep Analysis`
   - **Details** (bottom) — from `## Deep Analysis` section
4. **Images embedded as base64** — figures referenced via `### Figure N:`
   headings in notes are matched to PNG files in `images/{id}/` and
   converted to `data:image/png;base64,...` inline in the HTML
5. **Orphaned images dropped** — images without matching `### Figure N:`
   headings are NOT included (no standalone gallery)
6. **Local paths sanitized** — any `/home/`, `~/.cursor/`, `/apps/`
   paths are stripped from the HTML output

Labels: "Summary" and "Details" (NOT 粗读/精读, always use Summary/Details).

```bash
bash /apps/feiyue/upstream/zhaifeiyue.github.io/sync.sh
```

---

## Phase 10: Generate Per-Paper HTML Reader (Automatic)

Generate a cognitive-friendly, single-file HTML reader for this paper.

**Read the guide**: `~/.cursor/skills/paper-reader/html-reader-guide.md`

This guide defines:
- A **Role**: academic reading designer + senior deep learning professor
- **11 sections** in cognitive ladder order (导读面板 → 背景 → 架构图 →
  第一性原理 → 双栏批注 → 指标解读 → 实验解读 → 改进方向 → 答题卡)
- **Design specifications**: colors, fonts (Google Fonts), interactions
- **SVG architecture diagram**: hand-drawn style, re-drawn from paper descriptions

**Procedure**:
1. Read `html-reader-guide.md` using the `Read` tool
2. **Select and download the Hero Figure** — identify the paper's single
   most representative figure (usually Figure 1 / overview diagram).
   Download it and embed as base64 data URI at the top of the HTML.
   For arXiv HTML papers:
   ```bash
   curl -sL -o ~/.cursor/paper-db/images/{paper-id}/hero.png \
     "https://arxiv.org/html/{arxiv-id}v{N}/extracted/{hash}/figure.png"
   ```
   Convert to base64 and embed inline in the HTML guide panel section.
3. Using ALL content gathered in Phases 1-7 (paper text, deep analysis,
   images, notes), generate the complete HTML file
4. Save to `~/.cursor/paper-db/readers/{paper-id}-reader.html`

```bash
mkdir -p ~/.cursor/paper-db/readers
```

The HTML reader is a standalone file — all CSS/JS inlined, fonts from
Google Fonts CDN only, no other external dependencies.

---

## Phase 11: Publish to GitHub Pages (Automatic)

Phase 9 already runs `sync.sh`. Now commit and push:

```bash
cd /apps/feiyue/upstream/zhaifeiyue.github.io
git add -A
git commit -m "add: {paper title}"
git push
```

This updates https://zhaifeiyue.github.io/ with the latest paper readings.
The site uses Summary/Details labels, base64 images, no local paths.

---

## Final Output to User

After all 11 phases complete, present a consolidated report:

```markdown
## 📄 {Title}

**Category**: {category} | **Tags**: {tags}

### Core Contribution
{one sentence}

### Summary
{2-3 paragraphs}

### Key Findings
- ...

### Deep Analysis Highlights
{Top 3-5 most important insights from the deep read}

### Infrastructure Impact
| Layer | Impact |
|-------|--------|
| Algorithm | ... |
| Kernel | ... |
| Framework | ... |
| LLM | ... |
| Agent | ... |

### Connections to Knowledge Base
- Related to: {existing papers}
- New cross-category insight: ...
- Open questions: ...

> Saved to paper-db: papers.json + notes/{id}.md
> HTML updated: overview.html + {category}.html
> Paper reader: readers/{id}-reader.html
> View: file://~/.cursor/paper-db/overview.html
> Read: file://~/.cursor/paper-db/readers/{id}-reader.html
```

---

## Searching for Papers

When the user asks to search for papers on a topic (not reading a specific one):

```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:QUERY&max_results=10&sortBy=submittedDate&sortOrder=descending" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
for i, entry in enumerate(root.findall('a:entry', ns)):
    title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
    arxiv_id = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
    published = entry.find('a:published', ns).text[:10]
    cats = ', '.join(c.get('term') for c in entry.findall('a:category', ns))
    print(f'{i+1}. [{arxiv_id}] {title}')
    print(f'   Published: {published} | Categories: {cats}')
    print()
"
```

Present the list, and if the user picks one, run the full pipeline on it.
