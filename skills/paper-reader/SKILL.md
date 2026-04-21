---
name: paper-reader
description: >-
  Unified reader for all AI infrastructure content. Handles arXiv papers, blogs,
  whitepapers (PDF), GitHub repos, PRs, and issues. Automatically classifies
  and runs the full pipeline: fetch, summarize, classify, deep read, save,
  generate HTML, and publish to GitHub Pages. Triggers on: "解读", "读paper",
  "帮我读", "read paper", "精读", "粗读", "解读代码", "解读repo", "read code",
  "code reading", or similar phrases. When input is a GitHub repo/PR/issue URL,
  category is automatically "code" and deep-code.md guide is used.
---

# AI Infra Paper Reader (slim v0.2.1)

Four stages. Nine notes sections. One HTML. No silent skips.

```
ABSORB  →  READ  →  WRITE  →  CONNECT  →  PUBLISH
fetch      preread   7-section    cross-paper    sync + commit
classify   per §     notes        synthesis      + completeness
extract    逐段+小结+  (paper-     (related +      gate
figures    核心约束   internal    delta + 可攻击+
                     only)       生态位, via
                                 paper-synthesis)
```

**Skill separation of concerns**:

- **`paper-reader`** (this skill) owns Stages 1, 2, 3, 5 — everything
  that can be produced by reading ONE paper in isolation. Notes from
  Stage 3 are pure single-paper content.
- **`paper-synthesis`** (sibling skill) owns Stage 4 CONNECT — the
  high-level thinking and cross-paper analysis that REQUIRES the notes
  library to produce. Output: `knowledge/synthesis-{id}.md`.

The rules that live in `paper-synthesis` (moved out of this skill in
v0.3): adversarial rebuttal, ecosystem reality check, paradigm-shift
positioning, design-binding critique (with cross-paper grounding),
trade-off axis + unexplored hybrid, 根本性 vs 缓解性 审视, this paper
vs other papers' delta. These CANNOT be written well from just one
paper — they need the library.

**The READ-before-WRITE discipline** is the single most important
architectural rule: LLMs cannot decide "what's important" while reading;
that decision needs full paper context. Forcing structured output during
reading produces selective skimming, missed cross-references, downgraded
details.

The **preread** (Stage 2 output) is structured per-paper-§, in the
authors' original order. Each § block contains three mandatory
sub-blocks: 逐段复述 (paragraph restatement) + 章节小结 (section
summary) + 核心约束 (design-space constraints introduced in this §).
This prevents the agent from silently re-ordering, selectively reading,
or skipping "less relevant" sections. The preread is the raw material;
Stage 3 WRITE picks and distills from it; Stage 4 CONNECT then
cross-references with the note library.

**Before every paper read**: skim `~/.cursor/paper-db/incidents.md` for past
failure modes. Every rationale / postmortem / historical "why" lives there.
This skill only says **what to do**.

---

## Stage 1 — ABSORB

### 1.1 Fetch by source type

| Input | Action |
|---|---|
| arXiv ID (e.g. `2402.03300`) or arxiv.org URL | `curl` atom API for metadata → `WebFetch` on `https://arxiv.org/html/{id}` first → PDF fallback |
| GitHub repo URL | `git clone --depth 1` + `gh api` for stars/issues/PRs. Category = `code`. |
| GitHub PR URL | `gh pr view N --json ...` + `gh pr diff N`. Category = `code`. |
| GitHub issue URL | `gh issue view N --json ...`. Category = `code`. |
| Blog / article URL | `WebFetch` |
| Local PDF | `Read` tool |

Capture: title, authors, date, source URL, abstract. For PDFs, Read enough
pages to cover **all figures and tables** — not just abstract.

If source unreachable, explicitly mark Tier-3 fallback. Never proceed with
`[内容未抓取]`.

### 1.2 Classify

Pick exactly one **primary category** from this table:

| Category | Scope | Examples |
|---|---|---|
| `algorithm` | Training recipes, optimizers, RL, data strategy | GRPO, DPO, RLHF, MoE routing |
| `kernel` | GPU kernels, numerical formats, hardware mapping | FlashAttention, Triton, CUTLASS, CK, FP4/FP8 GEMM |
| `framework` | Inference / training systems, scheduling, serving | vLLM, SGLang, TRT-LLM, DeepSpeed, Megatron |
| `llm` | Model architecture, quantization | Llama, DeepSeek, MLA, GQA, GPTQ, AWQ |
| `agent` | Agent runtime, tool use, planning | ReAct, MCP, code agents |
| `cluster` | Networking, topology, collective, storage | Fat-tree, RDMA, NCCL, 3FS |
| `hardware` | Chip architecture, whitepapers, ISA | Blackwell/Hopper, CDNA4, HBM |
| `code` | Repos, PRs, issues, design docs | vLLM repo reading, RFC issues |

Plus 2–5 **secondary tags**. Content that is ≥30% of the paper in another
category MUST surface as a secondary tag (e.g. Fleet = framework primary +
`kernel` secondary because half of §4-5 is ISA-level).

#### Hybrid disambiguation (when two categories feel plausible)

1. **List contributions** with 1-word category tag each. If one category
   covers ≥70%, pick it.
2. **Thesis test** — read TITLE + abstract's first line + "We present X"
   sentence. Match the thesis pattern:
   - "new abstraction / runtime / system" → `framework` / `llm` / `agent` / `cluster` / `code`
   - "optimization for operator X" → `kernel`
   - "new chip / ISA / instruction" → `hardware`
   - "new training recipe / loss / RL method" → `algorithm`
3. **Fit test** — which `deep-<cat>.md` guide would have ≤10% `[N/A]`
   sections for this paper? Pick that one.
4. **Record**: if two candidates were close, add a 2-3 sentence hybrid
   justification at the top of the notes, under `> Note on classification:`.

### 1.3 Save base entry to `~/.cursor/paper-db/papers.json`

17 required fields: `id, title, authors, date, source, url, category,
secondary_tags, core_contribution, summary, key_findings, limitations,
infra_impact (6 sub-keys), related_paper_ids, open_questions, read_depth,
date_read`.

All 6 `infra_impact` sub-keys (algorithm / kernel / framework / llm / agent
/ code) must be populated, even if value is `"N/A — reason"`.

### 1.4 Extract figures

Run `~/.cursor/paper-db/tools/extract_figures.py {pdf} {out_dir}` (for
PDFs) or download from `https://arxiv.org/html/{id}v{N}/x{N}.png` (for
arXiv with HTML).

**Closed-loop quality gate**: after extraction,
1. Tool's `EXTRACTION SUMMARY` must show N extracted figures matching the
   N captions found. If `N=0` but captions exist, the tool failed silently
   — DEBUG the tool, never fall back to manual `get_pixmap` crops.
2. `Read` every output PNG. Each must show: (a) caption included, (b) full
   figure content, (c) no body-text leakage, (d) no sub-figure cut off.
3. Each PNG > 5 KB.

---

## Stage 2 — READ (section-by-section, anchored to paper's own structure)

### 核心约束 — Stage 2 整体纪律（忠于原文，不做任何取舍）

> This is THE constraint of READ. Every other rule in this section is
> a refinement of it. Read it out loud before starting every new paper.

- **不允许做任何取舍决策**。"这一段不重要 / 这张图不核心 / 这条方程
  decorative" — 全部是 Stage 3 的判断，不是 Stage 2 的。
- **输出**：`~/.cursor/paper-db/preread/{paper-id}.md` — free-form, 无
  结构，无长度限制。不是最终交付物，是 Stage 3 的原材料 + 审计痕迹。
- **必须覆盖** (7 items — the paper's complete surface):
  1. 逐节走一遍，按论文原序 —— **包括 appendix / supplementary /
     supplementary tables / supplementary figures**。重点往往在附录。
  2. 每条方程：**LaTeX 原样保留** + 变量含义（notation table 原样复现）
     + 物理解读 + `[load-bearing]` / `[decorative]` 标签
  3. 每张表：**原表格 row-by-row 复现**（不能只写 "showed improvement"）
  4. 每张图：**caption 原文引用** + 轴 / 图例 / sub-figure labels / 所有
     组件 + 与其他图表的 coordination
  5. 每条显式 claim + 其证据（引用对应 §/Table/Fig 编号）
  6. **参考文献（References）**：见下方 References 子规则
  7. 开放性惊讶点 —— 不预判、不 rebut，只 flag

- **保留原件 (Preserve-Verbatim rule)** —— 单独拎出来强调：
  - 方程用 LaTeX 原样写（`$...$` / `$$...$$`），**不要用自然语言替代**
    （"公式 3 说 X 等于 Y" 是 Stage 3 的简写，不是 Stage 2）
  - 表格 row-by-row 用 markdown table 复现，**不要简写**（"论文给了
    各 seq len 的结果" 不合格）
  - 图表 caption 原文照抄并打 `> " ... "` quote marks
  - **逻辑链** (the author's argument chain) 必须在 preread 中可追溯：
    每条 claim 写出 `premise → conclusion → evidence (§/Table/Fig)` 的
    原始形态，不要跳步
  - **为什么这条规则存在**：Stage 3 WRITE 做抽象，靠的是 preread 里保留
    的原始细节。抽象是单向有损操作；如果 preread 已经抽象了，Stage 3
    会级联放大误差，或者到时候回头重读 paper（浪费时间）

- **禁止** (7 items):
  1. 写 TL;DR / Q1/Q2/Q3 / 任何 Stage-3 的 7 节字段
  2. 预选"重要图"—— 所有图都记，3–5 张的挑选在 Stage 3
  3. 跳读 **appendix / supplementary / limitations / related work /
     references**（references 尤其容易被忽略）
  4. 合并相邻 §（"反正同主题"）
  5. 重排 § 顺序（论文 §4 是实验就先写 §4，哪怕你觉得应该先 §5）
  6. 分类 / 取舍 / 综合（所有 taxonomic 操作留给 Stage 3）
  7. 用自然语言替代原件（方程必须 LaTeX, 表格必须 row-by-row, caption
     必须 verbatim）

**Goal**: build complete paper context before any distillation decision
is made. You are NOT allowed to start Stage 3 (the 7 sections) until
READ is done and its exit criteria pass.

The preread is an audit trail of what the authors actually wrote (in
their order), with enough verbatim preservation that Stage 3 can do
the abstracting WITHOUT going back to the paper. Your job is to
**restate + describe + quote**, not to **categorize + judge + summarize
away**. If you feel the urge to label something as "this is a constraint
the paper introduces" or "this is the key insight" — stop. That's
Stage 3 thinking.

### File structure — one block per paper §, plus References + Surprises

```markdown
# {Title} — preread

> Paper: {title}
> Date read: {YYYY-MM-DD}
> Paper sections: {N main body + M appendix/supplementary}
> Total figures / tables / equations: {F figs, T tables, E eqs}
> References count: {R entries}
> Category (from Stage 1): {category}

---

## §0 Abstract

### 逐段复述
- Para 1: ...
- Para 2: ...

### 章节小结
{2–4 sentences: what is this section's role in the paper's arc?}

---

## §1 Introduction

### 逐段复述
...

### 章节小结
...

---

## §2 Related Work / Background
... (same 2-block pattern)

... (continue for every numbered section the paper has)

---

## Appendix A / B / ... (if present)

### A.1 / A.2 / ... (for each appendix subsection)

#### 逐段复述
...

#### 章节小结
...

(Appendix sections get the SAME treatment as main-body sections. No
shortcuts — appendix is often where proofs, extended tables, hyperparam
grids, ablations, and dataset details live. For many papers these
contain the most important details.)

---

## Supplementary material (if separate document / files)

If the paper has a supplementary PDF, supplementary tables (Table S1,
S2, ...), or supplementary figures (Fig S1, S2, ...) distributed as
separate files — fetch and READ them. Give each supplementary section
its own § block under this heading with the same two sub-blocks.

---

## References

> 参考文献不是点缀。它是 paper 的 lineage map: 论文把自己 position
> 在哪些前人工作上, 比哪些 baseline, 引用哪些技术, 回避哪些。这些决定
> 了 Stage 4 synthesis 能否准确做跨篇关联。

For each non-trivial reference (see rules below), record:

- `[N] Author et al., year. Title. Venue.` (short form)
- **Cited where**: §N / Table N / Fig N that cites it (can be multiple)
- **Role**: one of `predecessor` (this paper builds on it) / `baseline`
  (this paper compares against it) / `related-technique` (referenced
  as an adjacent approach) / `tool` (used in the implementation, e.g.
  the framework / library) / `motivation` (establishes the problem
  this paper solves) / `hardware` (cited chip / system)

Rules:
- Capture **every citation that appears in §1 (motivation) and §2–3
  (background / method)** — these are the ones the paper's argument
  leans on.
- For §6 Related Work: capture **all** cited papers (the whole taxonomy
  the paper places itself into).
- For §4 Experiments: capture all **baseline models / systems** and
  their references.
- For §5 Discussion: capture all cited hardware / systems / techniques.
- Bibliography-only references (appear once in the reference list but
  not cited in body) can be skipped with one-line noting their
  existence.
- If paper has ≤ 15 references total, capture all of them.

---

## 全篇开放性惊讶点

{bulleted list of things that surprised you, contradicted your priors,
or were asserted without obvious proof — one bullet each, cross-refs
to paper §. Aggregated here because surprises are usually cross-section
observations, not localizable to one §. Stage 3 uses these as candidate
attack / open-question surfaces.}

---

## Exit criteria self-check

... (6 checkbox list + 1 preservation check, see below)
```

### The two required sub-blocks per paper § (details)

**1. 逐段复述 (paragraph-by-paragraph restatement)**

This is where every item from the "必须覆盖" list lands. One § block,
all restatement content inline in paper order.

**Text content**:
- One bullet per paragraph (or per logical sub-paragraph for long ones),
  2–4 sentences each, in plain language, in the paper's order
- **Quote verbatim** any sentence that is a headline claim, a formal
  definition, a surprising admission, or an explicit motivation /
  constraint statement. Wrap quotes in `> " ... "` — these anchor
  Stage 3's §6 论证链 steps to the paper's own words.
- **Every explicit claim** paired with the evidence the authors cite
  (other §§, tables, figures, external references). Format:
  `paper claims X (§N para Y) — evidence: Fig.Z, Table K, cite [M]`

**Every equation** (LaTeX preserved, not paraphrased):
- Reproduce verbatim with `$$...$$` display math. Do NOT rewrite as
  prose ("Equation 3 says throughput equals the min of two terms" is
  Stage 3 shorthand, not Stage 2).
- If paper has a notation table (Table of symbols), reproduce it as-is
  in markdown table form under the relevant § (usually §Method /
  §Model). Do not defer to "see Table N" — reproduce.
- For each equation, attach three annotations:
  - **Variables**: what each symbol means (copy from notation table if
    one exists)
  - **Physical interpretation**: what the equation "does" intuitively;
    why `min` and not `sum`, why divide by $p$, why this exact form
  - **Tag**: `[load-bearing]` (reused downstream in other equations or
    in case-study derivation) vs `[decorative]` (scene-setting, not
    referenced after). Unsure → `[load-bearing]` by default; revisit
    in Stage 3.

**Every table** (row-by-row preservation):
- Reproduce the markdown table with **all rows and all columns**. "Paper
  shows a comparison table" is insufficient; "Table 3 has 6 models ×
  4 seq lengths" is insufficient. Rewrite the actual cells.
- Below the table, note: column semantics, best cell per column, any
  row that breaks the expected trend, cells the paper explicitly
  highlights.

**Every figure** (caption + all components):
- Caption: `> " ... "` verbatim quote.
- Components: axes (what they measure, units, scale log vs linear),
  legend entries, line/bar colors and what they represent, annotations,
  sub-figure labels (a)(b)(c) and what each sub-figure shows.
- Cross-reference: which other figs / tables / equations are
  coordinated with this one (e.g. "Fig.5(b) plots the RHS curves of
  Eq.7 at $N_p = 3, N_d = 5$").
- Downloaded image path (from Stage 1 Phase 1.4 extraction):
  `../images/{paper-id}/figN-{name}.png`. If image couldn't be
  extracted, note explicitly.

**Logic chain preservation**:
- For each § 章节小结 that summarizes "what the reader carries forward",
  trace the **linkage backward to previous § and forward to next §**.
  Example: "§2.1 established the bandwidth wall. §2.2 shows hybrid
  attention reduces this wall by 4–13×, which §2.3 uses to argue
  cross-DC PD becomes feasible, which §3 uses to design PrfaaS."
- This preserves the paper's argument chain in its native form, so
  Stage 3's §6 论证链 can point at the exact steps the paper intended.

Do NOT categorize or judge content. If the paper says "we cannot do X
because Y" — reproduce the sentence, note the evidence. Do not create
a separate "constraint" bucket; that is Stage 3's taxonomy task.

**2. 章节小结 (section summary)**

2–4 sentences answering:
- What role does this § play in the paper's overall narrative?
  (motivation / background / method-core / model / experiments /
  ablation / discussion / limitation / appendix-detail)
- What does the reader now know that they didn't before this §?
- What must the reader carry forward to understand subsequent §§?

Content-only. Do NOT yet judge whether the § is "important" or
"skippable" — that's Stage 3.

### Final top-level section: 全篇开放性惊讶点

After every paper § block is written, add one final top-level section
collecting all "surprises" across the paper. Each bullet:
- 1 sentence describing the surprise
- Cross-ref to paper §(paragraph) where it appears
- (optional) 1 sentence on why it's surprising

These are the raw material for Stage 4 CONNECT's attack surface and
open-question analysis. Do NOT resolve or rebut them in Stage 2; just
flag them.

### Category-specific READ hints

After the section-by-section walk is complete, open the matching
`deep-<category>.md` and scan its § headings to check whether there's a
detail you should have captured but missed. The deep-*.md guides are
**READ-time checklists** as much as WRITE-time templates. Example:
deep-llm.md will remind you to capture every `config.json` field, every
per-module parameter count, every KV cache size statement. Go back and
add missing details to the relevant paper § block.

### Exit criteria

**Part A — 6 from-memory checkboxes** (test your understanding):

Without re-opening the paper, you should be able to:

- [ ] Recall the 3–5 headline numbers (% improvement, throughput, token
      counts, param counts) from memory
- [ ] Sketch the paper's architecture / method diagram from memory
- [ ] Recite the 3 biggest claims in one sentence each
- [ ] Name which § contains the paper's formal model / theorem /
      proposition / cost model (or confidently state "no formal model")
- [ ] Point to which experiment configuration is the "case study" the
      model's prediction should match
- [ ] Name ≥2 rows in the main results table where the paper **loses**
      to a baseline (every paper has these; if you can't find any,
      you missed them)

**Part B — preservation / completeness checks** (grep your preread):

- [ ] **Every paper § in the main body has a block** — including §1, §2,
      §3, ..., conclusion. No skips.
- [ ] **Every appendix / supplementary subsection has a block** (if
      paper has any). Empty appendix OK if paper truly has none.
- [ ] **LaTeX coverage**: count `$$` display-math blocks in preread ≥
      count of numbered equations in the paper (Eq.1, Eq.2, ...).
- [ ] **Table coverage**: count markdown tables (`|...|`) in preread ≥
      count of Table N in the paper.
- [ ] **Figure coverage**: every `Figure N` caption in the paper has a
      corresponding `> "..."` verbatim caption quote in preread.
- [ ] **References block populated**: top references from §1/§2/§3/§6
      captured with role tag (not just [N] AuthorName).

If any checkbox fails → re-read the relevant § and extend the preread.
Partial understanding or premature abstraction at this stage produces
the classic PrfaaS failure (作者证明 reduced to "Eq.X 给出..." hand-wave
because the reader didn't see the model → case-study mapping).

---

## Stage 3 — WRITE (distill preread into the 7-section paper-internal notes)

Now read your `preread/{id}.md` back and produce the real
`~/.cursor/paper-db/notes/{paper-id}.md` with the following **7
mandatory sections**. **Paper-internal content only** — cross-paper
connections, high-level thinking, and attacks on the paper's argument
are Stage 4's job and go in a separate file.

Category-specific extensions go in `deep-<category>.md`.

**Discipline**: Stage 3 is where you DELETE, PROMOTE, and STRUCTURE —
in that order. From the preread:
- **Delete** content that's redundant, tangential, or pure background
- **Promote** the 3–5 most important figures into §5, the load-bearing
  equations (tagged in preread) into §4, the ≥3-step chain into §6
  — the preread had them all; here you pick
- **Aggregate 核心约束**: the preread has constraints scattered across
  every paper §. In Stage 3 you consolidate them into §6's argument
  chain (one constraint → one step)
- **Structure** into the 7 sections below

If a section field would be empty because the preread doesn't have the
raw material → the READ was incomplete, not the paper. Go back to
Stage 2 and extend the relevant paper § block, rather than writing
`[论文未披露]` prematurely. `[论文未披露]` is reserved for things the
paper genuinely does not disclose, verified via Stage 2 coverage.

**What does NOT belong in notes** (these go to Stage 4 synthesis, never
to notes):
- "This paper vs paper X" comparison
- Adversarial rebuttal of the paper's limitations
- Ecosystem reality check ("but MHA is dead in 2024+")
- Paradigm-shift positioning ("this reframes implicit HW → explicit SW")
- Design-binding critique with cross-paper grounding
- Hybrid / trade-off research directions informed by the note library
- "根本性 vs 缓解性" discussion that requires seeing trends across papers

If you notice such content emerging while writing notes, **save it to
a scratch location** (e.g. inline TODO comment `<!-- STAGE4: ... -->`)
and carry it into Stage 4 rather than inlining it into notes.

### The 7 sections (paper-internal only)

| # | Section | Contents | Hard minimum |
|---|---|---|---|
| 1 | **TL;DR** | 3 sentences, include core numbers / algorithm idea / project scope | ≤ 300 chars |
| 2 | **Q1 / Q2 / Q3** | Q1 痛点, Q2 方法, Q3 结果 + baseline numbers | each ≤ 3 sentences, must progress |
| 3 | **架构 / 方法图** | Mermaid first; drawio only for multi-page ladder or chip floorplan; SVG only for hero figure | ≥ 1 diagram |
| 4 | **作者证明** | Notation table + each equation's physical meaning + monotonicity/convexity + first-order mapping from model → case-study numbers. If paper has no formal model, mark `**无形式化作者证明 — 仅实证**`. | 6 checks, see below |
| 5 | **实验与数据** | 3–5 key figures/tables embedded **next to the text that discusses them** (not in a separate gallery section) | bold best per column in tables |
| 6 | **论证链 (paper-internal)** | ≥3-step table (step / premise + citation / conclusion / evidence) with load-bearing markers. **Pure reconstruction of the paper's own argument** — no attacks, no rebuttals, no ecosystem checks here (those are Stage 4). | ≥3 steps, not dichotomy |
| 7 | **实现 cross-reference** | Any public code at all — paper's own repo, runtime it extends, baseline it compares against — MUST be walked. File:line citations required. | ≥ 1 code citation OR explicit `[实现未公开]` |

### 作者证明 — the 6 minimum checks (section §4)

1. Notation table reproduced (not just referenced as "see Table N")
2. Each core equation explained: form → physical meaning → why this form
   (e.g. "why `min` not `sum`?", "why divide by p?")
3. Monotonicity / convexity / uniqueness argument (add one if paper
   implicit)
4. First-order mapping: plug case-study config into the model, verify
   reported numbers are derivable — NOT sweep-fit
5. Pure-model defense: "assume no case study, how would you convince a
   reader the method works?" — 1-sentence version
6. Attack surface: what simplifications did the model make (Jensen?
   stationary distribution? closed-form assumption?) and when would they
   break the predictions?

If the paper genuinely has no formal model, write `**无形式化作者证明 —
仅实证**` and state which aspects would have been model-verified if one
existed. Never fabricate a model.

### §6 construction rules (paper-internal argument chain only)

- **Argument chain** must be a numbered table, not prose:
  `| step | premise (citing §/Table/Fig) | conclusion | evidence |`
- Mark each step as **load-bearing** or **decorative**
- If you can't express the paper's claim in ≥3 steps, you have read it
  shallowly — go back to Stage 2 and re-read.
- Note where the paper itself already pre-empts common objections
  (e.g. "Table 3 pre-empts 'isn't GQA enough?' by showing 33–60 Gbps").
  This is still paper-internal (the author's own preemption) — not a
  reader's attack.

**What stays out of notes §6** (moves to Stage 4 synthesis):
- Attacks on individual steps ("step 3's premise is questionable")
- Ecosystem reality check on counterfactuals
- Adversarial rebuttals to paper's limitations
- Binding critiques that cite other papers

Notes §6 is the setup for Stage 4's attack surface analysis. Write a
clean, unambiguous reconstruction of what the paper argues — so that in
Stage 4 you can point to specific steps as attack targets.

### Category-specific extensions

After writing the 9 base sections, read the matching `deep-<category>.md`
and add **only the category-specific content** it prescribes (typically
3–5 extra points). The 9 base sections cover the universal scaffold;
deep-*.md files are now additive, not replacement templates.

| Category | Guide |
|---|---|
| algorithm | `deep-algorithm.md` |
| kernel | `deep-kernel.md` |
| framework | `deep-framework.md` |
| llm | `deep-llm.md` |
| agent | `deep-agent.md` |
| cluster | `deep-cluster.md` |
| hardware | `deep-hardware.md` |
| code | `deep-code.md` |

### Code cross-reference (§7) — by category

| Category | Walk this critical path |
|---|---|
| llm | `forward()` + `config.json` + modeling file's attention / MoE / norm |
| kernel | hot-path kernel + launch config + tile/block params |
| framework | scheduler main loop + KV / memory manager |
| algorithm | loss function + data loader + optimizer wrap |
| agent | agent main loop + tool call wrapper |
| cluster | collective implementation + transport layer |
| hardware | driver IOCTL + compiler backend pass |
| code | the repo itself IS the walk (see `deep-code.md`) |

**No backbone-reuse exemption**: paper N reusing M's architecture
(e.g. K2.6 reuses K2.5) still gets its own drawio + file:line citations.
"See X's notes" is NOT an acceptable substitute for N's own §3 diagram
or §7 code walk.

### Diagram tool choice

| Scenario | Preferred | Fallback |
|---|---|---|
| Pipeline / data flow / multimodal routing | **Mermaid flowchart** | drawio |
| Decoder block / MLA / MoE expand | **Mermaid flowchart + subgraph** | drawio |
| Request / tool-call lifecycle | **Mermaid sequenceDiagram** | — |
| State machine | **Mermaid stateDiagram-v2** | — |
| ≥3 coordinated views of same system | **drawio multi-page** | — |
| Chip floorplan / die / physical geometry | **drawio** | — |
| Hero figure (Stage 3 HTML top) | **hand-drawn SVG** | — |
| Linear 3-step text (`A → B → C`) | **markdown prose** | — |
| Table of numbers | **markdown table** | — |

**Banned everywhere**: ASCII box art (`┌─ │ ─┐`), code blocks as
architecture substitute. See `diagram-tool-choice.md` for templates.

### Content hygiene

**Swap-paper-name test**: for every paragraph you write, ask "if I
replaced this paper's name with 'X', does the paragraph still make
sense for any other paper?" If yes, it's skill-leakage — it belongs in
the skill or `incidents.md`, NOT in the paper's notes.

Before saving, scan your draft for these leak patterns: rule-tag
suffixes in section titles (e.g. `### 作者证明 (SJM)`), citations of
SKILL.md / deep-*.md, "Inherited universal rule" callouts, "本笔记
采用 X 结构" meta, `(Phase 2b)` parentheticals, tool-choice rationale
prose. Delete all matches before file write.

---

## Stage 4 — CONNECT (cross-paper synthesis via `paper-synthesis` skill)

After `notes/{id}.md` is written, hand off to the `paper-synthesis`
skill. Stage 4 has two outputs:

1. **Graph update** (fast, automatic): `papers.json` `related_paper_ids`
   is updated bidirectionally. This is the existing "Incremental Update"
   mode of `paper-synthesis`.

2. **Per-paper synthesis document** (the new v0.3 addition):
   `~/.cursor/paper-db/knowledge/synthesis-{id}.md` — a review-style
   document that places THIS paper in the context of the note library.

**Procedure**: read `~/.cursor/skills/paper-synthesis/SKILL.md` and
execute its "Per-Paper Synthesis" mode against the newly written notes.
That skill owns:

- Adversarial rebuttal rules (attack which step of §6 论证链)
- Ecosystem reality check (term gate + plausibility gate on
  counterfactuals)
- Design-binding critique grounded in cross-paper evidence
- Paradigm-shift positioning (implicit HW constraint → explicit SW
  resource; roofline-axis reframing; etc.)
- Trade-off axis identification and unexplored hybrid directions
- 根本性 vs 缓解性 审视 (informed by seeing multiple papers' trends)
- 对立观点主动搜索 (adversarial applicability check)
- This paper vs related papers' delta analysis

The synthesis skill reads the note library (not just the new paper's
notes), so its output can cite other papers by ID and build influence
chains.

### What the synthesis document contains (skill spec, details in
`paper-synthesis/SKILL.md`)

Expected structure of `knowledge/synthesis-{id}.md`:

1. **相关论文** — which papers in the DB are related, and why
2. **本篇 vs 相关论文的 delta** — what's new, incremental, contradictory
3. **可攻击面** — adversarial rebuttal against specific steps in notes §6
4. **生态位** — paradigm-shift positioning, adoption evidence, forced
   bindings (with cross-paper grounding)
5. **未探索方向** — hybrid / adaptive / trade-off directions suggested
   by patterns in the library

### When to skip Stage 4

If the note library is empty (this is the 1st paper), you may write a
minimal synthesis document stating "first entry in the library; no
cross-paper analysis possible yet; will be revisited when ≥2 papers
in same category exist". Save it anyway — future Stage 4 runs may
retroactively update it.

---

## Stage 5 — PUBLISH

1. **Render HTML** —
   ```bash
   bash /apps/feiyue/upstream/zhaifeiyue.github.io/sync.sh
   ```
   sync.sh reads `notes/{id}.md` + `papers.json` and produces a **single
   HTML per paper** at `papers/{id}.html` that serves as both the full
   notes and the pedagogical reader (hero + TL;DR + Q1Q2Q3 at top, full
   7 sections below, images inlined as base64, drawio embedded via
   lazy-load pattern). The separate `readers/{id}-reader.html` is
   deprecated in v0.2 — sync.sh auto-merges both roles.

3. **Nav unifier** — run the post-sync nav script (see
   `html-reader-guide.md` for the snippet) to ensure every page has the
   Papers / Knowledge / Graph / Tools / GitHub links.

4. **Commit + push** —
   ```bash
   cd /apps/feiyue/upstream/zhaifeiyue.github.io
   git add assets/{id}_*.drawio papers/{id}.html index.html \
           knowledge/synthesis-{id}.html knowledge/synthesis-{id}.md \
           knowledge-graph.html
   # DO NOT use `git add -A` — stages unrelated work
   git diff --cached --name-only  # verify
   git commit -m "add: {title}"
   git push
   ```

5. **Final completeness gate** —
   ```bash
   ls ~/.cursor/paper-db/preread/{id}.md     # Stage 2 output must exist
   ls ~/.cursor/paper-db/notes/{id}.md       # Stage 3 output must exist
   ls ~/.cursor/paper-db/knowledge/synthesis-{id}.md  # Stage 4 output must exist
   python3 ~/.cursor/paper-db/tools/check_paper_completeness.py {id} --strict
   ```
   Preread / notes / synthesis presence is a soft signal — not yet
   enforced by the checker, but absence of any of them means the
   corresponding stage was skipped, a process failure even if the final
   HTML looks OK. Once the checker is updated, it will enforce:
   (a) preread's top-level § count matches paper's actual § count (±1),
   (b) every preread § block has non-empty 逐段复述 / 章节小结 / 核心约束,
   (c) notes has 7 sections and none of them contain adversarial /
   ecosystem / paradigm-shift content (those belong in synthesis),
   (d) synthesis has ≥1 cited related paper OR explicit "first entry".
   If `check_paper_completeness.py` exits ≠ 0, fix every blocker before
   presenting the Final Output to the user. No exceptions.

---

## Final Output to user

```markdown
## 📄 {Title}

**Category**: {category} | **Tags**: {tags}

### TL;DR
{from §1}

### Q1 / Q2 / Q3
{from §2}

### Deep Analysis Highlights
{3–5 sharpest insights from §4 作者证明 / §6 论证链 (paper-internal)
+ synthesis §3 可攻击面 / §4 生态位 (cross-paper)}

### Infrastructure Impact
| Layer | Impact |
|---|---|
| algorithm | ... |
| kernel | ... |
| framework | ... |
| llm | ... |
| agent | ... |
| code | ... |

### Cross-Paper Connections (from Stage 4 synthesis)
- Related: {ids}
- Key delta: {this paper vs closest related paper, 1 sentence}
- Open: {hybrid / trade-off direction from synthesis §5}

> Preread: `~/.cursor/paper-db/preread/{id}.md`
> Notes (paper-internal): `~/.cursor/paper-db/notes/{id}.md`
> Synthesis (cross-paper): `~/.cursor/paper-db/knowledge/synthesis-{id}.md`
> HTML: https://zhaifeiyue.github.io/papers/{id}.html
```

---

## History & rationale

All historical postmortems, rule origins, real-incident narratives, and
failure-mode taxonomy live in `~/.cursor/paper-db/incidents.md`. This
skill is intentionally terse; `incidents.md` is the long-form memory.
If you feel the urge to inline a "why this rule exists" paragraph here,
resist — add it to `incidents.md` and leave this file as pure action.
