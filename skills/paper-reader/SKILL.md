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

**Goal**: build complete paper context before any distillation decision
is made. You are NOT allowed to start Stage 3 (the 7 sections) until
READ is done and its exit criteria pass.

**Output**: `~/.cursor/paper-db/preread/{paper-id}.md` — a long-form
file whose **top-level structure mirrors the paper's own section
numbering and titles**. You are not allowed to reorder, merge, or
skip the paper's sections. The preread is an audit trail of what the
authors actually wrote (in their order), not your synthesis.

### File structure — one block per paper §

```markdown
# {Title} — preread

> Paper: {title}
> Date read: {YYYY-MM-DD}
> Paper sections: {N}
> Category (from Stage 1): {category}

---

## §0 Abstract

### 逐段复述
- Para 1: ...
- Para 2: ...

### 章节小结
{2–4 sentences: what is this section's role in the paper's arc?}

### 核心约束
{What design-space constraints does this section introduce, explicitly or
implicitly? Every "X is the only way because Y" / "we cannot do Z because
W" belongs here. If the section introduces no constraint, write
"N/A — 本节 {role}, 未对设计空间施加约束".}

---

## §1 Introduction

### 逐段复述
...

### 章节小结
...

### 核心约束
...

---

## §2 Related Work / Background
... (same 3-block pattern)

---

## §3 Method
... (same)

... (continue for every numbered section the paper has, including
appendices and supplementary material if present)
```

### The three required sub-blocks per paper § (details)

**1. 逐段复述 (paragraph-by-paragraph restatement)**

- One bullet per paragraph (or per logical sub-paragraph for long ones)
- 2–4 sentences each, in plain language, in the paper's order
- Quote verbatim any sentence that is a headline claim, a formal
  definition, or a surprising admission. Wrap quotes in `> " ... "`
- Include every equation that appears in this §, with:
  - The LaTeX reproduction
  - Each variable's meaning (notation table)
  - The physical / intuitive interpretation
  - A tag: `[load-bearing]` (reused in later §§ or in case-study
    derivation) or `[decorative]` (scene-setting, not reused). Unsure →
    tag `[load-bearing]` by default and revisit in Stage 3.
- Include every table in this §, row-by-row:
  - Column semantics
  - Best cell per column + runner-up gap
  - Any row that breaks the trend
- Include every figure in this §, fully described:
  - Caption verbatim
  - Components / axes / legend
  - What it demonstrates
  - Which other figs / tables / equations it is coordinated with
- Include every explicit claim that this § makes, paired with the
  evidence the authors cite for it (other §§, tables, figures, or
  external references)

**2. 章节小结 (section summary)**

2–4 sentences answering:
- What role does this § play in the paper's overall narrative?
  (motivation / background / method-core / model / experiments /
  ablation / discussion / limitation / appendix-detail)
- What does the reader now know that they didn't before this §?
- What must the reader carry forward to understand subsequent §§?

The summary is **content-only**. Do not yet judge whether the § is
"important" or "skippable" — that's Stage 3's job.

**3. 核心约束 (design-space constraints introduced here) — MANDATORY**

This is the block that keeps READ from degenerating into a neutral
paraphrase. Every § either introduces, uses, or assumes a constraint
— surface it explicitly. Examples of constraints to capture:

- **Hardware constraints**: "MHA's KV cache scales O(N · H · D), so at
  N=128K, H=128, D=128, KV = 1 GB/request → incompatible with 80 GB
  GPU at batch > 75" (§2 Background)
- **Algorithmic constraints**: "Per-channel quantization of K fails
  because QKᵀ produces N×N output with no d-dimension to apply the
  per-channel scale" (§3 Method, rejection-of-alternatives)
- **Data / training constraints**: "Joint multimodal training requires
  ≥10% vision tokens in the mix, else vision capability regresses
  within 50B tokens" (§4 Training)
- **Experimental constraints**: "H100 FP8 tensor core makes INT8
  comparison weaker than FP16 baseline — all numbers reported assume
  FP16 accumulator" (§5 Experiments)

Format each constraint as: `constraint statement → why (evidence from
this § or cross-reference) → implication for design space`.

If a § genuinely introduces no constraint (e.g. §1 Introduction is often
pure motivation), write: `N/A — 本节 {motivation / narrative / survey /
...}, 未对设计空间施加约束`. Do NOT leave the block empty.

### Rules during READ (what NOT to do)

- **Do NOT** write TL;DR, Q1/Q2/Q3, Core Contribution, or any of the 9
  sections yet. If you feel the urge, tell yourself "that decision
  needs Stage 3 context, not enough information yet."
- **Do NOT** pre-select "important" figures / tables / equations. All
  of them go into the preread; the 3–5 picks happen in Stage 3.
- **Do NOT** reorder the paper's sections. If the paper's §4 is
  Experiments and §5 is Method, your preread has §4 = Experiments and
  §5 = Method. Respect the authors' structure; your job is to annotate
  it, not refactor it.
- **Do NOT** merge two adjacent sections because "they're on the same
  topic". Same §.5 subsections can be grouped if the paper explicitly
  nested them, but top-level § are always separate blocks.
- **Do NOT** make hybrid-disambiguation, category, or code cross-reference
  decisions yet. The classification was already done in Stage 1 from the
  abstract/title; if READ reveals the classification was wrong, flag it
  at the end of preread and re-run Stage 1.2 before entering Stage 3.
- **Do NOT** skim appendices, limitations, related work, or ablations.
  These are where load-bearing details hide. If a section truly adds
  nothing, write one sentence saying so in 章节小结 — never silently
  skip.

### Category-specific READ hints

After the section-by-section walk is complete, open the matching
`deep-<category>.md` and scan its § headings to check whether there's a
detail you should have captured but missed. The deep-*.md guides are
**READ-time checklists** as much as WRITE-time templates. Example:
deep-llm.md will remind you to capture every `config.json` field, every
per-module parameter count, every KV cache size statement. Go back and
add missing details to the relevant paper § block.

### Exit criteria (must pass before Stage 3)

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
- [ ] Every paper § has non-empty 核心约束 block (either a real
      constraint or an explicit `N/A — {reason}` line)

If any checkbox fails → re-read the relevant § and extend the preread.
Partial understanding at this stage produces the classic PrfaaS failure
(author证明 reduced to "Eq.X 给出..." hand-wave because reader didn't
see the model → case-study mapping).

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
