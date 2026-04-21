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
ABSORB  →  READ  →  WRITE  →  PUBLISH
fetch      full      9-section    sync + commit
classify   scratch   notes        + completeness
extract    notes     (from        gate
figures    (no       scratch,
           cuts)     pick &
                     distill)
```

**The READ-before-WRITE discipline** is the single most important change
in v0.2.1: LLMs cannot decide "what's important" while reading; that
decision needs to be made from a position of full paper context.
Forcing structured output during reading produces selective skimming,
missed cross-references, and downgraded details. The scratch notes from
Stage 2 are the raw material; Stage 3 picks and distills from them.

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

## Stage 2 — READ (exhaustive read-through, NO structured output yet)

**Goal**: build complete paper context before any distillation decision
is made. You are NOT allowed to start Stage 3 (the 9 sections) until
READ is done and its exit criteria pass.

**Output**: `~/.cursor/paper-db/notes-scratch/{paper-id}.md` — a
free-form long-form dump. No section template, no length cap, no
aesthetics. Just thorough. The scratch is an audit trail, not a
deliverable to the user — but it must exist on disk.

### What the scratch must contain

1. **Section-by-section walk** (in the author's order). For each paper
   §N, write 3–8 sentences summarizing what it says in plain language,
   plus any quote you want to preserve verbatim.
2. **Every equation, reproduced**. Copy it, state what each variable
   means, state the physical interpretation, state what follows from
   it. Tag each equation as either **load-bearing** (used in downstream
   reasoning or case-study derivation) or **decorative** (scene-setting
   only). Unsure → tag as load-bearing and revisit.
3. **Every table, row by row**. For each: column semantics, the best
   cell per column, runner-up gap, which row breaks the trend, is the
   baseline fairly configured.
4. **Every figure, fully described**. Caption, components, what it
   shows, how it supports the surrounding text, which other figures or
   tables it is coordinated with (a result in Fig.7 often re-cites
   Table 3 — note both sides).
5. **Every explicit claim with its evidence**. "Authors claim X (§N
   para Y); evidence cited: Fig.Z + Table K + the number A from §M."
6. **Open surprises**. Anything you didn't expect, anything that
   contradicts your priors, anything asserted without obvious proof.
   These become §6 论证链 attack candidates later.

### Rules during READ (what NOT to do)

- **Do NOT** write TL;DR, Q1/Q2/Q3, Core Contribution, or any of the 9
  sections yet. If you feel the urge, tell yourself "that decision
  needs Stage 3 context, not enough information yet."
- **Do NOT** pre-select "important" figures / tables / equations. All
  of them go into the scratch; the 3–5 picks happen in Stage 3.
- **Do NOT** make hybrid-disambiguation, category, or code cross-reference
  decisions yet. The classification was already done in Stage 1 from the
  abstract/title; if READ reveals the classification was wrong, flag it
  and re-run Stage 1.2 before entering Stage 3.
- **Do NOT** skim sections you judge "less relevant". Appendices,
  limitations, related work, and ablations are where the load-bearing
  details hide. If a section truly adds nothing, write one sentence
  saying so; don't silently skip.

### Category-specific READ hints

After the section-by-section walk is complete, open the matching
`deep-<category>.md` and scan its § headings to check whether there's a
detail you should have captured but missed. The deep-*.md guides are
**READ-time checklists** as much as WRITE-time templates. Example:
deep-llm.md will remind you to capture every `config.json` field, every
per-module parameter count, every KV cache size statement.

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

If any checkbox fails → re-read the relevant part. Partial understanding
at this stage produces the classic PrfaaS failure (author证明 reduced
to "Eq.X 给出..." hand-wave because reader didn't see the model →
case-study mapping).

---

## Stage 3 — WRITE (distill scratch into the 9 sections)

Now read your `notes-scratch/{id}.md` back and produce the real
`~/.cursor/paper-db/notes/{paper-id}.md` with the following **9
mandatory sections**. Category-specific extensions go in
`deep-<category>.md`.

**Discipline**: Stage 3 is where you DELETE, PROMOTE, and STRUCTURE —
in that order. From the scratch:
- **Delete** content that's redundant, tangential, or pure background
- **Promote** the 3–5 most important figures into §5, the load-bearing
  equations into §4, the ≥3-step chain into §6, the binding-worthy
  claims into §8 — the scratch had them all; here you pick
- **Structure** into the 9 sections below

If a 9-section field would be empty because the scratch doesn't have
the raw material → the READ was incomplete, not the paper. Go back to
Stage 2 and capture the missing detail, rather than writing `[论文未
披露]` prematurely. `[论文未披露]` is reserved for things the paper
genuinely does not disclose, verified via Stage 2 coverage.

### The 9 sections

| # | Section | Contents | Hard minimum |
|---|---|---|---|
| 1 | **TL;DR** | 3 sentences, include core numbers / algorithm idea / project scope | ≤ 300 chars |
| 2 | **Q1 / Q2 / Q3** | Q1 痛点, Q2 方法, Q3 结果 + baseline numbers | each ≤ 3 sentences, must progress |
| 3 | **架构 / 方法图** | Mermaid first; drawio only for multi-page ladder or chip floorplan; SVG only for hero figure | ≥ 1 diagram |
| 4 | **作者证明** | Notation table + each equation's physical meaning + monotonicity/convexity + first-order mapping from model → case-study numbers. If paper has no formal model, mark `**无形式化作者证明 — 仅实证**`. | 6 checks, see below |
| 5 | **实验与数据** | 3–5 key figures/tables embedded **next to the text that discusses them** (not in a separate gallery section) | bold best per column in tables |
| 6 | **论证链 + 可攻击面** | ≥3-step table (step / premise + citation / conclusion / evidence) with load-bearing markers; every limitation paired with 1 adversarial rebuttal attempt | ≥3 steps, not dichotomy |
| 7 | **实现 cross-reference** | Any public code at all — paper's own repo, runtime it extends, baseline it compares against — MUST be walked. File:line citations required. | ≥ 1 code citation OR explicit `[实现未公开]` |
| 8 | **生态位** | 1 paragraph: paradigm-shift positioning, adoption by downstream systems, forced bindings (what the method locks you into) | 1 paragraph |
| 9 | **开放问题** | 3–5 items, ≥1 hybrid / adaptive / trade-off direction the paper didn't explore | ≥ 3 |

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

### §6 construction rules

- **Argument chain** must be a numbered table, not prose:
  `| step | premise (citing §/Table/Fig) | conclusion | evidence |`
- Mark each step as **load-bearing** or **decorative**
- If you can't express the paper's claim in ≥3 steps, you have read it
  shallowly — go back.
- For every critique / binding in the row "可攻击面": state which step's
  premise it attacks. Unlabeled attacks land on strawmen.
- **Ecosystem reality check** before committing any "若 X 失效" counterfactual:
  (a) Term gate — is the paper's "X" the same as casual usage? (e.g.
  "dense" in 2024+ papers means GQA/MLA-only, not MHA). (b) Plausibility
  gate — can you name ≥2 production deployments in the last 12 months
  where X is dominant or has a revival path? If not, drop the counterfactual.
- Every binding gets an **adversarial rebuttal** attempt: is there a
  configuration where this "limitation" actually makes the method *more*
  suitable? Even if the rebuttal concludes "no, binding holds universally",
  the attempt must be recorded.

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

## Stage 4 — PUBLISH

1. **Incremental synthesis** — read `~/.cursor/skills/paper-synthesis/SKILL.md`
   and run its "Incremental Update" procedure. Updates `related_paper_ids`
   bidirectionally in papers.json.

2. **Render HTML** —
   ```bash
   bash /apps/feiyue/upstream/zhaifeiyue.github.io/sync.sh
   ```
   sync.sh reads `notes/{id}.md` + `papers.json` and produces a **single
   HTML per paper** at `papers/{id}.html` that serves as both the full
   notes and the pedagogical reader (hero + TL;DR + Q1Q2Q3 at top, full
   9 sections below, images inlined as base64, drawio embedded via
   lazy-load pattern). The separate `readers/{id}-reader.html` is
   deprecated in v0.2 — sync.sh auto-merges both roles.

3. **Nav unifier** — run the post-sync nav script (see
   `html-reader-guide.md` for the snippet) to ensure every page has the
   Papers / Knowledge / Graph / Tools / GitHub links.

4. **Commit + push** —
   ```bash
   cd /apps/feiyue/upstream/zhaifeiyue.github.io
   git add assets/{id}_*.drawio papers/{id}.html index.html knowledge-graph.html
   # DO NOT use `git add -A` — stages unrelated work
   git diff --cached --name-only  # verify
   git commit -m "add: {title}"
   git push
   ```

5. **Final completeness gate** —
   ```bash
   ls ~/.cursor/paper-db/notes-scratch/{id}.md  # scratch must exist
   python3 ~/.cursor/paper-db/tools/check_paper_completeness.py {id} --strict
   ```
   Scratch presence is a soft signal — not yet enforced by the checker,
   but its absence means Stage 2 was skipped, which is a process
   failure even if Stage 3 output looks OK.
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
{3–5 sharpest insights from §4 作者证明 / §6 论证链 / §8 生态位}

### Infrastructure Impact
| Layer | Impact |
|---|---|
| algorithm | ... |
| kernel | ... |
| framework | ... |
| llm | ... |
| agent | ... |
| code | ... |

### Connections
- Related: {ids}
- Open: {questions}

> Notes: `~/.cursor/paper-db/notes/{id}.md`
> HTML: https://zhaifeiyue.github.io/papers/{id}.html
```

---

## History & rationale

All historical postmortems, rule origins, real-incident narratives, and
failure-mode taxonomy live in `~/.cursor/paper-db/incidents.md`. This
skill is intentionally terse; `incidents.md` is the long-form memory.
If you feel the urge to inline a "why this rule exists" paragraph here,
resist — add it to `incidents.md` and leave this file as pure action.
