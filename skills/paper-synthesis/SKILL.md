---
name: paper-synthesis
description: >-
  Cross-paper synthesis: relationships, high-level thinking, and knowledge
  gaps across all read AI infra papers. Three modes: (1) Per-Paper Synthesis
  — called by paper-reader Stage 4 after each new paper, produces
  knowledge/synthesis-{id}.md with related-paper delta, 可攻击面, 生态位,
  hybrid directions. (2) Graph Update — fast papers.json bidirectional
  link update. (3) Full Synthesis — user-requested comprehensive trend
  analysis over the entire library. Owns all high-level / cross-paper
  rules: adversarial rebuttal, ecosystem reality check, paradigm-shift
  positioning, binding critique, 根本性 vs 缓解性, trade-off axis + hybrid.
---

# AI Infra Paper Synthesis (v0.3)

Owns **cross-paper** analysis. `paper-reader` produces paper-internal
notes; this skill produces everything that requires the note library.

## Data sources

```
~/.cursor/paper-db/
├── papers.json             # Master index — read this first
├── notes/{id}.md           # Per-paper internal content (from paper-reader Stage 3)
├── preread/{id}.md         # Raw per-§ paper walk (from paper-reader Stage 2)
└── knowledge/
    ├── synthesis-{id}.md   # Per-paper cross-paper synthesis (this skill, Mode 1)
    └── synthesis-report.md # Full-library synthesis (this skill, Mode 3)
```

## Operational modes

| Mode | Triggered by | Output | Cost |
|---|---|---|---|
| **1. Per-Paper Synthesis** | paper-reader Stage 4 (after each new paper) | `knowledge/synthesis-{id}.md` + graph update | medium |
| **2. Graph Update only** | no new paper, just relationships need refresh | `papers.json` only | low |
| **3. Full Synthesis** | user-requested trend / gap analysis | `knowledge/synthesis-report.md` | high |

---

# Mode 1 — Per-Paper Synthesis (primary mode, called by paper-reader Stage 4)

## Inputs

- `notes/{new-id}.md` — the newly written paper-internal notes (7 sections)
- `papers.json` — the master index
- `notes/*.md` of related papers (read on demand, guided by Step 1 below)
- `preread/{new-id}.md` — optionally, for details not promoted into notes

## Output

`~/.cursor/paper-db/knowledge/synthesis-{paper-id}.md` with 5 mandatory
sections. **This is the file that answers "how does this paper relate
to everything I've read?"** It does NOT restate paper-internal content
(that's in notes).

## Procedure

### Step 1: Discover related papers (don't guess)

Iterate through **every** paper in papers.json. Do not pick "obvious"
matches from memory — that leaves cross-category connections buried.

Relatedness signals (combine multiple):

| Signal | Weight |
|---|---|
| Same primary category | Medium |
| Shared secondary tags (≥2) | High |
| One's `infra_impact` mentions the other's category domain | High |
| Same research thread (cited, follow-up, competitor) | Critical |
| Shared open questions | Medium |
| Paradigm overlap (e.g. both reframe "HW constraint → SW resource") | Critical |

Papers scoring High/Critical on any signal → read their `notes/{id}.md`
in full. Record citations as `[paper-id]` (you'll inline these in the
synthesis document).

### Step 2: Update `papers.json` bidirectional graph

```python
import json, os
DB = os.path.expanduser("~/.cursor/paper-db/papers.json")
with open(DB) as f: db = json.load(f)
new_id = "NEW_PAPER_ID"; related_ids = ["RELATED_1", "RELATED_2"]
for paper in db["papers"]:
    if paper["id"] == new_id:
        paper["related_paper_ids"] = list(set(paper.get("related_paper_ids", []) + related_ids))
    elif paper["id"] in related_ids:
        paper.setdefault("related_paper_ids", [])
        if new_id not in paper["related_paper_ids"]:
            paper["related_paper_ids"].append(new_id)
with open(DB, "w") as f: json.dump(db, f, indent=2, ensure_ascii=False)
```

If 0 related papers found, write that explicitly in Step 3 §1 (this is
a real signal the library is under-populated in the new paper's area).

### Step 3: Write `synthesis-{id}.md` with these 5 sections

```markdown
# {Title} — Cross-Paper Synthesis

> Notes: ../notes/{id}.md
> Date synthesised: {YYYY-MM-DD}
> Library size at time of synthesis: {N papers, breakdown by category}

## 1. 相关论文 (Related papers)

Table of related papers with WHY they are related:

| ID | Title | Category | Relatedness signal | Strength |
|---|---|---|---|---|
| ... | ... | ... | shared tag X + overlapping open question Y | Critical |

If 0 related papers: "本篇是 {category} × {topic} 领域在 note 库里的
第一篇；无法做跨篇对比。未来读到 ≥1 篇同领域时应回写本文件。"

## 2. 本篇 vs 相关论文的 delta (This paper's delta)

For each of the top 3–5 related papers, state:
- What's **genuinely new** in this paper relative to [id]
- What's **incremental** (improves a dimension [id] already explored)
- What **contradicts** [id]'s claim (if any) — and who's right given
  the evidence both cite
- What **paradigm shift** (if any) this paper makes that [id] did not:
  e.g. "makes implicit X into explicit Y", "exposes HW primitive as
  SW resource", "collapses dimension Z that [id] kept separate"

One paragraph per related paper. Avoid "similar to [id]" — force yourself
to write the concrete delta.

## 3. 可攻击面 (Attack surface — grounded in library evidence)

Take notes' §6 论证链 (the paper-internal argument chain, each step
marked load-bearing / decorative). For each **load-bearing** step,
attempt:

### 3a. Adversarial rebuttal — does the library show an exception?

For each load-bearing step `step k: premise P → conclusion C`:

- Does any paper in the library show a configuration where P fails
  but C still holds (decorative premise)?
- Does any paper show a configuration where P holds but C fails
  (premise sufficient but not necessary — weak step)?
- Does any paper propose an alternative method that satisfies C
  without needing P?

Format each finding as: `step k attacked via [related-id]: their
result Y contradicts our step k's necessity because ...`

### 3b. Ecosystem reality check on counterfactuals

Every "若 X 失效" counterfactual the paper itself or your earlier
analysis considered, subject to TWO gates:

1. **Term gate (术语锚定)**: Is the paper's "X" the same as casual
   usage? Cite the paper's Table / §N defining it. Common pitfalls:
   "dense" in 2024+ papers often means GQA/MLA, not MHA; "long context"
   in 2025+ often means ≥128K; "MoE" sometimes excludes shared-expert.
2. **Plausibility gate (生态可达性)**: Name ≥2 production deployments
   or shipped models in the last 12 months where X is dominant or has
   a credible revival path. If you cannot, drop the counterfactual or
   reframe to an adjacent real risk.

"Doomer ≠ deep": phrasing like "全部论点失效 / 整个方案崩塌" is a smell.
Prefer calibrated: "在 X 出现的 1–2 年窗口内错位; 当前主流仍是 Y;
短期 bet 安全."

### 3c. 根本性 vs 缓解性 审视 (Root-cause vs mitigation)

For every "paper claims Y× / +Z%" in §5 of notes:

1. Is this a **universal** improvement (across all workloads) or
   **conditional** (depends on a regime like batch ≥ B, context ≥ C,
   sparsity p ≤ P)?
2. What's the regime formally? Express as inequality when possible:
   `m_tiles ≥ 2 ⇔ B ≥ 2·T_M`.
3. When the regime fails, which part of the speedup survives
   (dispatch compression, instruction reduction) and which goes to
   zero (L2 cooperation, in-register reuse)?

If you can conclude "in regime R the effect ≈ 0", that is the paper's
most honest limitation. It usually also points to a genuine
software-only ceiling → suggests HW change required for root fix.

### 3d. 对立观点主动搜索 (Adversarial applicability check)

For every **binding** you're tempted to write ("X 架构下本方案失效"),
actively search for the mirror claim: "is there an X configuration
where this method is *more* suitable?" Record format:

```
Binding: "MoE 不适用 (step 3), per-token expert route breaks M-major
         cooperation assumption"
Adversarial rebuttal: "If gating does expert→XCD affinity routing,
         different experts live on different XCD L2, which makes this
         MORE suitable than dense. Precondition: n_experts ≈ n_XCD
         (MI350: 8); matches DeepSeek-V2's 64 routed + 2 shared config."
```

Required for every binding. Even when conclusion is "no, binding holds
universally", the search process must be recorded.

## 4. 生态位 (Ecological niche — where this paper sits in the field)

### 4a. Paradigm-shift positioning

Does this paper convert an **implicit constraint** into an **explicit
software resource**? Examples:
- Fleet: implicit "L2 physical partition" → explicit `Chiplet-task`
- PagedAttention: implicit "KV cache contiguous allocation" → explicit
  page
- FlashAttention: implicit "SRAM ≫ HBM BW" → explicit tile-streaming
  kernel pattern

If yes, is the abstraction generalizable to other memory / compute
tiers?

### 4b. Adoption evidence (cross-paper and cross-framework)

- Merged into which upstream runtimes? (vLLM commit X, SGLang PR Y,
  TRT-LLM release Z, DeepSpeed branch W)
- Which other DB papers explicitly build on or cite this one
  (by [id])?
- Which production model cards reference the method?
- Stated vs verified: if paper claims "adopted by X", verify by
  reading X's notes or searching X's repo.

### 4c. Design bindings with cross-paper grounding

What does this method FORCE downstream? (chunked prefill + TP, fixed
register budget, specific scheduler, etc.) For each binding:
- Does any library paper propose a way to relax it?
- Does any library paper pay a *different* price to achieve similar
  gains without this binding?

## 5. 未探索方向 (Unexplored directions — hybrid + trade-off axes)

### 5a. Trade-off 轴 identification

For every row in notes §5 where the paper LOSES to a baseline:
- Is this a simple "not yet optimized" (will be fixed in v2) or a
  **fundamental trade-off axis** (mutually exclusive with this method's
  core)?
- If fundamental, what's the axis name? Examples: "cross-op
  persistent reuse vs single-kernel peak tuning", "static block size
  vs adaptive chunking", "shared expert coherence vs dispatch
  parallelism".

### 5b. Hybrid / adaptive directions

For each trade-off axis identified:
- Can an **adaptive / hybrid** policy win both regimes? (e.g. "small
  batch Fleet, large batch hipBLASLt"; "low concurrency continuous
  batch, high concurrency radix cache")
- Is there a library paper already exploring hybrid in adjacent
  domains whose pattern could transfer?
- Open a concrete research direction if none exists: give the hybrid
  a name and outline the runtime decision criterion.

### 5c. Cross-paper open questions

Aggregate open_questions across related papers (from Step 1). Flag
which are **answered** by this new paper, which remain **open**, and
which are **newly opened** by this paper's claims.
```

### Step 4: Regenerate HTML

```bash
python3 ~/.cursor/skills/paper-reader/scripts/generate_html.py
```

This rebuilds the relationship graph and cross-links in overview.html.
Synthesis documents themselves also render under `knowledge/` via
sync.sh.

### Step 5: Brief report to user

```markdown
### Synthesis Update (Mode 1)

**New paper**: {title} ({category})
**Synthesis doc**: knowledge/synthesis-{id}.md
**Related papers found**: {N}
**New cross-category link**: {category_A} ↔ {category_B} via {...}
**Key delta summary**: {1 sentence — this paper's single sharpest
  advance over the nearest library paper}
**New open question surfaced**: {...}
```

---

# Mode 2 — Graph Update only

Same as Step 1 + Step 2 + Step 4 above. Skip Step 3 (no synthesis doc
written). Use when the library's relationship graph needs a targeted
refresh without re-doing synthesis (e.g. after bulk tag edits).

---

# Mode 3 — Full Synthesis (user-requested)

Triggered when the user asks for trend analysis, knowledge map, gap
identification, or comprehensive review across the whole library.

### Procedure

1. Read `~/.cursor/paper-db/papers.json` in full
2. For papers with complex relationships, read their `notes/*.md` and
   `knowledge/synthesis-*.md` (Mode 1 outputs accumulate — reuse them)
3. Generate the full report using the template below
4. Save to `~/.cursor/paper-db/knowledge/synthesis-report.md`
5. Regenerate HTML

### Full report template

```markdown
# AI Infra Knowledge Synthesis — {date}
> Library: {count} papers across {N} categories
> Last Mode-1 synthesis: {date of most recent synthesis-*.md}

## 1. Category timelines
For each category, ordered by date_read:
- Paper → key contribution → forward link to Mode-1 synthesis doc

## 2. Cross-category connections
| From | Category | Relationship | To | Category |

## 3. Influence chains
Multi-hop: {paper (algo)} → {mechanism} → {paper (llm)} → {mechanism}
→ {paper (agent)}

## 4. Stack impact map
| Layer | Key enablers (paper ids) | Current bottlenecks | Trend |

## 5. Emerging trends (3–5)
For each trend:
- Signal, supporting papers, prediction, infra implication

## 6. Knowledge gaps
- Under-explored topics (< 2 papers in category × subtopic)
- Missing connections (infra_impact mentions that no paper addresses)
- Unanswered open_questions (grouped, deduplicated)

## 7. Recommended next reading
Prioritized topics or specific papers to fill gaps or address
persistent open questions.

## 8. Library health metrics
- Papers per category, per quarter
- Mode-1 synthesis coverage: do all papers have synthesis-{id}.md?
- Stale synthesis: any synthesis-{id}.md whose library has grown > 5
  papers in same category since synthesis date → schedule re-synthesis
```

---

## High-level thinking rules — DO / DON'T quick reference

These rules govern everything this skill writes. They are the "how to
think cross-paper" core that was moved out of paper-reader in v0.3.

### DO

- **Cite by [paper-id]** so downstream readers can follow the chain
- **Express conditionality as inequality** when possible
  (`regime ⇔ batch ≥ 2·T_M`)
- **Call out paradigm shifts explicitly** — "implicit X → explicit Y"
  is the format
- **Record adversarial search attempts** even when they fail; the
  process IS the discipline
- **Prefer calibrated phrasing** ("在 {regime} 窗口内错位, 主流仍是
  {Y}") over doomer maximalism

### DON'T

- Restate paper-internal content (TL;DR, architecture, formal proofs) —
  that's in notes
- Write `X is similar to Y` without stating the concrete delta
- Commit to "若 X 失效" counterfactuals without passing both term and
  plausibility gates
- Write universal-sounding critiques when the library only contains
  one narrow regime — acknowledge the library limitation

---

## History & rationale

Incidents that motivated these rules (Fleet MoE binding reversal,
PrfaaS 稻草人 critique, Seedance silent figure failures) are in
`~/.cursor/paper-db/incidents.md`. Read before starting a synthesis
document.
