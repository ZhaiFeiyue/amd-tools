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

# AI Infra Paper Reader — Automated Pipeline

This is an **orchestration skill**. When the user provides a paper or article,
execute ALL steps below automatically without waiting for confirmation between
steps. The user provides an input; you deliver the complete analysis.

## ⚠️ Pipeline Discipline (read first, every time)

This pipeline has **11 phases**. Skipping any of them — silently or "just for
this paper" — produces incomplete output that the user must catch manually.
That is unacceptable. Before reporting success to the user, you MUST:

1. Execute every Phase end-of-phase checklist (✅ boxes printed below each
   Phase). If any check fails, fix it before advancing.
2. Run the **final completeness gate** at the very end:
   ```bash
   python3 ~/.cursor/paper-db/tools/check_paper_completeness.py {paper_id} --strict
   ```
   This script validates papers.json fields, notes sections, image quality,
   drawio presence (for arch-bearing categories), published HTML, AND the
   per-paper cognitive HTML reader. If exit code is non-zero, **DO NOT
   present the Final Output to the user** — fix the gaps first.
3. If a phase legitimately cannot be completed (e.g. paper has no figures,
   so Phase 6 produces 0 PNGs), explicitly justify the deviation in the
   Final Output. Never quietly skip.

**Repeat-offender protection**: real incidents tracked in
`~/.cursor/paper-db/incidents.md`. Read that file before starting any new
paper to remind yourself of past failure modes.

### ⚠️ No Skill-Leakage in Paper Notes (MANDATORY)

> Real incident, 2026-04-21F: after the Mermaid-first policy shift, the
> K2.6 paper notes contained **5 paragraphs that were verbatim skill/rule
> explanations** — "Mermaid 的优势: (1) LLM 生成零错...", "工具选择
> 说明: 本笔记 2026-04-21 改用 Mermaid", "📁 多页 drawio ladder (备份,
> 多视角 tab 切换时用)", "文字说明 = 架构图的 caption, 不是架构图的
> 替代", "不贴大段 Python 代码块——原则是图里有的不写字". User caught
> it directly: "为什么最终生成的结果里很多 skill 控制的内容和 drawio
> 控制的内容?" See `incidents.md` 2026-04-21F for full postmortem.

**The rule**: Paper notes (`~/.cursor/paper-db/notes/{id}.md`) are
content-only artifacts. They answer "what is this paper?" — NOT
"why is this notes format chosen?" or "what are the tool selection
trade-offs?". Such meta content belongs in `SKILL.md`, `deep-*.md`,
`diagram-tool-choice.md`, or `incidents.md` — never in individual
paper notes.

**The universal test** (applies to every paragraph the agent writes):

> If a paragraph, dropped verbatim into a DIFFERENT paper's notes,
> would still make sense and be equally applicable — that paragraph
> is **skill-leakage** and must live in the skill, not in the paper.

**Forbidden content in paper notes** (any match is a blocker):

- ❌ Tool selection rationale (why Mermaid vs drawio, why drawio vs SVG)
- ❌ Policy change announcements ("本笔记 YYYY-MM-DD 改用 X"; "per
  policy change in ...")
- ❌ `deep-*.md` principle restatements ("图里有的不写字、字里有的
  不贴码", "文字说明不是图的替代")
- ❌ Cross-paper rule citations ("见 diagram-tool-choice.md", "按
  SKILL.md §X 执行", "deep-framework.md §12", "Phase 2b 强制")
- ❌ Classification-rationale meta ("### 分类依据 (hybrid
  disambiguation — Phase 3 Step 4)"; "thesis test / fit test" — the
  classification decision is in papers.json, don't document the
  4-step protocol in notes)
- ❌ Inheritance / trigger declarations ("🚨 Inherited universal
  rule ...", "📌 触发说明: ...", "本节强制执行")
- ❌ SJM abbreviation in section headers ("作者证明 (SJM)", "SJM —
  Solution Justification Mechanism", "SJM 类型学") — the Chinese
  "作者证明" is fine as content-describing, but the "(SJM)" suffix
  is the skill rule's acronym and leaks the skill taxonomy
- ❌ Fallback / ladder / variant explanations ("drawio 保留供多视角
  tab 切换", "📁 备份版本")
- ❌ Tool comparison tables (Mermaid vs drawio pros-cons)
- ❌ Meta statements about the notes format itself ("本节只写 X 不
  讨论 Y"; "本笔记 deep read 采用的是 Z 结构"; "applies the SKILL.md
  Phase 2 framework")

**Allowed meta-commentary in paper notes** (these are content, not skill):

- ✅ "Paper §3.4 claims X; we verify with code at file:line, conclude Y"
- ✅ "Table 5 shows 54% throughput but the number comes from Eq.3-8
  applied to the case-study profile" (paper-specific derivation)
- ✅ "The author calls this 'MTP', but the code (`modeling_xxx.py:N`)
  reveals it is actually a 5-layer dense transformer" (paper × code
  mismatch, paper-specific)
- ✅ `[论文未披露]` markers on specific missing data points
- ✅ Paper-level limitations and open questions

**Judgment heuristic for borderline cases**:

| Meta text candidate | Test: swap paper name for "X" — still true? | Action |
|---|---|---|
| "K2.6 architecture is identical to K2.5 (config.json field-by-field)" | False (K2.6-specific) | ✅ keep |
| "Mermaid is preferred because LLM generation accuracy is higher" | True (universal skill claim) | ❌ leak, move to skill |
| "The ignore regex `re:.*mlp\.(gate\|up)_proj.*` matches BOTH layer-0 dense AND shared expert" | False (K2.6-specific, config-derived) | ✅ keep |
| "图里有的不写字, 字里有的不贴码 — 原则是..." | True (universal rule) | ❌ leak, move to skill |
| "§3.4 Table 5 profile + Eq.3-8 → case study 54% number" | False (paper-specific derivation) | ✅ keep |

**When a new policy / rule is created**, the temptation to write
"this paper is the first under policy X, here's why X" INTO the paper
is strong. DO NOT. Record the policy + reasoning in `incidents.md`
and `SKILL.md`, then apply the policy silently in the paper. The paper
should look identical to one written 6 months from now by an agent
who never knew there was a transition.

### ⚠️ Skill-Modification Protocol (rule-stating ≠ rule-following)

When you (the LLM) **modify this SKILL.md or any `deep-*.md` to add a new
MANDATORY rule** (e.g. a new gate, a new required field, a new Phase
sub-step), you MUST in the SAME turn:

1. **Identify the affected scope**: which Phase does the new rule extend?
   Which paper categories does the affected Phase apply to?
2. **Find the affected papers** — run:
   ```bash
   python3 -c "
   import json, os
   db = json.load(open(os.path.expanduser('~/.cursor/paper-db/papers.json')))
   # filter by category if rule is category-specific, else all
   papers = sorted(db['papers'], key=lambda p: p.get('date_read',''), reverse=True)
   for p in papers[:3]:  # top 3 most recent
       print(p['id'], p.get('date_read'), p['category'])
   "
   ```
3. **Retroactively re-validate** at least the **most recent 1 paper** in
   the affected scope (top 3 if the rule is foundational like a Phase 2
   change). For each: re-read its notes, fix every gap the new rule
   exposes, re-run sync.sh, re-run `check_paper_completeness.py --strict`.
4. **Justify any deferred re-validation**: if you cannot retroactively
   fix older papers in this turn (e.g. >5 affected, scope too large),
   explicitly tell the user which papers are now "skill-stale" and
   propose a follow-up plan. Never silently leave older papers
   inconsistent with the new rule.
5. **Term-echo check — MANDATORY (added 2026-04-21I)**: for every new
   rule name / section title / abbreviation you introduce in the skill,
   ask yourself:

   > *"Will an LLM, having read this rule, echo this exact phrasing into
   > its output (paper notes)?"*

   If yes → the name is **leak-prone**. Either:
   - (preferred) Use a **content-level name** (e.g. "作者证明") that is
     BOTH the skill rule's name AND the expected section title in notes.
     No dual naming, no English abbreviation, no Phase-tag inside the
     section title. One name for one thing.
   - (fallback) Add the new phrase to `check_paper_completeness.py
     ::_SKILL_LEAK_PHRASES` red-flag list AND to SKILL.md §"No Skill-
     Leakage in Paper Notes" forbidden list. Make the ban explicit.

   **Historical leak examples** (enumerated in incidents.md 2026-04-21F
   and 2026-04-21H): "(SJM)" abbreviation, "Phase 2b" section suffixes,
   "Solution Justification Mechanism" full name, "hybrid disambiguation —
   Phase 3 Step 4" rule tag, "Inherited universal rule" callout style,
   "📌 触发说明" meta blocks. All of these leaked because the skill
   teaching-text used them; once agent had them in context, they echoed
   into every paper's notes. Fixed by (A) renaming the underlying rules
   to pure content-level names + (B) adding Phase 7 pre-save self-check.

**Why this rule exists** (real incident, 2026-04-19, PrfaaS rule-stating
≠ rule-following): the agent added "Argument Chain Reconstruction" to
SKILL.md Phase 2b and said "下一篇 paper 会强制执行", but did not
retroactively apply it to PrfaaS itself (the very paper that triggered
the rule). The paper's §3c/3e/Limitations still ran on the old free-form
template until the user pointed it out. **Self-serving scope narrowing
("已做完的不算 next") is a documented LLM failure mode**: when the agent
states a rule, it implicitly excludes its own current task from the
rule's scope.

**Test for compliance** before declaring this rule satisfied: open the
most recent affected paper's notes file, search for the new
rule's required artifact (e.g. "step k", "Term gate", "Argument Chain
table"). If the search returns 0 hits, you have NOT complied — you have
only stated the rule.

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

**GitHub repository** (URL like `github.com/owner/repo`, NOT pointing to a PDF):
→ This is source code. Category is automatically `code`. Clone and analyze:

```bash
REPO_URL="https://github.com/{owner}/{repo}.git"
CLONE_DIR="$HOME/.cursor/code-db/repos/{owner}-{repo}"
git clone --depth 1 "$REPO_URL" "$CLONE_DIR" 2>/dev/null || \
  (cd "$CLONE_DIR" && git pull --ff-only)
```

Extract metadata: repo name, description, primary language, stars, license.
Read README.md for project overview. Use `tokei` or `wc -l` for LOC breakdown.
The cloned repo is the "paper content" for subsequent phases.

**Code-specific enrichment** (run alongside Phase 1 for GitHub repos):
- **Usage guide**: provide a concrete `pip install` / `cargo build` + minimal working
  example (NOT copy-paste from README — a realistic end-to-end example)
- **Dependency analysis**: list the 5-10 most important dependencies with versions
- **Community stats** (use `gh` CLI):
  ```bash
  gh api repos/{owner}/{repo} --jq '{stars:.stargazers_count,forks:.forks_count,open_issues:.open_issues_count}'
  gh issue list -R {owner}/{repo} --limit 30 --state all --json number,title,labels,state
  gh pr list -R {owner}/{repo} --limit 20 --state all --json number,title,state,mergedAt
  gh api repos/{owner}/{repo}/contributors --jq '.[0:10]|.[].login'
  ```

**GitHub Pull Request** (URL like `github.com/owner/repo/pull/N`):
→ This is a PR. Category is automatically `code`. Fetch PR content:

```bash
gh pr view N -R owner/repo --json title,body,author,createdAt,state,labels,files,comments,reviews
gh pr diff N -R owner/repo
```

Extract: title, author, date, description, changed files, review comments.
The PR diff and discussion form the "paper content" for subsequent phases.

**GitHub Issue** (URL like `github.com/owner/repo/issues/N`):
→ This is an issue. Category is automatically `code`. Fetch issue content:

```bash
gh issue view N -R owner/repo --json title,body,author,createdAt,state,labels,comments
```

Extract: title, author, date, description, labels, discussion thread.
The issue content forms the "paper content" for subsequent phases.

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

### ✅ Phase 1 End-of-Phase Checklist

Before advancing to Phase 2, verify:

- [ ] Title, authors, date, source URL captured (not `[未知]`)
- [ ] **Full content** retrieved — for PDFs, you Read enough pages to cover
      all figures/tables (not just abstract or first few pages)
- [ ] For GitHub repos, you ran `gh` CLI commands and have stars/issues/PR
      counts in hand
- [ ] If the source is unreachable (404, paywall), you explicitly noted it
      and chose Tier 3 fallback — NOT proceeded with `[内容未抓取]` placeholder

---

## Phase 2: Summarize (Summary)

### Role: 学术论文首席解读者

You are a patient mentor who transforms obscure academic language into
clear, logical, accessible Chinese explanations. When encountering
technical terms (e.g. Attention Mechanism, Transformer), **always explain
with a real-life analogy/example first**, then give the academic definition.
Use bold for emphasis, `>` blockquotes for concept explanations.

### 0. Content Authoring Discipline (责任分工 + 写作原则)

**架构原则 — 内容 vs 格式**：
- **内容（content）= LLM 写**：所有用户可见的 prose（tldr / contribution
  / summary / key_findings / limitations / open_questions /
  deep_analysis[].body / table.takeaway / figure.what_shows /
  figure.why_matters）。脚本不会摘要、截断、拼接、改写任何 prose 字段。
- **格式（format）= 脚本做**：HTML 结构、CSS、字体、KaTeX delimiter、
  markdown 渲染（bold/list/table）、列对齐、`bold_best_per_col` 应用、
  图片 base64、重复检测、papers.json 派生。

**TL;DR (≤300 字, LLM 必填)**：
- 1-3 句"扫读卡"，给读者 5 秒判断要不要继续看
- 必须包含文章核心，按论文类型侧重不同：
  - **效果提升类**（如 SageAttn、PCD）→ 包含具体提升数字（"3× 加速 / GPQA +7.5"）
  - **算法类**（如 AttnRes、IN2）→ 简述算法核心（"用 softmax 替代固定权重残差"）
  - **项目代码类**（如 Sutradhara、KVFlow）→ 这个项目是做什么的、解决什么问题
  - **评估 / 诊断类**（如 NoLiMa、Lost in the Middle）→ 揭示了什么现象、对生态的影响
- 可包含 `$..$` 数学（KaTeX 渲染），但不要让公式占满整段
- 不要重复 `core_contribution` 字段——TL;DR 是"扫读"，contribution 是"完整一句话"

**Summary (≤300 字, LLM 必填)**：
- **从 Deep Analysis 再提炼一次**，不是 Deep Analysis 的复制
- 三段式（不强制分段，但要有逻辑次序）：motivation → method → results
- 每个论文类型的 Summary 必含元素同 TL;DR，但展开 1 层细节（数据集、参数、关键 baseline）
- 禁止重复 `core_contribution` 或 TL;DR 已有的句子

**Q1/Q2/Q3 (核心三问, 每问 ≤3 句)**：
- **三连必须递进，不能割裂**：Q1 痛点 → Q2 方法 → Q3 结果
- Q1 问 "为什么这是问题、之前怎么解的"
- Q2 答 "这次怎么解、与之前差在哪"
- Q3 给 "实测数字 + 与 baseline 对比"

**渐进披露 (Progressive Disclosure) 原则**：
- 顶部三连（TL;DR + Q1Q2Q3 + Key Findings）= 决策面板，5 秒读完
- 中部 Summary = 30 秒理解
- 底部 Deep Analysis = 5 分钟精读
- 各部分**不允许割裂**——上一层提到的数字 / 名词 / 关键概念，下一层要展开
- Figures / Tables / Diagrams **不再单独成区**，而是 `embed_at` 到讨论它们的
  Deep Analysis section 旁边（v2 schema 强制此规则）

**反例（要避免的"割裂"）**：
- ❌ TL;DR 说 "GPQA +7.5"，Deep Analysis 没解释怎么测的、对比谁
- ❌ Q3 说 "节省 1.25× compute"，Summary 不再提此数字
- ❌ 顶部 Key Figures 区独立罗列 3 张图，Deep Analysis 各 section 内文又重复说 "见 Figure 2"

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

- **作者证明 — MANDATORY**: 任何严肃论文都有一套**作者用来证明"我提出的方案是对的"的形式/半形式装置**。这套装置和"架构 (是什么)"、"创新点 (新在哪)"、"实验 (跑出多少数)"是**三件不同的东西**——作者证明是**架构和实验之间的桥**。识别 + 完整复现 作者证明 是 deep-read 的硬约束, 不允许仅以 "Eq.X 给出..." / "见 §Y" 一笔带过。

  > **Naming convention** (vocabulary hygiene, 2026-04-21I): this rule
  > uses exactly one name — **作者证明** — both in prescriptive text here
  > and as the expected section title in paper notes. We deliberately do
  > NOT give it an abbreviation, an English expansion, or a Phase-tag.
  > Earlier revisions used "作者证明 (Solution Justification Mechanism,
  > SJM)" — that pattern caused widespread leakage of the "(SJM)"
  > suffix into paper notes section titles. See incidents.md 2026-04-21I.

  **作者证明 类型学 (按论文类别速查)**:

  | 论文类别 | 典型 作者证明 形式 | 在论文里的位置信号 | 必须复现的内容 |
  |---|---|---|---|
  | framework / cluster | closed-form throughput / latency / cost model + 优化方程 | §"Modeling" / §"Analysis" / 编号 ≥3 的 Equation 出现在 §3 而非 §2 | notation 表 + 每条方程的物理含义 + 单调性 + 与 case study 数字的一阶映射 |
  | kernel | roofline / arithmetic intensity / SM occupancy 推导 | §"Performance Model" / 计算 FLOPs vs Bytes 比值的段落 | roofline 数值 + 实测 %peak + 为什么这个 tile size / launch config 是最优 |
  | algorithm | convergence theorem / loss decomposition / variance bound / sample complexity | "Theorem N." / "Proposition N." / "Lemma N." 块 + 假设清单 | 定理陈述 + proof sketch + 假设条件 + 假设 break 时结论是否仍成立 |
  | llm (architecture) | scaling-law fit / FLOPs-param-data 三角约束 / 各模块 param breakdown | 散点图 + 拟合曲线 + 缩放指数; 或 param 表 + capacity 推导 | 拟合方程 + 残差讨论 + 与 Chinchilla / DeepSeek 等参照系的偏离 |
  | hardware | compute / BW / power budget 表 + area 推导 + 性能投影模型 | "Performance projection" / 面积 vs 性能 trade-off 图 | 各资源 budget 表 + 每个设计选择对应的约束方程 |
  | agent | tool-call success rate model / latency budget per turn / planning depth bound | 多为经验 sweep, 偶尔有 success-prob 的解析模型 | sweep 矩阵 + 各 axis 单调性 + failure mode 分类 |
  | code (repo/PR) | API 不变量 / 算法复杂度 / 关键路径性能模型 | tests/types/comments + 性能基准 | 不变量列表 + 复杂度 + hot-path latency breakdown |

  **作者证明 识别启发式** (Phase 1 fetch 时就开始扫):
  - 出现在 §3+ (而非 §2 Background) 的**编号 Equation** ≥ 2 条 → 必有 analytical 作者证明
  - "Theorem / Proposition / Lemma / Claim N." 块 → formal proof
  - 标题含 "Model / Analysis / Cost / Budget / Bound / Capacity / Limit / Roofline" 的 §/§§ → 作者证明 落点
  - **Notation table** (符号表) → 作者证明 必备前置, 一定要抄录
  - 拟合曲线 + 数据点的散点图 → empirical-fit 作者证明
  - 全文找不到上述任何信号 → 显式标注 "**无形式化作者证明 — paper 只给实证**", 这是**警示信号**: 论文的 claim 缺少独立于实验的论证基础

  **作者证明 复现的最低要求** (deep-read 必须满足):

  1. **Notation 表完整复现** (如果有), 不要只引用论文 Table N
  2. **每条核心方程逐条解释**: 形式 → 物理含义 → 为什么是这个形式而不是别的 (e.g. "为什么是 min 不是 sum?", "为什么除以 p?", "为什么 RDMA 不进 min?")
  3. **单调性 / 凸性 / 唯一性** 论证 (如果作者隐含使用却没明写, 必须补)
  4. **从模型到 reported numbers 的一阶映射**: 把论文实验配置代入模型, 验证 case study 数字**真的是从模型一阶解出来的**, 而不是 sweep 出来再 fit 模型。这是检验作者证明是否 load-bearing 的硬测试。
  5. **"假设没有 case study, 我如何只用模型说服读者?"** — 用一句话给出纯论证版本。如果写不出来, 说明你没真正消化模型。
  6. **作者证明 的可攻击面**: 模型做了哪些简化 (Jensen 不等式? 平稳分布? closed-form 假设?), 这些简化在什么 workload / 规模下会破坏数字预测能力。这是 §3c assumption audit 的真正攻击点, 优于脑补反事实。

  **反模式** (要避免):
  - ❌ "throughput model 数学很优雅但好懂, 所以略过" — 易懂 ≠ 不必记录; load-bearing 不取决于数学难度
  - ❌ 把 作者证明 压缩成 §"Key Innovations" 表里的一行 — 一行无法承载 6 条最低要求
  - ❌ Figure caption 描述了 Eq 的几何形状就以为 model 已被覆盖 — figure 是 model 的可视化, 不是 model 本身
  - ❌ 抄了 case study 的结果表两遍, 但生成结果的 model 一笔带过 — show-the-result, skip-the-derivation 的失衡
  - ❌ 找不到作者证明时**伪造一个** — 没有作者证明是一种合法状态, 显式标注比硬凑一个伪模型更诚实

  **真实事故** (PrfaaS 2604.15039, 2026-04-19): paper §3.4 有完整 Throughput Model (Eq.3-8 + Table 4 notation + §3.4.2 单调性论证), 但 deep-read 只在 Summary 末尾抄了一句 Eq.6, 把整个推导当 "Eq.X 给出..." 的引用糖。下游 case study 的 54%/19.4K/N_p=3/N_d=5/13 Gbps **本是从 Table 5 profile + Eq.3-8 一阶解出**, 笔记里却像 sweep 出来的 magic numbers——论证桥梁断裂。**根因**: §3d 把 throughput model 标成 "数学很优雅但好懂", 把 load-bearing 误判成 decorative。详见 incidents.md 同日条目。

- **核心技术壁垒 (Core Technical Barrier):** Identify the ONE hardest-to-
  replicate technique that makes the whole method actually work in
  practice. This is not the high-level idea but the low-level engineering
  insight. Example: "NanoFlow的核心壁垒是custom execution unit scheduling
  ——限制kernel执行的SM个数。仅用108个SM中的35个(32%)，网络
  kernel即可实现92%峰值性能——这个非线性关系是整个方案能work的
  根本原因。" Every paper has one; find it and call it out explicitly.
- **论证链重构 (Argument Chain Reconstruction) — MUST RUN BEFORE 质疑假设
  和 设计绑定批判**: Every paper has a multi-step argument chain. Before
  attacking ANY part of it, reconstruct the chain explicitly as a numbered
  table:

  | Step | Premise (引用 §/Table/Fig) | Conclusion | Evidence |
  |---|---|---|---|
  | 1 | ... | ... | ... |
  | 2 | ... | ... | ... |
  | N | ... | 作者最终论点 | ... |

  Rules for the reconstruction:
  - Each step's conclusion must cite a paper section, table, or figure
    that explicitly states it (not your rephrasing).
  - **If you cannot map the paper's claim into ≥3 explicit steps, you have
    not read it carefully enough** — go back and re-read. A 2-step chain
    means you collapsed the paper into a strawman dichotomy.
  - Identify which step(s) are **load-bearing** (the whole conclusion
    falls if this step is wrong) vs **decorative** (could be removed
    without changing the main claim).
  - Note where the paper itself **already pre-empts** common objections
    (e.g. PrfaaS Table 3 already pre-empts "isn't GQA enough?" by
    showing 33-60 Gbps).

  Real incident (PrfaaS 2604.15039, 2026-04-19): the agent collapsed a
  3-step chain (MHA→不可行 / GQA-MLA→必要不充分 / hybrid stack→充分)
  into a 2-step dichotomy (hybrid vs dense), then attacked "若 dense 回归"
  — but step 2 in the real chain had already demonstrated GQA/MLA exists
  AND is insufficient. The attack landed on a battlefield the paper had
  already won. **Always reconstruct the full chain before critiquing.**

- **质疑假设 (Challenge Assumptions):** What statistical or empirical
  assumptions does the method rely on? Are they validated by independent
  evidence? Under what conditions might they break? When uncertain,
  explicitly say so: "这部分笔者也不是十分确定" — honest uncertainty
  builds reader trust more than false confidence.

  > 🎯 **每个质疑必须明确攻击论证链的哪一步**: 写"X is questionable"
  > 的时候, 先问 "X is step k 的 premise 还是 step k 的 conclusion?"
  > 如果你说不出 k, 你的质疑是在攻击你自己脑补的版本, 不是论文。
  > 例: PrfaaS 的可质疑点是 step 3 的 premise "hybrid stack 持续主流"
  > (medium-beta), 而不是 step 1 的 "MHA 不可行" (那已经被 step 2 让位)。

- **设计绑定批判 (Design Binding Critique):** What prerequisites does
  the method FORCE? List all forced dependencies (e.g. "NanoFlow
  强制绑定了chunked prefill和张量并行——这两个条件不再绑定时，
  很多设计可能会面临挑战"). Discuss what happens if these prerequisites
  are removed or changed.

  > 🎯 **每条 binding 必须对应论证链中一个 load-bearing premise**: 如果
  > binding 攻击的是 decorative premise, 移除它根本不影响 main claim,
  > 这条 binding 就没价值。例: PrfaaS 的真实 binding 是 step 3 的
  > "hybrid stack 路线持续", 不是 step 1 的 "非 MHA 模型存在"。

  > 🚨 **Ecosystem Reality Check (生态现状校验) — MANDATORY for every
  > "若 X 失效 / 若 X 回退 / 若 X 被替代" counterfactual you write in
  > both 质疑假设 and 设计绑定批判**.
  >
  > Real incident (PrfaaS 2604.15039, 2026-04-19): the agent wrote
  > "若未来流行模型回归 dense, PrfaaS 论点失效"——但论文里 "dense"
  > 已经包含 GQA/MLA, 而 dense MHA 自 Llama-3 (2024) 之后已经从生产
  > 环境消失。这是一个**稻草人批判**: 形式上是 critical, 实际上攻击
  > 一个不会发生的场景。
  >
  > Before writing any "若 X ..." counterfactual, run BOTH gates:
  >
  > 1. **Term gate (术语锚定)**: Is the paper's "X" the same as casual
  >    usage of "X"? Cite the paper's Table/Section defining it. Common
  >    failure: "dense" in 2024+ papers usually means GQA/MLA-only, not
  >    MHA; "long context" in 2025+ usually means ≥128K, not ≥8K;
  >    "MoE" sometimes excludes shared-expert designs.
  > 2. **Plausibility gate (生态可达性)**: Can you name ≥2 production
  >    deployments / shipped models in the last 12 months where X is
  >    the dominant choice (or has a credible revival path)? If not,
  >    the counterfactual is a hallucinated risk — drop it or reframe
  >    to a real adjacent risk.
  >
  > **Doomer ≠ deep**: maximalist phrasing like "全部论点失效 / 整个
  > 方案崩塌" is a smell that you are reaching for impressiveness rather
  > than grounded skepticism. Prefer calibrated, ecosystem-grounded
  > phrasing such as: "在 X 这条特定 alternative 出现的 1-2 年窗口内
  > 可能错位; 当前 (YYYY-MM) 主流仍是 Y, 短期 bet 安全。"
  >
  > **Why this rule exists**: LLM critical-analysis failure mode #1 is
  > running inside the paper's internal coordinate system without ever
  > exiting to external ecosystem reality. The form of critique is
  > correct but the grounding is misplaced — a straw-man dressed as
  > deep skepticism. The fix is to explicitly require an "outside view"
  > check before any counterfactual is committed to the notes.
- **拆解 (Deconstruction):** Concrete steps from input to output (numbered 1, 2, 3 list).
- **实践上下文 (Deployment Context):** Where does this method apply in
  real systems? (e.g. prefill vs decode, training vs inference, which
  GPU architectures benefit most, ecosystem integration)
- **生态影响追踪 (Ecosystem Influence):** Trace the paper's influence
  on downstream systems. Has it been adopted by vLLM, SGLang,
  TRT-LLM, or other frameworks? Did subsequent papers (e.g.
  DeepSeekV3) adopt similar ideas? This connects the paper to the
  living ecosystem rather than treating it as an isolated work.

- **根本性 vs 缓解性审视 (Root-cause vs Mitigation Critique) — MANDATORY**:
  对每一个 "paper claims X× / +Y%" 的性能提升类贡献，必须显式回答三个问题：
  1. **这是根本性突破（universal improvement，across所有场景）还是
     条件性缓解（conditional，依赖某种 regime / batch / scale 条件）？**
  2. **在什么前提下有效？前提是 paper 自己明确说的，还是你读进去才发现的？**
     把前提列成不等式（如 `m_tiles ≥ 2 ⇔ B ≥ 2·T_M`）。
  3. **前提消失时加速如何归零？**按分项拆开：哪部分加速保留
     （如 dispatch compression）、哪部分归零（如 L2 协作复用）。
  如果能得出"某 regime 下效果 ≈ 0"的定量结论，这就是该 paper 最诚实的 limitation。
  > **Why this rule exists** (real external review, 2026-04, Fleet paper):
  > 外部读者立刻把 Fleet 的 bs=1 加速 16.9% vs Mirage 16.4% 看出来 ——
  > 在延迟最敏感的单请求场景里，Fleet 对"内存墙"根本没破，仅靠 dispatch
  > 压缩吃了 1.16× 加速。我们原本的笔记把这个观察写了，但没上升到
  > "根本性 vs 缓解性" 的审视高度。外部读者还进一步指出："真正根本性
  > 的解决可能需要硬件层面变革 (跨 chiplet L2 共享 / 一致性协议优化)"——
  > 这是软件方案的天花板，值得显式声明。
  > **格式要求**：每篇 paper 在 §3c assumption audit 里必须有一小节
  > "根本性 vs 缓解性" 标注，不允许省略。

- **Trade-off 轴识别 + 未探索 Hybrid (Trade-off Axis + Unexplored
  Hybrid) — MANDATORY**: 当 paper 出现一个 "失败场景"（某 batch / model
  size / hardware 下 paper 方案被 baseline 反超），不要满足于作者给的
  解释（"实现未完善 / future work"）。强制回答：
  1. **这是否暴露了一个根本的 trade-off 轴？**（如 Fleet bs=64 反超不是
     "K-split 没做"那么简单，是 **"跨算子持久化复用"与"单 kernel 极致调优"
     本身就互斥**——前者要求 kernel 用固定 register budget + 单波占用；
     后者要求 wave-level K-split + register spilling 自由度）
  2. **作者是否考虑了 hybrid / adaptive 组合？**（如 "小 batch 用 Fleet，
     大 batch 切换到 hipBLASLt" —— 这种 adaptive 策略是 paper 通常不讨论
     的 free lunch）
  3. 若未探索，**你作为读者能否开一条独立研究方向**描述该 hybrid？
     （如 "persistent megakernel with runtime fusion granularity switch"）
  > **Why this rule exists** (real external review, 2026-04, Fleet paper):
  > 外部读者把 bs=64 Fleet 落败 vLLM 解读为 "跨算子复用 vs 单算子极致
  > 的张力"，并推导出 "最优策略是混合式自适应调度" —— 一个作者完全
  > 没探讨的方向。我们原本的笔记只说 "bs≥64 让位 hipBLASLt"，没识别
  > 这是 trade-off 轴，也没开辟自适应 hybrid 方向。
  > **触发条件**：只要 paper 有任何"baseline 在某 regime 胜出"的数据点，
  > 这条规则强制触发，必须在 §3e 或 Open Questions 写出 hybrid 方向。

- **对立观点主动搜索 (Adversarial Applicability Check) — MANDATORY when
  writing "X 不适用 / X binding severe"**: 每当你在 §3e 或 Limitations
  写下一条 "X 架构/场景下本方案失效" 的 binding critique，必须**主动
  反向搜索**：是否存在某种 X 的配置、变体、或组合，反而让本方案**更适合**？
  格式：
  ```
  Binding: "MoE 不适用 (chain step 3)，因为 per-token expert route 破坏
           M-major 协作假设"
  Adversarial rebuttal: "**但如果 gating network 做 expert→XCD 亲和路由**，
           不同 expert 驻留不同 XCD 的 L2，反而比 dense 更契合 chiplet
           抽象。前提：expert 数 ≈ XCD 数（MI350: 8）; 这恰好对应
           DeepSeek-V2 的 64 routed + 2 shared 等 MoE 配置。"
  ```
  > **Why this rule exists** (real external review, 2026-04, Fleet paper):
  > 我们笔记里坚定写 "Binding 3 (最严重): MoE per-token route 破坏 M-major
  > 协作前提"，外部读者却得出**完全相反**的结论："MoE 专家稀疏激活天然
  > 具有数据并行特性，不同 expert 驻留不同 XCD 的 L2，gating 做 expert→XCD
  > 路由，**比 dense 更契合** Fleet 抽象"。同一现象，两种解读，我们的是
  > Fleet 的现行设计视角（M-major 协作），他们是**重新设计 Fleet 给 MoE**
  > 的视角。两者都对，但只写一条就是视角狭窄。
  > **格式要求**：每一条 binding / limitation 必须附一个 "Adversarial
  > rebuttal" 尝试，即使结论是"找不到反例，binding 确实普适"。

- **软件存在性证明 → 硬件反向推演 (Software-to-Hardware Existence Proof)
  — CATEGORY-CONDITIONAL, NOT universal**: 适用范围按类别**显式区分**，
  不是基类强制：
  - ✅ **`kernel`**: MANDATORY (kernel 直接操纵硬件原语，每条优化都对应
    未来硬件指令 / cache scope / occupancy 建议) → 详见 `deep-kernel.md`
  - ✅ **`hardware`**: MANDATORY (反向适用：白皮书 → 软件可以新做什么)
    → 详见 `deep-hardware.md`
  - ✅ **`cluster`**: MANDATORY (NIC / RDMA / Ultra Ethernet 硬件特性
    需求) → 详见 `deep-cluster.md`
  - ⚠️ **`framework`**: **CONDITIONAL** — 仅当 paper 是 hardware-proximal
    (持久化 megakernel / cache scope 控制 / chiplet affinity / KV cache
    物理布局 / 互连感知调度等) 才触发；纯软件 framework (continuous
    batching / request routing / load balancing / KV prefix cache 策略)
    **不触发**此规则 → 详见 `deep-framework.md`
  - ❌ **`algorithm` / `llm` / `agent` / `code`**: 默认不触发（这些类别
    的 paper 通常与硬件特性只有间接关系；特殊情况如 "量化 paper 给硬件
    tensor core 提建议" 可以手动 opt-in，但不是默认 checklist 项）

  **Why this scoping rule exists** (user feedback, 2026-04, Fleet):
  把 software→hardware 规则放基类等于强制所有 paper 都写"这对未来硬件
  有什么启示"——对纯框架 paper (continuous batching / 请求调度) 这是
  一种硬凑。正确做法是按类别决定是否适用；Fleet 之所以适用是因为它
  不是纯框架，而是 framework-that-exposes-hardware-features，属于
  conditional-triggered 的子集。具体模板和检查项下移到各 `deep-*.md`。

- **范式演进定位 (Paradigm-Shift Positioning)**: 除了 SOTA 数字，问：
  这篇 paper 是否把某种"隐式硬件约束"转变成了"显式软件资源"？
  （如 Fleet 把 "L2 物理分区" 从透明约束变成 `Chiplet-task` 显式资源；
  vLLM PagedAttention 把 "KV cache 连续分配" 约束变成 page 显式资源）
  如果是，这种"隐式约束→显式资源"的抽象能否推广到其他内存层次 /
  计算资源？这是把单篇 paper 放进范式演进坐标轴的关键视角。

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

#### Mathematical Formula Convention (MANDATORY)

All formulas in notes MUST use standard LaTeX notation with `$...$` (inline)
and `$$...$$` (display block) delimiters. These are rendered by KaTeX in the
HTML output.

**Correct** (LaTeX delimiters, KaTeX will render):
```markdown
The attention mechanism computes $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.

$$L = -\sum_{i=1}^{N} y_i \log p_i$$
```

**WRONG** (plain text, renders as ugly monospace):
```markdown
The attention mechanism computes softmax(QK^T / sqrt(d_k)) * V.

L = -Σ y_i * log(p_i)
```

Rules:
- Use `\frac{}{}` not `/`, `\sqrt{}` not `sqrt()`, `\sum` not `Σ`
- Use `\text{}` for text labels, `\operatorname{}` for custom operators
- Use `^\top` for transpose, `\cdot` for dot product
- Subscripts: `x_i`, superscripts: `x^2`, grouped: `x_{ij}`, `x^{n+1}`
- Greek letters: `\alpha`, `\beta`, `\gamma`, `\theta`, `\lambda`, `\sigma`
- Display formulas get their own `$$...$$` block, never inline

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

### ✅ Phase 2 End-of-Phase Checklist

Before advancing to Phase 3, verify:

- [ ] **All 16 sub-items in 2b are answered** (时代定位 / 背景 / 约束推导 /
      破局 / **作者证明** / 核心壁垒 / 论证链重构 / 质疑假设 /
      设计绑定批判 / 拆解 / 实践上下文 / 生态影响 /
      **根本性 vs 缓解性** / **Trade-off 轴 + 未探索 Hybrid** /
      **对立观点主动搜索** / **软件→硬件反向推演** / **范式演进定位**).
      Missing items = shallow read = unacceptable.
- [ ] **作者证明满足 6 条最低要求**: notation 复现、每条方程
      逐条解释、单调性/凸性论证、模型→数字的一阶映射、"无 case study
      纯模型防守"、可攻击面。如果论文确实没有作者证明, 显式标注
      "无形式化作者证明 — 仅实证" 而不是硬凑。
- [ ] **"根本性 vs 缓解性" 审视存在**：每个性能贡献都标注了是 universal
      improvement 还是 conditional，前提是什么，前提失效时效果如何。
      仅写 "bs=1-8 有 1.3× 加速" 而不说 "bs=1 加速 100% 来自 dispatch
      压缩而非 L2 复用" 是 shallow read。
- [ ] **Trade-off 轴 + 未探索 Hybrid 已识别**：paper 里每一个"baseline
      反超本方法"的数据点都对应到了一根 trade-off 轴的描述，且在
      Open Questions 里至少列了一条 hybrid / adaptive 方向。
- [ ] **每条 binding / limitation 都附 Adversarial rebuttal**：即使
      rebuttal 结论是"找不到反例 → binding 确实成立"也必须显式写出
      尝试过程。单向 binding 是视角狭窄的信号。
- [ ] **"软件→硬件反向推演" 按类别 conditional 触发**：`kernel` / `hardware`
      / `cluster` 默认强制；`framework` 仅当 paper 是 hardware-proximal
      (持久化 kernel / cache scope 控制 / chiplet affinity / 互连感知
      调度) 才触发；`algorithm` / `llm` / `agent` / `code` 默认不触发 —
      除非论文显式给硬件提建议 (如量化 paper → tensor core)。具体模板
      见对应的 `deep-*.md`。
- [ ] You identified **3-5 priority figures** + **3-5 priority tables**
      and recorded them for Phase 6.
- [ ] All math formulas use LaTeX with `$...$` / `$$...$$` — NO Unicode
      math glyphs (Σ, ∑, √, α, β, …) outside fenced code blocks.
- [ ] 2e structured fields drafted in your scratch buffer (will be
      written to papers.json in Phase 4).

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
| `code` | Source code repos, pull requests, GitHub issues, design docs | Architecture analysis, critical path tracing, code quality, PR review, issue triage, community health, API design, performance hotspots | vLLM/SGLang/Triton repo reading, key PRs (new scheduler, kernel optimization), RFC issues, design proposals |

Also assess **infrastructure impact** — for each of the OTHER categories,
one sentence on how this paper affects that layer. `"N/A"` if no connection.

### Hybrid Paper Disambiguation (MANDATORY decision protocol)

Many real papers straddle two categories. To avoid silently picking the
less-appropriate one (and then missing all the category-specific sections
in its deep-read guide), follow this **explicit 4-step protocol** whenever
two categories feel plausible:

#### Step 1: List the paper's contributions with a 1-word category tag

Write out every distinct contribution with its most natural category
label. Example for Fleet (2604.15379):

| Contribution | Natural tag |
|---|---|
| Chiplet-task abstraction (new programming layer) | framework |
| Persistent megakernel runtime | framework |
| Per-XCD software scheduler | framework |
| M-major windowed tile traversal | kernel |
| 3-tier cache modifier policy (sc1/nt bits) | kernel |
| Hierarchical 2-level event sync (ISA-level atomics) | kernel |

If one category clearly dominates (≥ 70% of contributions), pick that
as primary, done.

#### Step 2: If split ≥ 30/70 or closer → apply the "thesis test"

Ask: **what is the paper's one-sentence thesis?** Look at the TITLE +
first sentence of Abstract + the sentence "We present X, ..." (usually
§1 last paragraph).

- If the thesis is **"a new abstraction / system / runtime that enables X"**
  → the paper is **framework / llm / agent / cluster / code** depending on
  where the abstraction lives
- If the thesis is **"an optimization technique for operator / kernel X
  that achieves Y"** → **kernel**
- If the thesis is **"a new architecture / chip / instruction Y"** → **hardware**
- If the thesis is **"a training method / loss / RL recipe for Y"** → **algorithm**

Fleet title: "Hierarchical Task-based **Abstraction** for Megakernels"
→ thesis = new abstraction → **framework** primary.
HipKittens title: "Fast and Furious AMD **Kernels**" → thesis = kernel
library → **kernel** primary.

#### Step 3: Cross-check with the deep-*.md fit test

Open both candidate `deep-*.md` guides mentally. For each:
- How many of that guide's numbered sections would have meaningful,
  non-padding content for this paper?
- How many core sections would you have to mark `[N/A]` or stretch?

Pick the guide that has ≤ 10% `[N/A]` sections. If both guides fit
equally well, pick the one whose **#1 section** best matches the
paper's #1 contribution.

Fleet fit test (illustrative):
- `deep-framework.md` 12 sections → 10 fit well, 2 stretched (§8 API,
  §10 Comparison matrix is weak since Fleet has no OpenAI-compat API)
- `deep-kernel.md` 12 sections → 11 fit well, 1 `[N/A]` (§10 Source
  walkthrough since Fleet code not public)

By fit test, `deep-kernel.md` actually fits slightly better. But Step 2
thesis test says `framework`. **This is a real hybrid** — either choice
is defensible. In this case, **framework wins** because the thesis test
is more load-bearing than the fit test (thesis = author's framing;
fit = our analysis tooling convenience).

#### Step 4: Record the hybrid decision in the paper's notes

When you picked category X over Y for a hybrid, write a 2-3 sentence
justification at the top of the notes (under Category line) like:

```markdown
> Category: framework | Secondary tags: kernel, ...
> Note on classification: Fleet is framework/kernel hybrid. Picked
> `framework` primary because the thesis test says "task-based
> abstraction" (§1, title), even though ≈50% of contributions are
> kernel-level (§4 cache modifier, §5.2 ISA sync). `kernel` is the
> strongest secondary tag, explicitly surfaced. Deep-read ran on
> deep-framework.md but consciously imported §12 Software→Hardware
> from deep-kernel.md (via the conditional trigger in deep-framework
> §12).
```

This makes the classification decision auditable and helps future
readers who grep for "kernel" find the paper.

**Why this rule exists** (user feedback, 2026-04, Fleet): Fleet was
filed as `framework` with 0 `kernel` mentions in `secondary_tags`,
even though half the paper is ISA-level kernel work. Silent primary-
category pick + missing secondary tag = paper is harder to find by
category and loses the kernel-specific deep-read section coverage.
The disambiguation protocol + mandatory hybrid justification fixes
both.

### ✅ Phase 3 End-of-Phase Checklist

- [ ] Primary category picked from the 8 above (not made up)
- [ ] 2-5 secondary tags drafted
- [ ] **All 6 `infra_impact` keys filled** — `algorithm / kernel / framework /
      llm / agent / code` — even if the value is `"N/A — reason"`. Empty
      strings will fail the completeness check.
- [ ] **Hybrid disambiguation protocol run** if two categories felt
      plausible: Step 1 contribution-list produced, Step 2 thesis test
      answered, Step 3 deep-*.md fit test compared, Step 4 hybrid
      justification written at top of notes. If paper was NOT a hybrid,
      write "N/A — single-category paper" explicitly to prove you
      considered it.
- [ ] **Cross-category content gets a secondary tag** — if the paper
      has a substantial body of kernel work but you picked `framework`,
      `kernel` MUST be in `secondary_tags`. Missing a secondary tag
      where the content clearly belongs is a documented failure mode.

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
        "agent": "...",
        "code": "..."
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
- Code: ...
```

**Do NOT stop here.** Proceed immediately to Phase 5.

### ✅ Phase 4 End-of-Phase Checklist

- [ ] papers.json entry written with **all 17 required fields** populated
      (run `python3 ~/.cursor/paper-db/tools/check_paper_completeness.py
      {paper_id}` — even mid-pipeline this will tell you which fields are
      empty).
- [ ] `key_findings` has ≥ 3 items
- [ ] `secondary_tags` has ≥ 2 items
- [ ] Notes file created at `~/.cursor/paper-db/notes/{paper-id}.md`
      with all template sections present (even if some are placeholders
      to be filled in Phase 5/7).

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
| code | [deep-code.md](deep-code.md) |

### Inheritance Model (IMPORTANT)

SKILL.md (this file) is the **base class (总纲)**. Each `deep-*.md` is a
**subclass (子类)**. The relationship follows OOP inheritance:

```
SKILL.md Phase 2 (逻辑故事还原)    ← base: ALWAYS executed first
    │
    ├─ deep-algorithm.md           ← extends Phase 2 + algorithm-specific
    ├─ deep-kernel.md              ← extends Phase 2 + kernel-specific
    ├─ deep-framework.md           ← extends Phase 2 + framework-specific
    ├─ deep-llm.md                 ← extends Phase 2 + llm-specific
    ├─ deep-agent.md               ← extends Phase 2 + agent-specific
    ├─ deep-cluster.md             ← extends Phase 2 + cluster-specific
    ├─ deep-hardware.md            ← extends Phase 2 + hardware-specific
    └─ deep-code.md                ← extends Phase 2 + code-specific (repos, PRs, issues)
```

**Inheritance rules**:
1. Phase 2（时代定位、约束推导、核心技术壁垒、质疑假设、设计绑定批判、
   生态影响追踪）是所有类别的 **公共基础**，在 deep read 之前已执行
2. 每个 `deep-*.md` 可以 **扩展 (extend)** Phase 2 —— 补充
   category-specific 的约束推导和技术壁垒分析
3. 当 `deep-*.md` 的要求与 Phase 2 通用要求 **冲突** 时，
   **以 deep-*.md 为准**（子类 override 基类）
4. 如果 Phase 2 对某些分析做得不够深（如约束推导过于表面），
   deep read 阶段必须 **补深**，不能放过

**Procedure**:
1. Read the guide file using the `Read` tool
2. **Review Phase 2 quality**: check that notes already contain 约束推导、
   核心壁垒、假设审计、设计绑定。If any is shallow or missing, deepen
   it during this phase.
3. For EACH analysis section in the guide, write the analysis for this paper
4. Collect all analysis into a structured deep-read output

This step is NOT optional. Every paper gets a deep read.

### Diagram Tool Selection — Mermaid preferred, drawio for specific cases, SVG for hero (适用所有类别)

**CRITICAL — applies to EVERY category and EVERY diagram in the notes.**

> **Policy changed 2026-04-21** from "drawio ONLY" to Mermaid-first.
> Reason: drawio's XML-handwriting error rate, large-file embed
> instability, and劳动-cost drove 3 separate incidents in 24h
> (2026-04-21 A/B/D). Full selection rationale + templates:
> `~/.cursor/skills/paper-reader/diagram-tool-choice.md`.

#### The three tools

| Tool | Use when | Strengths | Avoid when |
|---|---|---|---|
| **Mermaid** (DEFAULT) | any single-viewpoint diagram (pipeline / decoder block / encoder expand / MoE / training recipe / swarm topology / sequence / state machine) | LLM generation accuracy very high, line-level diff, auto-layout, native browser SVG, already wired into sync.sh | > 30 nodes, precise geometry, multi-page tab switching |
| **drawio** (FALLBACK) | multi-page "high→low" ladder (≥ 3 coordinated views), chip floorplan / die, precise proportional geometry, forking a predecessor paper's .drawio | tab switching, precise coordinates, heavy annotation density | any case where Mermaid would do — drawio is not the default anymore |
| **SVG** (SPECIFIC) | Hero figure for Phase 10 reader, one-off artistic sketch, pedagogical cartoon | hand-drawn aesthetic, vector-scaling, full style control | architecture diagrams (→ Mermaid/drawio), charts/plots (→ matplotlib→PNG) |

**What's always BANNED everywhere**:

- **ASCII box art in markdown code blocks** — `┌─` `│` `─┐` and friends.
  Misaligns after HTML render, can't zoom/pan/click. No exceptions for
  structural diagrams.
- **Code blocks as architecture substitute** — see `deep-llm.md §1.05`
  "Diagrams-over-code principle". Pasting `class Foo(nn.Module)` is
  code transcription, not architecture. Use Mermaid/drawio + file:line
  citations instead.

**The only places ASCII text-flow is allowed**:

- Linear 3-4-step text flow where no spatial layout is implied
  (e.g. `SFT → Distill → RL`, or `prefill → decode → output`)
- Plain markdown tables (use markdown table syntax, not ASCII grids)
- Short code snippets / shell commands / JSON-YAML config
- Simple bullet lists

Everything else → Mermaid first, drawio for the specific fallback cases.

#### Decision matrix (quick reference)

| Scenario | Preferred | Fallback |
|---|---|---|
| Top-level pipeline / data flow / multimodal routing | **Mermaid flowchart TB/LR + subgraph** | drawio single page |
| Decoder block proj-level expand (MLA / MoE / attention) | **Mermaid flowchart + nested subgraph** | drawio |
| Vision / audio / speech tower internals | **Mermaid flowchart LR** | drawio |
| Training pipeline (multi-stage recipe) | **Mermaid flowchart LR** | drawio |
| Quantization layout (what's quantized vs FP16) | **Mermaid flowchart + classDef** | drawio |
| Agent swarm topology / heterogeneous agents | **Mermaid flowchart + peer edges** | drawio |
| Request lifecycle / tool-call trace | **Mermaid sequenceDiagram** | — |
| State machine (agent runtime, VM lifecycle) | **Mermaid stateDiagram-v2** | — |
| ≥ 3 coordinated views of same system (need tab switch) | **drawio multi-page** | Mermaid split into multiple diagrams |
| Chip floorplan / GPU die / SM/CU physical geometry | **drawio** | — |
| Hero figure / pedagogical cartoon | **hand-drawn SVG** | — |
| Linear 3-step text (`A → B → C`) | **Markdown text** | — |
| Table of numbers | **Markdown table** | — |

#### Workflow (Mermaid-first path)

1. **Plan diagrams up front** — before writing notes, list the diagrams
   you need (typically 3-6: overview + sub-systems). For each, decide
   which tool from the matrix above.
2. **Write Mermaid inline in notes** under fenced `` ```mermaid ```` ``
   blocks. sync.sh wraps them as `<div class="mermaid">...</div>` and
   loads mermaid@11 with the canonical theme (`MERMAID_BOOTSTRAP_JS`
   in sync.sh around line 465).
3. **If and only if** the scenario needs multi-page ladder / precise
   geometry / pre-existing drawio fork, create the .drawio at
   `/apps/feiyue/upstream/zhaifeiyue.github.io/assets/<paper-id>_arch.drawio`
   and embed via `{{drawio:<file>.drawio#page=N&height=NNN}}`.
4. **For reader HTML** (Phase 10), follow the lazy-load pattern in
   `html-reader-guide.md §5` regardless of which tool you used.
5. **Preview with sync.sh** — verify Mermaid blocks render (open
   `papers/<id>.html` in browser or check for `class="mermaid"` divs).

#### Pragmatic hints

- **Start with Mermaid; fall back only when necessary**. If you find
  yourself hand-calculating x/y coordinates in drawio, that's a smell —
  the Mermaid auto-layout would have saved you the time.
- **Never mix-match inside one diagram** — each block is Mermaid OR
  drawio OR SVG, not a Frankenstein.
- **For algorithm pseudocode, use ``` ```pseudocode ``` ``` block** —
  pseudocode is code, not a diagram.
- **Complex branching training pipelines (e.g. Thinker SFT → off-policy
  / on-policy parallel → merge → GSPO)**: Mermaid `flowchart LR` with
  `&` multi-target edges handles this cleanly. Only drop to drawio if
  the branches cross > 5 layers visually.

#### Mermaid common pitfalls (one-liners)

- `|` inside a label → use `&vert;` or `&#124;`, NEVER `&pipe;` (not
  a valid HTML entity, will render as literal text — K2.6 bug)
- `<` / `>` in labels → `&lt;` / `&gt;`
- `classDef` → define at top, apply with `node:::className` (triple-colon)
- Nested subgraphs need matching `end` for each; auto-layout handles
  up to ~2 levels well, flatten beyond that
- `<br/>` for line breaks in labels; `<b>...</b>` and `<i>` also work
  (htmlLabels already enabled in sync.sh mermaid config)

See `diagram-tool-choice.md` for full templates + more pitfalls.

### Category-specific Dual Priority (架构 + 训练 同等重要)

For `llm` / `algorithm` / `hardware` / `code` papers that describe a **model or system
being trained/built**, TWO top-level sections are always mandatory and must be
equally deep:

1. **Model / System Architecture** — structure, dimensions, data flow, design rationale
   for every choice. All diagrams in this section follow the Universal Diagram
   Rule above (drawio only).
2. **Training / Build Methodology** — how it was trained or constructed: data volume
   (tokens / samples) broken down by stage; data mix and filtering pipeline; compute
   budget (GPU-hours, hardware, MFU); full multi-stage training recipe table (goal,
   data, LR, context, techniques per stage); itemized training techniques (parallelism,
   precision, optimizer, stability tricks, post-training DPO/GRPO/distillation); and
   explicit identification of the ONE hardest-to-replicate training barrier.

**Never summarize training as a one-liner like "trained on 2T tokens with SFT+RLHF".**
If the paper itself is terse on training, explicitly flag `[论文未披露]` for each
missing field rather than skipping the section. Readers need to know what is known
vs. what the authors withheld.

### Paper × Implementation Cross-reference — UNIVERSAL RULE (all categories)

> 🚨 **MANDATORY for EVERY paper / blog / whitepaper whenever ANY relevant
> code is publicly available.** This includes the paper's own repo, an
> open-source predecessor it builds on, a baseline it compares against,
> or a downstream inference framework that has already shipped the method.
> **Code trumps prose** — papers routinely simplify, omit, or mis-label
> implementation details (e.g. Qwen3-Omni paper calls its residual predictor
> "MTP module" but the code is a 5-layer dense transformer running 15 AR
> iterations with 15 independent LM heads — fundamentally different from
> what "MTP" usually means). Not cross-referencing code when it exists is
> a documented failure mode.
>
> **Formulation of the rule**: "如果 paper 或文章中有对应代码实现，必须
> 结合代码解读"—— if the paper, its predecessors, or its baselines have
> open-source code, the deep read MUST walk through that code and let the
> code correct / enrich / verify the paper's claims.

#### 🚫 No Backbone-Reuse Exemption (MANDATORY anti-shortcut clause)

> **Real incident, 2026-04-21, Kimi K2.6**: paper reused K2.5's backbone
> verbatim (HF config.json byte-identical). Agent concluded "architecture
> unchanged → drawing it again is redundant" and wrote "see K2.5 notes
> 互动图" instead of producing K2.6's own architecture drawio + code
> walkthrough. This is the **backbone-reuse shortcut** failure mode.
> See `incidents.md` 2026-04-21 entry for the full postmortem.

**The rule**: if paper N reuses paper M's architecture (stated explicitly,
or detected by config equality, or by "same class name in modeling code"),
paper N's notes and drawio MUST STILL CONTAIN ITS OWN ARCHITECTURE CONTENT:

1. **Redraw the architecture in paper N's own drawio** at
   `assets/<N-paper-id>_arch.drawio`. You MAY copy the XML from M's drawio
   as a starting point, but the file must be independently committed
   under N's paper-id. "Reference M's drawio" in the notes is NOT allowed
   as a substitute.
2. **Independently cite the source code of record** — even if M already did.
   Give file:line citations into the actual modeling file (sglang / vllm /
   HF transformers / HF Model Card `modeling_*.py`) in paper N's notes §1.
   Do not re-use M's citations by reference.
3. **Explicitly call out the equality**. In notes §1, state: "per-field
   comparison with M's config → byte-identical" or "same modeling class
   `FooForCausalLM` used, diff is only in X". Make the equality a
   **documented fact in N's own notes**, not an implicit redirection.
4. **Diff-only pages are additive, not substitutive**. If the release is a
   quantization / post-training / orchestration update (K2.5 → K2.6 pattern),
   produce BOTH the full architecture pages (redrawn) AND the delta pages
   (what's new). Never ship only the delta pages.

**Shortcut phrases that MUST NOT appear in notes** as the sole replacement
for architecture content:

- ❌ "架构不画了——与 X 完全相同, 直接引用 [X notes 互动图]"
- ❌ "see X notes §Y for the architecture"
- ❌ "same as X, not repeated here"
- ❌ "refer to the predecessor paper for model details"

**Self-audit at plan time**: before writing notes, scan your outline for
any sentence of the shape "X is the same as Y, see Y". Every such sentence
is a偷懒 trigger and must be replaced with "X is the same as Y, demonstrated
here by [concrete per-field table / code walkthrough / drawio redraw]".

#### Source priority (per category)

Before drafting the architecture diagram / algorithm table / runtime
pipeline, locate and read code from ONE (or more) of these source tiers,
picking whichever applies to the paper's category:

#### Source priority by paper category

| Category | Primary code source | Where to look |
|----------|--------------------|-----------------|
| `llm` (model arch) | Official repo → HF `transformers` `modeling_<name>.py` + `configuration_<name>.py` → HF Model Card `modeling_xxx.py` → vLLM/SGLang `model_executor/models/<name>.py` | `pip show transformers` / `https://huggingface.co/<org>/<model>/raw/main/modeling_<name>.py` |
| `kernel` | Paper repo (Triton / CUDA / HIP / CK / Cutlass) → FlashAttention / ThunderKittens / HipKittens / cuDNN-style references | `git clone` paper repo; `find` for kernel file matching the op |
| `framework` | Runtime repo (Mirage MPK / vLLM / SGLang / TRT-LLM / DeepSpeed-Inference) — focus on scheduler, task graph generator, KV cache manager | `git clone` + `rg <scheduler class name>` |
| `algorithm` (training / RL / distillation) | Official training repo → OpenRLHF / TRL / verl / ColossalAI reference implementation | paper `Code` link → HF Trainer callback → `grep` for loss function |
| `agent` | Agent framework repo (the paper's own, OR LangGraph / AutoGen / CrewAI / OpenAI-agents-python reference) | `git clone` agent repo; trace planner / tool-call loop |
| `cluster` | NCCL / RCCL / IB verbs / Ultra Ethernet patches; open-source congestion control implementations | Linux kernel tree / NCCL main / RDMA-core |
| `hardware` | AMDGPU / Nouveau / open-source driver patches; ISA disassembler outputs; CK / rocBLAS kernels emitting the new ISA | amdgpu kernel tree / LLVM backend / open-source compiler commits |
| `code` (repo / PR / issue) | The repo itself IS the primary source — SKILL.md Phase 1 already clones it. §9 / §10 of `deep-code.md` is where the walkthrough lives. | N/A — already done |

#### When the paper's own code is not public

The rule is **not** "paper repo must exist" — it's "some relevant code must
be inspected if any exists". Common cases:

1. **Paper code not released, but built on open-source runtime** (e.g. Fleet
   2604.15379 extends Mirage MPK, code not released but Mirage IS public) →
   MUST cross-reference the **runtime / predecessor** code to verify
   baseline behavior the paper claims to improve on.
2. **Paper code not released, but compares against open-source baselines**
   (e.g. "we beat vLLM 0.17.2") → MUST inspect the baseline's source to
   confirm the authors measured it in a fair configuration.
3. **Only the idea is published, zero related open code** (very rare for
   infra papers) → mark honestly as `[实现未公开，无可对照代码]` AND
   state explicitly which aspects would have been code-verified if
   available — this is a **skill-stale** signal that a future reader
   should revisit when code lands.

Do NOT silently skip this step. If you choose tier 3 above, you MUST
print the justification in the notes, not omit it.

#### Procedure (same for all categories)

1. **Find the repo** — use `gh search repos`, paper's URL, HF Model Card,
   or the "Code" arXiv field.
2. **Clone shallow** — `git clone --depth 1 <url> /tmp/<name>` (for
   large repos, use `--filter=blob:none` and check out only the files
   you need).
3. **Walk the critical path** — don't read everything. Identify:
   - For `llm`: `forward()` of the model + `config.json`
   - For `kernel`: the hot-path kernel function + launch config
   - For `framework`: the scheduler main loop + memory manager
   - For `algorithm`: the loss function + data loader
   - For `agent`: the agent main loop + tool call wrapper
   - For `cluster`: the collective implementation + transport layer
   - For `hardware`: the driver IOCTL + compiler backend pass
4. **Cross-check paper vs code** — when they conflict, **trust the code**
   and call out the discrepancy in the notes. Discrepancies are the
   highest-value output of this phase.
5. **Extract hyper-parameters / config** — put real numbers (from `config.json`,
   Makefile flags, kernel launch params, etc.) on the diagram / table.

See `deep-llm.md` §1a for the most detailed template; other deep-*.md guides
inherit the same discipline via short reminder blocks.

During the deep read, identify **3-5 important figures** from the paper
(architecture diagrams, result charts, key illustrations). Record their
URLs or descriptions for Phase 6.

### ✅ Phase 5 End-of-Phase Checklist

- [ ] You **read the category-specific `deep-*.md` guide via the Read tool**
      (don't skip — it has critical category-specific requirements you can't
      guess from memory).
- [ ] **Every numbered section in the guide has a written analysis** in
      the notes file. The completeness checker requires ≥5 numbered
      `### N. Title` sections under `## Deep Analysis`. Realistic targets:
      LLM=10, Framework=11, Hardware=12, Code=20, Algorithm=9, Kernel=11,
      Cluster=10, Agent=9. Counting them is the test.
- [ ] **Code Cross-reference done — UNIVERSAL rule, all categories** (not
      just `llm` / `code` / `framework` / `kernel`): if ANY relevant code
      is publicly available — the paper's own repo, an open-source
      predecessor/runtime it builds on (e.g. Mirage MPK for a persistent-
      kernel framework paper), or a baseline it compares against — you
      MUST have cloned/fetched and walked through the relevant critical
      path (model `forward()`, kernel launch, scheduler loop, loss
      function, agent runtime, collective impl, driver IOCTL — picked
      by category). See SKILL.md "Paper × Implementation Cross-reference
      — UNIVERSAL RULE" for the full source-priority table.
      If truly zero relevant code exists (rare), you MUST explicitly
      mark `[实现未公开，无可对照代码]` + list what you would have
      verified if code existed. Silent skipping is a documented
      failure mode.
- [ ] **drawio file created** at `assets/{paper-id}_*.drawio` for any
      paper in `{llm, kernel, framework, hardware, cluster, code, agent}`
      categories. ASCII box art is forbidden for architecture/topology
      diagrams. The completeness checker will fail if drawio is missing.
- [ ] You used `[论文未披露]` markers honestly for fields the paper does
      not disclose — NOT to skip work you could have done.

---

## Phase 6: Save Images

Download important figures identified during the deep read.

### ⚠️ MANDATORY Quality Gate (no exceptions)

You MUST treat figure extraction as a **closed-loop process**:
**extract → count → visually verify → on-failure: debug & retry**.

Do NOT proceed to Phase 7 until ALL of the following are true:

1. The number of extracted PNG files is **non-zero** AND matches the
   number of important figures you planned to cite in the deep read
   (off by ≤1 is acceptable for sub-figure pages).
2. You have called the `Read` tool on **EVERY** extracted PNG and
   visually confirmed: (a) the figure caption is included, (b) the
   actual figure content (chart / diagram / screenshot) is fully
   visible, (c) no large body-text paragraphs leaked in, (d) no
   important sub-figure or row prompt was cut off.
3. If any image fails verification, you MUST debug the extraction
   (caption regex mismatch? body-vs-label misclassification?
   page layout edge case?) and re-extract — NOT silently fall back to
   crude page cropping.

**Failure mode to avoid (real incident, Seedance 2.0 / 2604.14148)**:
- Tool returned 0 figures silently because the paper used
  `Figure N <text>` instead of `Figure N: <text>` (no colon).
- Agent fell back to `page.get_pixmap(clip=fitz.Rect(...))` with
  hand-guessed page coordinates → all 4 figures were either
  garbage-cropped or completely missing the actual content.
- User had to catch this manually. Never again.

**Hard rule**: if the smart extractor produces 0 figures on a paper
that visibly has figures (check `page.get_text('blocks')` for
`Figure ` prefixes), you are forbidden from falling back to manual
`get_pixmap` crops without first:
  (a) printing a diagnostic showing which captions exist on which
      pages and why each was rejected by the extractor;
  (b) **fixing the upstream tool** so the same failure can't recur on
      the next paper. The fix MUST be committed alongside the notes.

### Three-Tier Extraction Strategy

Try each tier in order. Stop at the first one that succeeds AND
passes the Quality Gate above.

#### Tier 1: arXiv HTML images (preferred for arXiv papers)

```bash
mkdir -p ~/.cursor/paper-db/images/{paper-id}
# Check if HTML version exists (returns 200 vs 404)
curl -sI "https://arxiv.org/html/{arxiv-id}v{N}" | head -1
# If 200: find image filenames
curl -sL "https://arxiv.org/html/{arxiv-id}v{N}" | grep -oP 'src="([^"]+\.png)"'
# Download — x1.png may be a title icon; Figure 1 is usually x2.png
# Verify each download: file *.png (must say "PNG image data", not "HTML document")
curl -sL -o fig1.png "https://arxiv.org/html/{arxiv-id}v{N}/x2.png"
file fig1.png  # MUST be "PNG image data", reject "HTML document"
```

**Common pitfalls with Tier 1:**
- Many arXiv papers return 404 for HTML — fall through to Tier 2
- `x1.png` is often a tiny title icon (26x26), not Figure 1
- Some images are in `figures/` subdirectory, not `x{N}.png`
- Always `file` check every download — HTML error pages get saved as .png

#### Tier 2: Smart PDF figure extraction (primary for non-HTML papers)

Use the `extract_figures_from_pdf()` function from
`~/.cursor/paper-db/tools/extract_figures.py`. This is the **recommended
method for all PDF-based papers** including arXiv papers without HTML
versions, local PDFs, and vendor whitepapers.

```python
# Download PDF if needed
# curl -sL -o /tmp/paper.pdf "https://arxiv.org/pdf/{arxiv-id}"

import sys
sys.path.insert(0, os.path.expanduser("~/.cursor/paper-db/tools"))
from extract_figures import extract_figures_from_pdf

results = extract_figures_from_pdf(
    pdf_path="/tmp/paper.pdf",
    out_dir=os.path.expanduser(f"~/.cursor/paper-db/images/{paper_id}"),
    zoom=3,       # 3x resolution (default)
    padding=8     # 8pt padding around figure (default)
)
for r in results:
    print(f"Fig {r['fig_num']} (p{r['page']}): {r['width']}x{r['height']}, "
          f"{r['size_kb']}KB")
```

Or from shell:

```bash
curl -sL -o /tmp/paper.pdf "https://arxiv.org/pdf/{arxiv-id}"
python3 ~/.cursor/paper-db/tools/extract_figures.py /tmp/paper.pdf \
    ~/.cursor/paper-db/images/{paper-id}
```

**MANDATORY: Inspect the printed summary AND the diagnostic exit code.**
The tool now prints `EXTRACTION SUMMARY: N figures extracted, M captions
detected on K pages` at the end. If `N == 0` but `M > 0`, the tool failed
silently — STOP and debug. If `N` is significantly less than the figures
you saw in the paper, also debug. Do NOT proceed.

After running the tool, **immediately Read each output PNG**:

```python
# Pseudocode for what you must do — actually call the Read tool
import os
out_dir = os.path.expanduser("~/.cursor/paper-db/images/{paper-id}")
for png in sorted(os.listdir(out_dir)):
    if png.endswith(".png"):
        # Read tool call here, look at the image, confirm it shows
        # the figure content + caption with no body-text leak
        pass
```

If any image is wrong (truncated, missing caption, contains body text,
too tall with whitespace, etc.) — fix the extractor and re-run. Do not
ship the notes with bad figures.

**How it works (three-layer detection):**
1. **Classify text blocks** into body (>80 chars), caption (`Figure N:`),
   and label (short annotations like axis labels, sub-figure titles)
2. **Sandwich zone**: For each caption, find the body text blocks
   immediately above and below — these are hard boundaries that MUST NOT
   be crossed (no body text ever leaks into the figure)
3. **Tight crop**: Within the sandwich zone, compute the bounding box of
   all vector drawings + text labels + raster images + the caption itself
4. **Render** the tight bounding box + padding at 3x zoom

This handles vector-graphic PDFs (LaTeX academic papers), embedded raster
images (hardware whitepapers), and mixed content.

#### Tier 3: Fallback

If both Tier 1 and Tier 2 fail (e.g., blog with JS-rendered charts,
DRM-protected PDF), note the figure description in text and skip the
download:

```markdown
### Figure N: {title}
[图片未提取 — 见原文 Page X, Figure Y]
**What it shows**: ...
```

### Image naming and referencing

Output files are named `fig{N}.png` (matching the paper's Figure number).
In the notes file, reference them as:

```markdown
![Figure N: caption](../images/{paper-id}/figN.png)
```

For hardware whitepapers, also prioritize:
1. Chip block diagram / die overview
2. CU/SM internal structure diagram
3. Memory hierarchy diagram
4. Interconnect topology diagram

### ✅ Phase 6 End-of-Phase Checklist (HARD GATE)

- [ ] `~/.cursor/paper-db/images/{paper-id}/` contains ≥ 2 PNGs (most
      papers warrant 3-5).
- [ ] Each PNG is > 5 KB (smaller = HTML error page or empty crop).
- [ ] You called the **`Read` tool on EVERY PNG** and visually confirmed:
      caption included + figure content fully visible + no body-text leak +
      no sub-figure cut off.
- [ ] If `extract_figures.py` printed `SILENT FAILURE DETECTED`, you
      fixed the upstream tool (regex, classification logic, etc.) and
      committed the fix — NOT bypassed it with manual `get_pixmap`.

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

### 🚨 Pre-Save Self-Check (MANDATORY, added 2026-04-21I)

**Before writing the draft to `notes/{id}.md`**, the agent MUST do an
explicit in-context scan of its own draft for skill-leakage patterns.
Post-hoc `check_paper_completeness.py` scanning catches symptoms but
does NOT prevent generation-time leaks; this pre-save step closes the
loop at the authoring phase itself.

**Procedure** — before the file write:

1. Re-read your full draft in one pass.
2. For every paragraph / section title / blockquote, apply the
   universal **swap-paper-name test**:
   > "If I replace this paper's name with 'X', does the paragraph still
   > make sense for any other paper?"
   If yes, it's skill-leakage (universal meta that belongs in the skill,
   not this paper's notes).
3. Specifically scan for these generation-time leak patterns:

   | Leak pattern | Example | Fix |
   |---|---|---|
   | Rule-tag in section title | `### 作者证明 (SJM)`, `## Logic Flow Reconstruction (Phase 2b)` | Drop the `(...)` suffix: `### 作者证明`, `## Logic Flow Reconstruction` |
   | Skill/rule citation in prose | "本节依照 SKILL.md Phase 2b 的...", "per deep-framework.md §12" | Remove the citation; keep the content |
   | Inheritance callout (mimicking skill style) | "> 🚨 **Inherited universal rule** ..." | Delete entire callout |
   | Trigger/classification meta block | "> 📌 **触发说明**: X 归类为 Y 但属于 Z 子集..." | Delete entire block; classification is in papers.json |
   | Tool-choice rationale | "本笔记 2026-04-21 改用 Mermaid...", "Mermaid 的优势..." | Delete; skill already documents this |
   | Deep-*.md principle restatement | "图里有的不写字、字里有的不贴码", "文字说明 = 架构图的 caption" | Delete |
   | Self-referential format description | "本节只写 X / 不讨论 Y", "本笔记采用 Z 结构" | Delete |
   | Abbreviation that maps to a skill rule | `(SJM)`, `(Phase 2b)`, `(hybrid disambiguation)` | Drop parenthetical |

4. If ANY match found → fix in-draft BEFORE save.
5. As a belt-and-suspenders, after save run:
   ```bash
   python3 ~/.cursor/paper-db/tools/check_paper_completeness.py {paper_id} --strict
   ```
   If the scanner still catches something, loop back to step 1 on the
   residual content — the scanner may have a phrase your self-check
   missed (which means your self-check criteria need extension).

**Why this exists**: generation-time vs post-hoc debate. Scanner alone
= whack-a-mole. Agent-side self-check = root-cause closure. See
incidents.md 2026-04-21I.

### ✅ Phase 7 End-of-Phase Checklist

- [ ] Notes file contains all required top-level sections: `## Core
      Contribution / ## Summary / ## Key Findings / ## Limitations /
      ## Infrastructure Impact / ## Deep Analysis`.
- [ ] `## Deep Analysis` contains ≥ 5 numbered `### N. Title` sub-sections
      (the completeness checker counts these).
- [ ] Every `### Figure N:` heading in notes has a matching image link
      (no orphaned figure descriptions).
- [ ] papers.json `read_depth = "deep"` and `open_questions` non-empty.
- [ ] **Pre-Save Self-Check executed**: draft scanned for skill-leakage
      patterns (swap-paper-name test + 8 leak-pattern table above)
      BEFORE file write. Residual scan via
      `check_paper_completeness.py --strict` passes with 0 leak hits.

---

## Phase 8: Synthesis Update (Automatic)

After saving, **read the paper-synthesis skill** at
`~/.cursor/skills/paper-synthesis/SKILL.md` and execute its
**"Incremental Update"** procedure. This:

1. Reads `papers.json` to find existing papers related to the new one
2. Updates `related_paper_ids` bidirectionally
3. Identifies new cross-category connections
4. Reports a brief synthesis update to the user

### ✅ Phase 8 End-of-Phase Checklist

- [ ] You **actually iterated through every existing paper** in
      papers.json (not just guessed "this is similar to X"). Use
      shared-tag count + category overlap as the relatedness signal.
- [ ] If you found related papers, you updated **bidirectionally** —
      both the new paper's `related_paper_ids` AND each related paper's
      `related_paper_ids`.
- [ ] If `related_paper_ids` is empty after this, you justified WHY
      (e.g. "first paper in DB on this topic"); the completeness
      checker will warn otherwise.

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

7. **Nav bar consistency enforcement** — after `sync.sh` completes,
   run the nav unifier to ensure ALL HTML pages (index, papers, readers,
   tools, knowledge, knowledge-graph) have the canonical site-level nav:

```bash
cd /apps/feiyue/upstream/zhaifeiyue.github.io && python3 << 'NAVFIX'
import os, re

CANONICAL_NAV = '<nav>\n  <a class="logo" href="/">Feiyue Zhai</a>\n  <a href="/">Papers</a>\n  <a href="/knowledge/">Knowledge</a>\n  <a href="/knowledge-graph.html">Graph</a>\n  <a href="/tools/">Tools</a>\n  <a href="https://github.com/ZhaiFeiyue" target="_blank">GitHub</a>\n</nav>'

SITE_NAV_CSS = '\n.site-nav{background:#1b2a4a;color:#fff;padding:0 24px;height:48px;display:flex;align-items:center;gap:24px;position:sticky;top:0;z-index:9999;box-shadow:0 2px 12px rgba(0,0,0,0.15);font-family:system-ui,-apple-system,sans-serif}.site-nav .logo{font-weight:700;font-size:1rem;color:#fff;text-decoration:none}.site-nav a{color:#cbd5e1;font-size:.85rem;font-weight:500;text-decoration:none}.site-nav a:hover{color:#fff}\n'
SITE_NAV_BAR = '<div class="site-nav">\n  <a class="logo" href="/">Feiyue Zhai</a>\n  <a href="/">Papers</a>\n  <a href="/knowledge/">Knowledge</a>\n  <a href="/knowledge-graph.html">Graph</a>\n  <a href="/tools/">Tools</a>\n  <a href="https://github.com/ZhaiFeiyue" target="_blank">GitHub</a>\n</div>'

nav_pat = re.compile(r'<nav[\s\S]*?</nav>', re.MULTILINE)
fixed = 0
for root, dirs, files in os.walk('.'):
    for fn in files:
        if not fn.endswith('.html'): continue
        path = os.path.join(root, fn)
        with open(path) as f:
            content = f.read()
        if '/knowledge/' in content and '/knowledge-graph.html' in content:
            # Check site-level nav is canonical
            m = nav_pat.search(content)
            if m and 'Feiyue Zhai' in m.group(0):
                canon = re.sub(r'\s+', ' ', CANONICAL_NAV.strip())
                cur = re.sub(r'\s+', ' ', m.group(0).strip())
                if canon != cur:
                    content = content.replace(m.group(0), CANONICAL_NAV)
                    with open(path, 'w') as f: f.write(content)
                    fixed += 1
            continue
        # Page missing Knowledge or Graph — inject site nav
        body_m = re.search(r'<body[^>]*>', content)
        if not body_m: continue
        head_close = content.find('</head>')
        if head_close > 0 and '.site-nav' not in content:
            content = content[:head_close] + f'<style>{SITE_NAV_CSS}</style>' + content[head_close:]
            body_m = re.search(r'<body[^>]*>', content)
        content = content[:body_m.end()] + '\n' + SITE_NAV_BAR + '\n' + content[body_m.end():]
        with open(path, 'w') as f: f.write(content)
        fixed += 1
print(f'Nav unifier: {fixed} pages fixed')
NAVFIX
```

   This ensures every page — including reader HTMLs, tools, knowledge
   articles, and the knowledge graph — has the full site-level nav bar
   with **Papers | Knowledge | Graph | Tools | GitHub** links.

   **Why this rule exists** (real incident, 2026-04-16): knowledge-graph.html,
   tools/index.html, knowledge/index.html, and 26 reader pages were missing
   Knowledge or Graph links. The user had to catch this manually. The nav
   unifier script now runs as a mandatory post-sync step.

### ✅ Phase 9 End-of-Phase Checklist

- [ ] `sync.sh` finished with exit 0 AND printed
      `Wrote papers/{paper-id}.html`.
- [ ] `papers/{paper-id}.html` exists and is > 5 KB (smaller = template
      glue with no actual content).
- [ ] If the paper has a drawio file, you visually confirmed the
      `{{drawio:...}}` directive renders in the HTML (search the file
      for `mxfile` to confirm XML is inlined).
- [ ] **Nav unifier ran** — the post-sync nav consistency script printed
      its count. ALL HTML pages now contain `/knowledge/` AND
      `/knowledge-graph.html` links. If the script reports > 0 fixes,
      the fixed pages must be included in the Phase 11 git commit.

---

## Phase 10: Generate Per-Paper HTML Reader (Automatic, MANDATORY)

> **⚠️ This phase is NOT optional.** Skipping it (e.g. "this paper doesn't
> need it") was the second-largest failure of the Seedance 2.0 incident.
> Every paper deep-read produces a cognitive HTML reader, full stop. If
> the paper genuinely doesn't warrant 11 cognitive sections (rare —
> applies only to very short notes or pure model-card releases), use the
> minimal-template fallback described below, but ALWAYS produce
> `readers/{paper-id}-reader.html`.

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

### 🔗 Paper ↔ Reader bidirectional link (MANDATORY)

> Real incident, 2026-04-21C: `papers/{id}.html` and `readers/{id}-reader.html`
> were two separate HTML artifacts with **zero cross-links** between them.
> User had no way to discover the reader from the paper page and caught it:
> "为什么会重新生成另一个 html 原因是什么?如果需要做一个新的 html 你需要
> 从 paper 生成的 html 里 link 过去而不是自己偷偷做一个新的".

**Why two HTMLs exist** (must be clear in every agent turn that produces
a reader):

- `papers/{id}.html` — **authoritative notes** (from sync.sh, markdown-
  rendered). Serves as reference, complete, long-form.
- `readers/{id}-reader.html` — **pedagogical reader** (Phase 10, hand-
  crafted). 11-section cognitive ladder with interactive drawio arch,
  worksheet questions, double-column annotations. Serves as tutorial,
  opinionated, skim-friendly.

They complement each other. **Neither replaces the other**, but either
being orphan is a failure.

**Two required links**:

1. **Paper → Reader** (handled by sync.sh automatically):
   sync.sh injects a `📖 导读 Reader ↗` button into the paper header
   whenever `readers/{id}-reader.html` exists on disk. **You do NOT
   hand-code this link**; just make sure the reader file is present
   in `<site>/readers/` before running sync.sh. If sync.sh has not
   been updated to auto-link (check `grep "导读 Reader" sync.sh`),
   the feature is missing and must be re-added — see 2026-04-21C
   incident for patch location.

2. **Reader → Paper** (you must add in reader HTML):
   Every reader must contain `<a href="/papers/{id}.html">← 完整笔记
   (Notes)</a>` in its top bar, so users can jump back to the full
   notes. Omitting this is a failure.

### 🖼️ drawio embed in reader HTML — use lazy-load pattern (MANDATORY)

> Real incident, 2026-04-21B: reader used `data-mxgraph="{...84KB JSON...}"`
> inline-attribute embed; viewer-static.min.js silently failed to render
> because the attribute size and double-escape interact badly. Fix was
> to use the same **lazy-load pattern** that sync.sh uses for paper
> pages (proven working).

Correct pattern in reader HTML `<body>`:

```html
<!-- Hidden raw XML source (kept as textContent, not attribute) -->
<div id="drawio-xml-{paper-id}-arch"
     class="drawio-xml-source"
     style="display:none !important;" aria-hidden="true">{HTML-escaped drawio XML}</div>

<!-- Empty placeholder; bootstrap JS fills data-mxgraph at runtime -->
<div class="mxgraph mxg-lazy"
     data-xml-source="drawio-xml-{paper-id}-arch"
     data-page="0"
     style="width:100%;aspect-ratio:1500/900;border:1px solid #ddd;
            border-radius:8px;background:#fff;overflow:hidden;"></div>
```

Bootstrap JS at end of `<body>`:

```html
<script defer>
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.mxgraph.mxg-lazy').forEach(function (el) {
    var src = document.getElementById(el.dataset.xmlSource);
    if (!src) return;
    el.setAttribute('data-mxgraph', JSON.stringify({
      highlight: '#0000ff', nav: true, resize: false,
      toolbar: 'zoom layers tags lightbox pages', 'toolbar-position': 'top',
      fit: 1, 'auto-fit': 1, border: 5, 'page-visible': false,
      lightbox: false, edit: '_blank',
      xml: src.textContent.trim(),
      page: parseInt(el.dataset.page || '0', 10)
    }));
    el.classList.remove('mxg-lazy');
  });
});
</script>
<script defer src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
```

**Do NOT** attempt `data-mxgraph="{huge JSON with escaped XML inside}"`.
For drawios larger than ~10KB, HTML attribute escaping breaks silently.

### Minimal-template fallback (for very short release reports)

If the paper is genuinely too thin for the full 11-section reader (fewer
than ~10 pages of analysis-worthy content, pure model card, etc.), you
may produce a slimmer reader, but it MUST still:

- Be a complete standalone HTML file at the canonical path
- Include the hero figure + title + one-line summary (Section 1)
- Include the deep analysis content in a single scrollable view
- Use the same fonts/colors/KaTeX setup from html-reader-guide.md
- Be ≥ 8 KB (smaller = template glue with no actual content)

Even in fallback mode, **explicitly tell the user in your final report**
that you used the minimal template and why.

### ✅ Phase 10 End-of-Phase Checklist (HARD GATE)

- [ ] `~/.cursor/paper-db/readers/{paper-id}-reader.html` EXISTS.
- [ ] File size ≥ 8 KB.
- [ ] You opened the file (Read tool) and confirmed it has actual paper
      content, not just template scaffolding.
- [ ] If you used the minimal fallback, you justified it in the
      Phase 10 status line.

---

## Phase 11: Publish to GitHub Pages (Automatic)

Phase 9 already runs `sync.sh`. Now commit and push:

```bash
cd /apps/feiyue/upstream/zhaifeiyue.github.io
# DO NOT use `git add -A` — it pulls in unrelated work-in-progress changes
# from other paper reads. Stage only this paper's files explicitly.
git add assets/{paper-id}_*.drawio papers/{paper-id}.html index.html
# If you also touched papers from Phase 8 bidirectional updates, add them too:
# git add papers/{related-id}.html
# ALWAYS include nav-fixed files from the Phase 9 nav unifier:
git add knowledge-graph.html
git diff --cached --name-only   # confirm only intended files staged
git commit -m "add: {paper title}"
git push
```

This updates https://zhaifeiyue.github.io/ with the latest paper readings.
The site uses Summary/Details labels, base64 images, no local paths.

### ✅ Phase 11 End-of-Phase Checklist

- [ ] You staged this paper's files PLUS any nav-fixed files from Phase 9
      (verified via `git diff --cached --name-only`).
- [ ] `git commit` succeeded (exit 0).
- [ ] `git push` succeeded (exit 0, output contains `main -> main`).
- [ ] **Post-push nav verification**: confirmed ALL published HTML pages
      contain both `/knowledge/` and `/knowledge-graph.html` in their
      nav bar (the Phase 9 nav unifier should have handled this, but
      verify no files were missed from staging).

---

## 🚨 FINAL GATE — run before reporting to user

After all 11 phases complete, run the completeness checker in **strict** mode:

```bash
python3 ~/.cursor/paper-db/tools/check_paper_completeness.py {paper_id} --strict
```

This validates:

- papers.json has all 17 required fields populated (no empty strings, no
  missing infra_impact layers)
- Notes file has all 6 required top-level sections + ≥ 5 numbered
  Deep Analysis sub-sections + ≥ 2 figure refs that match images on disk
- ≥ 2 PNG figures exist, each > 5 KB
- drawio file exists for arch-bearing categories (llm/kernel/framework/
  hardware/cluster/code/agent)
- Published HTML at `papers/{paper-id}.html` exists
- Cognitive HTML reader at `readers/{paper-id}-reader.html` exists

If exit code is **non-zero**, you MUST fix every BLOCKING FAILURE printed
and re-run before producing the Final Output. Warnings (yellow) should be
addressed when reasonable but do not block.

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
| Code | ... |

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
