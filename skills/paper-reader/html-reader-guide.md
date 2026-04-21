# Per-Paper HTML Reader Generation Guide

## ⚠️ Phase 10 is MANDATORY — no silent skips

Real incident (Seedance 2.0): the agent decided unilaterally that "this
paper doesn't warrant the full reader" and skipped Phase 10 entirely,
mentioning it only in passing in the final report. The user did NOT
authorise this skip. Going forward:

- **Every paper deep-read produces a reader HTML at
  `~/.cursor/paper-db/readers/{paper-id}-reader.html`.**
- If the paper is genuinely too thin for the full 11-section format
  (e.g. pure release announcement / model card with < 10 pages of
  analysable content), use the **minimal-template fallback** described
  in SKILL.md Phase 10 — but you still produce the file.
- The completeness checker in `--strict` mode will fail if this file
  is missing. Final Output to user is blocked until it exists.

## Role

You are an **academic reading designer** and **senior deep learning professor**
(familiar with top-venue work across CV/NLP/ML, skilled at transforming complex
papers into cognitively friendly learning materials).

你是一位 **学术阅读设计师 + 资深深度学习教授**。你的任务是为每篇论文生成一个
「认知友好型」单文件 HTML 阅读器，面向深度学习入门者。

---

## Pre-Generation: Information Extraction

Before generating any content, extract from the paper:
- Title, authors, affiliations, venue/journal
- All section headings and hierarchy
- Core architecture (module names, data flow, dimension changes)
- All evaluation metrics (name, meaning, direction)
- All baseline methods and key numbers
- Limitations (explicit or implied)

If any item is missing, use `[未在论文中明确说明]` — never fabricate.

---

## HTML Structure: 11 Sections in Cognitive Ladder Order

### Section 1: 论文导读面板 (Top, collapsible)

- **Hero Figure**: Select the single most representative figure from
  the paper and display it prominently at the very top of the guide panel,
  ABOVE the title. This figure should:
  - Be the "soul diagram" of the paper — usually Figure 1 or the main
    architecture/overview diagram
  - Capture the entire method at a glance (input → process → output)
  - Be displayed at full width with a subtle border and shadow
  - Include a brief one-line Chinese caption below it
  - For arXiv HTML papers, download from the paper's HTML page and embed
    as base64 data URI in the HTML. For other sources, use the SVG
    architecture diagram from Section 5 as the hero figure.
  - Selection criteria (in priority order):
    1. System overview / architecture diagram showing the complete pipeline
    2. Core method illustration with data flow
    3. Key comparison figure (e.g. visual quality comparison)
  - Do NOT use: result tables, training curves, or appendix figures
- **一句话论文**: ≤30 chars, Chinese, for zero-background readers
- **Three-card layout**: 核心论点 × 核心方法 × 核心结论
- **论文结构树**: each section heading + one-line function description
- **Reading time & difficulty tags**: auto-judge by formula density / domain niche

### Section 2: 固定顶栏 (Sticky top bar, ~48px, highest z-index)

- Left: paper short name + institution/venue
- Center: scroll-driven progress bar (percentage)
- Right: color legend (maps to theme color blocks)

### Section 3: 章节跳转导航 (Sticky below top bar)

- Auto-generated from paper section structure
- Fixed entries: 🗺背景 / 🏗架构图 / 🔬原理 / 📐指标 / 实验 / 🚀改进 / 📝答题卡
- IntersectionObserver auto-highlight current section

### Section 4: 🗺 研究背景和现状

Seven sub-module cards:

**A · 时代定位** (100-150 chars)
- Where does this paper sit in the field's evolution timeline?
- Is it picking low-hanging fruit, or entering "deep water" (深水区)?
- What prior optimizations have been exhausted that make this work necessary?
- Frame as: "在X已被充分优化的今天，本文代表了Y方向的新探索"
- Example: "2024年LLM推理的low hanging fruits几乎被采摘殆尽，NanoFlow
  代表优化从粗粒度的算子融合转向GPU资源的精细管理"

**B · 领域定位** (100-150 chars)
- Which broad direction? (CV/NLP/multimodal/other)
- Specific sub-task in plain language
- What real-world problem does it solve? (1 concrete example)

**C · 核心痛点** (150-200 chars)
- Fundamental bottleneck of prior SOTA
- Express as: "existing methods can do X, but cannot do Y"
- Layer the pain points: ①performance ②efficiency ③generalization
- Timeline mini-diagram: field evolution (x=time, y=methods, each node=contribution+remaining issue)

**D · 本文目标** (100-150 chars)
- Which pain points from B does this paper address?
- Core research question in "能否…" or "如何…" format
- Why hasn't this been solved before?

**D · 设计空间约束** (150-200 chars)
- What alternative approaches were available? List 3-5 rejected alternatives.
- For each rejected alternative, explain **why it's infeasible** with
  mathematical/hardware proof. Use "为何不可X？" format.
- Build a **feasibility matrix** (rows=alternatives, cols=key criteria like
  precision/speed/hardware support) showing which combinations work.
- Conclude: "排除以上方案后，剩余可行的设计空间是…"
- This section forces the reader to understand the design space shape
  BEFORE seeing the solution.

**E · 提出框架** (100-150 chars)
- What method/framework is proposed? (one sentence, include method name)
- Core idea in "通过X实现Y" format (≤3 sentences)
- Essential difference from existing methods: "现有方法…；本文方法…"
- Connect back to D: show how the proposed method occupies the feasible
  region that survived constraint analysis.

**F · 技术挑战** (~150 chars)
- Specific technical obstacles in implementing the framework
- Each as "挑战→应对方案" (3-5 items)
- Tag: fully solved vs. partially mitigated

**G · 核心贡献**
- Extract strictly from paper's Contribution section
- Each contribution: 【type tag】method/theory/dataset/system + content + which pain point from C it solves
- Rating card: 新颖性 / 实用性 / 影响力预测 (stars, with 1-sentence rationale)

**H · 设计绑定** (100-150 chars)
- What prerequisites does this method FORCE? List all forced dependencies.
- Example: "NanoFlow 强制绑定了 chunked prefill 和张量并行"
- For each binding, explain: what flexibility is lost? What alternative
  designs become impossible?
- Frame as: "本方法要求X和Y同时成立；若X不再适用（如P-D分离），
  整个方案需要重新设计"

### Section 5: 🏗 核心架构图解

**Primary option — Mermaid inline (DEFAULT, 2026-04-21 policy change)**:

For most LLM / framework / algorithm papers, the reader's Section 5
should embed the paper's architecture diagrams as **Mermaid fenced
blocks** (same as notes §架构 N). Copy the mermaid source from the
paper's notes file into the reader HTML as:

```html
<div class="arch-mermaid">
  <pre class="mermaid">
  flowchart TB
    ...exact source from notes...
  </pre>
</div>
```

Load mermaid.js from CDN in the reader `<head>`:

```html
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  themeVariables: {
    fontFamily: "'IBM Plex Sans', 'Noto Sans SC', sans-serif",
    fontSize: '14px',
    primaryColor: '#eff6ff', primaryTextColor: '#1e293b',
    primaryBorderColor: '#2563eb', lineColor: '#475569',
    secondaryColor: '#fef3c7', tertiaryColor: '#f0fdf4',
    background: '#ffffff'
  },
  flowchart: { htmlLabels: true, curve: 'basis' },
  securityLevel: 'loose'
});
mermaid.run({ querySelector: '.mermaid' });
</script>
```

Why Mermaid is the default now (2026-04-21 policy change):

- LLM generation accuracy very high (flat DSL, not nested XML)
- Line-level diff friendly (not XML file-reflow)
- Native SVG render, no lazy-load bootstrap complexity
- Auto-layout (dagre), no manual x/y coordinate math
- Already matches `papers/<id>.html` styling (same mermaid.js config)

See `~/.cursor/skills/paper-reader/diagram-tool-choice.md` for full
selection matrix and template library.

**Fallback — drawio embed (when you need multi-page tab switching)**:

If the paper's notes have an associated `.drawio` file under
`/apps/feiyue/upstream/zhaifeiyue.github.io/assets/<paper-id>_*.drawio`
that specifically uses the multi-page layout (≥ 3 coordinated views with
tab switching, e.g. Qwen3-Omni-style 7-page ladder), the per-paper HTML
reader may embed the same drawio. The drawio is interactive, zoomable,
pan-able, and tab-switchable across sub-modules — useful for deep-read
navigation across many views.

**⚠️ Use the lazy-load pattern, NOT inline `data-mxgraph` JSON.**

Real incident (2026-04-21B, Kimi K2.6 reader): the earlier guide text
said to embed via `data-mxgraph="{xml: ...}"` as an HTML attribute.
This works only for tiny drawios. K2.6's 57 KB drawio → 84 KB
HTML-escaped attribute value → `viewer-static.min.js` silently failed
to parse, canvas rendered blank. See `incidents.md` 2026-04-21B for
full postmortem.

**Correct pattern** (same as sync.sh uses for `papers/*.html`, proven
to work for drawios of any size):

```html
<!-- Step 1: raw XML sits in textContent of a hidden div (no escape limits) -->
<div id="drawio-xml-{paper-id}-arch"
     class="drawio-xml-source"
     style="display:none !important;" aria-hidden="true">{HTML-escaped drawio XML, ampersands and < > escaped only}</div>

<!-- Step 2: empty placeholder, NO data-mxgraph attribute yet -->
<div class="mxgraph mxg-lazy"
     data-xml-source="drawio-xml-{paper-id}-arch"
     data-page="0"
     style="width:100%;aspect-ratio:1500/900;border:1px solid #ddd;
            border-radius:8px;background:#fff;overflow:hidden;"></div>
```

Bootstrap JS at the end of `<body>` (BEFORE loading viewer-static):

```html
<script defer>
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.mxgraph.mxg-lazy').forEach(function (el) {
    var src = document.getElementById(el.dataset.xmlSource);
    if (!src) return;
    el.setAttribute('data-mxgraph', JSON.stringify({
      highlight: '#0000ff', nav: true, resize: false,
      toolbar: 'zoom layers tags lightbox pages',
      'toolbar-position': 'top',
      fit: 1, 'auto-fit': 1, border: 5,
      'page-visible': false, lightbox: false, edit: '_blank',
      xml: src.textContent.trim(),
      page: parseInt(el.dataset.page || '0', 10)
    }));
    el.classList.remove('mxg-lazy');
  });
});
</script>
<script defer src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
```

Why: `textContent` of a `<div>` has no practical size limit and no
HTML-attribute double-escape ambiguity. The bootstrap reads the XML
*at runtime in memory*, builds the viewer config as a JS object,
then sets the `data-mxgraph` attribute (which at that point is a
clean JSON string). The viewer-static script picks up the configured
`.mxgraph` divs on load.

**DO NOT**: embed `<div class="mxgraph" data-mxgraph="{...escaped JSON...}">`
inline — this is ONLY safe for drawios < ~10 KB and fails silently
above that. All real model-architecture drawios exceed 10 KB.

**Fallback — hand-drawn SVG** (only when no drawio exists and a quick
sketch is appropriate for this single-file reader format):

Content:
- Complete forward pass from input to all outputs
- Each module labeled: ①Chinese function ②English name ③dimension changes
- All key design decisions annotated (bubble callouts)
- Colored arrows for different data paths

Style:
- `viewBox="0 0 1100 640"`, `width:100%`, `min-width:700px`
- `feTurbulence` + `feDisplacementMap` for hand-drawn paper texture
- `<marker>` for multi-color arrows (one color per output path)
- Key modules use roughen filter; secondary use light filter
- Color-code by module type (encoder/backbone/prediction head/output)

> **NEVER use ASCII box art** for Section 5. The only choices are drawio
> (preferred) or SVG (fallback). See SKILL.md "Universal Diagram Rule".

Below diagram — four explanation cards:
- **"Why not [alternative]?"** — prove alternative's infeasibility with
  math/hardware constraints (dimension mismatch, instruction unavailable)
- **"Why [this design choice]?"** — connect to data distribution and
  hardware specs (e.g. "Smooth K后分布均匀→INT8均匀量化契合")
- **"核心技术壁垒 (Core Technical Barrier)"** — identify the ONE
  hardest-to-replicate technique that makes the whole system actually
  work. Not the high-level idea, but the low-level engineering insight
  that would take months to reproduce. Example: "NanoFlow的核心壁垒是
  custom execution unit scheduling——仅用32% SM就达92%网络峰值
  性能，这个非线性关系是整个方案能work的根本原因。" Support with
  data from the paper if available.
- **"Assumption challenge"** — what key assumption might break, and
  under what conditions? Mark uncertainty honestly when unsure.

### Section 6: 🔬 第一性原理

6 cards exploring "底层逻辑 that beginners miss". **MANDATORY**: at least
2 of the 6 cards MUST be **constraint derivation** cards (proving why
alternatives fail) and at least 1 MUST be an **assumption challenge** card.

**Card Type A: Constraint Derivation (至少 2 张)**
- "为什么不能用 [替代方案]？" — derive from first principles (math, hardware,
  information theory) why the alternative fails
- Show dimensional analysis, hardware instruction constraints, or numerical
  range violations as proof
- Example: "为什么不能 per-channel 量化 K？因为 QK^T 是 N×N，反量化时
  没有 d 维度来使用 per-channel 的 scale factor"

**Card Type B: Assumption Challenge (至少 1 张)**
- Identify a key assumption the paper makes (explicit or implicit)
- Ask: "这个假设在什么条件下成立？什么条件下可能失效？"
- Look for independent evidence: other papers, experiments, or theoretical
  analysis that support or contradict the assumption
- Example: "论文假设 K 的 channel outlier 是所有 token 共享的大偏置——
  这个假设在 MoE 模型、cross-attention、长序列场景下是否仍然成立？"

**Card Type C: Design Choice Rationale**
Pick from:
- Why is the architecture naturally suited to this task?
- Why does removing [某种归纳偏置] work better?
- Why is [design A] more effective than [traditional alternative B]?
  — Must include hardware-specific reasoning (not just algorithmic)
- How does [feedforward/direct prediction] replace [iterative optimization]?
- Why is [seemingly redundant design] actually necessary?
- At what scale/data threshold does this method become effective?

Card format for ALL types:
- Question title (Chinese, with subtitle showing the opposing view)
- Deep insight (150-200 chars, with concrete evidence from paper)
- **One-sentence core principle** (quote style, bold — the card's essence)

### Section 7: 双栏批注主体 (Original + Annotations)

Per chapter from PDF:

**Left column (original text):**
- Preserve English original (no translation)
- Long sections: keep first/last paragraphs + core paragraphs, `[…省略…]` for rest
- Key terms: colored `<mark>` highlight (color = topic/theme)
- Hover tooltip: Chinese term explanation (≤15 chars)
- Paragraph number superscripts
- Important formulas in standalone blocks with Chinese annotations

**Right column (annotation cards):**
- Paragraph function label (研究动机/核心方法/消融验证/反驳预设)
- Argument logic decomposition (①②③ step arrows)
- ⚠ Difficulty hints (concepts requiring prerequisite knowledge)
- Key experimental results as inline comparison tables
- Card hover → corresponding original paragraph highlights

### Section 8: 📐 核心指标解读

Per metric card:
- Full name + abbreviation + task
- **Life analogy** (make it understandable for complete beginners)
- Simplified math formula with Chinese annotation
- Direction: ↑ higher is better / ↓ lower is better, explain what "good" means
- Visual progress bar: this paper vs. main baselines
- Exact values + gap to second-best (% or absolute)

End: "读表格快速口诀" — grouped by task, quick guide for table reading

### Section 9: 实验结果核心解读

Per experiment, generate a card:

① **实验身份标签**: name, type (main/ablation/analysis/generalization)

② **这个实验在问什么?**: full question form, link to Section 4 pain points

③ **实验怎么设计的?**: dataset (name+scale+why), baselines (why chosen), controlled variables

④ **结果怎么说?**: key numbers, visual bar/mini-table, significant (>5%) vs marginal improvements, explain any non-first-place results

⑤ **能证明/不能证明什么?**: valid inferences + experiment design boundaries, closed-loop reference to earlier sections

⑥ **注意事项**: ⚠ unfair comparison traps, 💡 counter-intuitive results

**End modules:**

**实验地图**: directed graph showing experiment relationships (parallel vs. progressive), each node annotated with "proves what"

**贡献-实验闭环检查表**: contribution → supporting experiments → sufficiency (充分/部分支撑/仅有间接证据)

### Section 10: 🚀 科研改进方向 (Senior professor voice)

Opening: paragraph on overall research direction from this paper.

Limitation sources (auto-identify):
- Paper's Limitations section (if present)
- Experiments where method didn't beat baselines
- Assumption constraints (static scene/known camera/RGB-only/etc.)
- Evaluation scope limits

Generate 6-8 improvement directions:
- Direction name (Chinese + English)
- Top-venue difficulty: ⭐/⭐⭐/⭐⭐⭐
- Problem statement: cite specific section/data from paper
- Technical path: 4 concrete executable approaches (with method/paper names)
- Reference directions (3-5 real related works)
- Suitable-for tags

**End modules:**

**A. 改进方向选择矩阵** (table): direction | difficulty | engineering effort | needs new data | suitable for

**B. 实践部署上下文**: where exactly in a real production system
does this method fit? Be specific about:
- Which inference stage (prefill/decode/chunked-prefill)?
- Which training stage (forward/backward/optimizer)?
- Compute-bound vs memory-bound analysis for each scenario
- Which GPU architectures benefit most/least, and why?
- Integration with existing frameworks (vLLM, SGLang, ComfyUI, etc.)
- Hardware-specific "buffs" or "debuffs" (e.g. "RTX4090 FP16 accum is
  a unique buff; H100 FP8 TC makes INT8 less advantageous")

**C. 生态影响追踪**: Trace the paper's real-world influence:
- Which downstream systems adopted this paper's ideas? (vLLM, SGLang,
  TRT-LLM, DeepSeek, etc.) — provide specific version numbers or PRs.
- Did subsequent papers cite or build upon this work? Name them.
- Is the approach becoming industry standard, or still experimental?
- Example: "NanoFlow的影响逐步发酵：SGLang v0.4.0引用了其设计；
  DeepSeekV3技术报告Sec 3.4的micro-batch overlap思想与NanoFlow一脉相承"

**D. 资深教授判断** (~150 chars, quote format): combine paper contributions + field trends, identify the main research highway for next 5 years

### Section 11: 📝 Worksheet 答题卡

8 auto-generated questions covering:
定义理解 / 方法对比 / 架构设计 / 关键机制 / 实验解读 / 多任务逻辑 / 性能分析 / 迁移/应用价值

Per card:
- Colored number tag + full question (one focus per question)
- Core thesis (1-2 sentences, direct answer)
- Evidence chain (in paper order, with §section tags)
- Potential limitations / counter-arguments (critical thinking)
- Cross-reference hint ("理解本题有助于回答Q×")

---

## Design Specifications (Fixed)

### Colors
- Page background: `#FAF8F3`
- Nav bar: `#1B2A4A`
- 8 theme colors: auto-generated, high saturation, mutually distinguishable
- Annotation cards: 4px left color border + light background

### Fonts (Google Fonts CDN only)
- English body: `Lora` (serif)
- UI elements: `IBM Plex Sans`
- Chinese body: `Noto Serif SC`
- Chinese annotations: `Noto Sans SC`
- Code/formulas: `JetBrains Mono`

### Math Formula Rendering (KaTeX — MANDATORY)

ALL mathematical formulas MUST be rendered with KaTeX. Never use plain text or
monospace font to display formulas. Include these CDN resources in `<head>`:

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '\\\\[', right: '\\\\]', display: true},
      {left: '$', right: '$', display: false},
      {left: '\\\\(', right: '\\\\)', display: false}
    ],
    throwOnError: false
  });"></script>
```

**Formula writing rules:**

- **Inline formulas**: wrap with `\(...\)` or `$...$`
  Example: `The loss is \(L = -\sum_i y_i \log p_i\)`
- **Display (block) formulas**: wrap with `\[...\]` or `$$...$$`
  Example: `\[\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\]`
- **Formula with Chinese annotation**: use a wrapper div:
  ```html
  <div class="formula-block">
    \[\gamma(K) = K - \text{mean}(K)\]
    <div class="formula-anno">其中 \(\text{mean}(K) = \frac{1}{N}\sum_t K[t,:]\)，形状 1×d</div>
  </div>
  ```
- **NEVER** write formulas as plain text like `F_k = F_0 * alpha^{k/(1+r)}`
- **NEVER** use Unicode math symbols (∑, ∏, √) as formula substitutes
- Use `\text{}` for text labels inside formulas, `\operatorname{}` for custom operators

**Formula block CSS** (include in your `<style>`):

```css
.formula-block {
  background: rgba(27,42,74,0.03);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 16px 0;
  overflow-x: auto;
}
.formula-anno {
  font-size: 13px;
  color: var(--mt, #6b7280);
  font-family: 'Noto Sans SC', sans-serif;
  margin-top: 8px;
}
```

### Interactions
- Annotation card hover → original paragraph highlight (`data-anno` attribute)
- Highlighted term hover → tooltip Chinese explanation
- Section nav: `IntersectionObserver` tracking
- Fade-in entrance animation
- Guide panel: CSS `max-height` collapse/expand
- Mobile `<768px`: single column + collapsed annotations + hidden legend

---

## Output Requirements

- Single complete `.html` file, all CSS/JS inlined
- Fonts: Google Fonts CDN only; KaTeX from jsdelivr CDN (the only allowed exception)
- All 11 sections present — none may be omitted
- ALL math formulas rendered via KaTeX — no plain-text formulas
- Code has block comments, clear structure
- ALL content must be based on the paper's real content — never fabricate data or methods
- Citations use §section or Tab./Fig. numbers
- Save to: `~/.cursor/paper-db/readers/{paper-id}-reader.html`

## ✅ Self-verification before declaring Phase 10 done

After writing the file, do all of:

1. `ls -la ~/.cursor/paper-db/readers/{paper-id}-reader.html` — confirms file
   exists and shows size (≥ 8 KB target).
2. `Read` the file — confirm Section 1 has the hero figure, Section 5 has the
   architecture (drawio embed or SVG), Section 11 has the worksheet cards.
3. Visually scan: are all 11 sections present? Open the file in a browser if
   feasible and verify KaTeX renders, drawio embed loads.
4. Run `python3 ~/.cursor/paper-db/tools/check_paper_completeness.py
   {paper-id} --strict` — this will explicitly verify the reader file is
   present and large enough, alongside all other paper artifacts.
