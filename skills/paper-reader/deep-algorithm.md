# deep-algorithm.md — Algorithm-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **algorithm-specific**.

Covers: training recipes, RL methods (PPO/DPO/GRPO/KTO), distillation,
data strategies, optimizer design, loss engineering.

## §2 Problem formulation

- Objective: the exact loss / reward function (write it in LaTeX, not prose)
- Assumptions: Markov? i.i.d. samples? stationarity? posterior mode?
- Inputs / outputs of the algorithm — what does one step consume and
  produce? (e.g. GRPO: group of G completions + scalar reward → updated
  policy)

## §3 Method core — the one novel mechanism

Every algorithm paper has one. Examples:
- GRPO: replace value model with group-relative advantage
- DPO: reparametrize PPO as supervised pair loss
- PASTE: prefix-matching early-exit on act vs think trace
- Polarization: attention-map re-weighting before softmax

State what it is in ≤2 sentences. Draw the diff from the previous method
as a 2-column table: `before` vs `after`.

## §4 作者证明 — Algorithm-specific asks

In addition to the 6 base checks (SKILL.md), algorithm papers often have:

- **Convergence theorem / proposition / lemma**: reproduce the statement;
  list the assumptions; discuss when each assumption breaks
- **Variance / sample complexity bound**: what function of `(N, d, T)`
  does the bound depend on? Is it tight?
- **Loss decomposition**: if loss is a sum of terms, what does each term
  enforce?
- **Proof sketch**: a 3–5 line skeleton that an expert could fill in —
  NOT the full proof, but enough to convey the structure
- If paper has no formal theorem: mark **无形式化作者证明 — 仅实证** and
  describe what kind of guarantee would have been desirable

## §5 Training recipe & scale

For training recipes / data strategy papers:

| Stage | Purpose | Data | Steps | LR | Batch | Special technique |
|---|---|---|---|---|---|---|

Must disclose (or mark `[论文未披露]`):
- Total tokens / samples per stage
- GPU hours (with hardware model) and MFU
- Critical hyperparameters (β for DPO, clip ratio for PPO, group size G
  for GRPO, KL coefficient, etc.)
- Stability tricks (gradient clipping, ZeRO stage, selective checkpointing)

## §5 Convergence & stability (if training recipe)

- Learning curve shape (plateau? divergence? oscillation?)
- Sensitivity to hyperparameters (what's the valid range for β / clip /
  KL?)
- Reward-hacking or over-optimization signals (KL explosion, length
  explosion, reward overfitting)
- Compared to supervised warm-up, where does the RL signal actually add
  value?

## §6 Dataset analysis

- Composition: web / code / math / chat / tool-call — what %?
- Quality filters: what was dropped and why?
- Contamination: any eval overlap check?
- Annotation: if human-labeled, by whom, how many, agreement rate?
- Synthetic data: what generator, what seed prompts, what validation?

## §9 Reproducibility & ecosystem

- Is the training code open? (If not, what's the closest open
  reference — TRL / OpenRLHF / verl / ColossalAI?)
- Has the recipe been re-implemented by the community? Any reports of
  matching or missing the paper's numbers?
- Which current models (production or research) use a variant of this
  recipe?
