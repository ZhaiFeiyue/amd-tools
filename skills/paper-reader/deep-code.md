# deep-code.md — Code (repo / PR / issue) specific additions

The 9 base sections (SKILL.md) apply. Below is what's **code-specific**.

Covers: GitHub repos, PRs, issues, design docs — anything where the
"paper" IS source code.

## §1 Project identity

- Repo / PR / issue URL, primary language, LOC (use `tokei` or `cloc`)
- License, stars, contributor count, last commit date
- Maintainer / sponsor org
- For PR: author, state (open/merged/closed), target branch, LOC changed
- For issue: author, state, labels, comment count, linked PRs

## §2 What & why — motivation analysis

- For repo: one-line pitch (steal from README if good, rewrite if hand-wavy)
- For PR: what problem does this change solve? link the issue it
  addresses. Is this the only way to solve it?
- For issue: is this a bug / feature / design proposal / question?
  How reproducible? Impact scope?

## §3 Architecture & module map (repo)

Mermaid flowchart of top-level modules with:
- Purpose of each module (1 line)
- Inter-module dependency arrows
- External API boundary (what users import)
- Internal API boundary (what modules expose to each other)

For large repos (>30 modules), draw the top 10 by LOC or by dependency
centrality.

## §4 Entry points & API surface (repo)

- Public API: what does `import <pkg>` expose?
- CLI entry points (from `setup.py` / `pyproject.toml` / Makefile)
- Configuration: env vars, config files, CLI flags (top 10 most
  important)
- Extension points: plugin system, hooks, subclasses users are meant
  to override

## §5 Core data structures (repo / PR)

For each key data structure in the hot path:
- Definition (file:line)
- Memory layout (struct / class / dataclass)
- Lifecycle: when created, when destroyed, who owns it
- Mutation rules: mutable? thread-safe? copy-on-write?

## §6 Critical path analysis (repo / PR)

Trace the hot path from user input → result:
- File:line of entry → intermediate layers → hot kernel/loop
- Latency budget at each hop (if measurable)
- Memory traffic at each hop

Compare to what the README / docstring CLAIMS the hot path is. Code
trumps docs — if they disagree, flag it in the notes.

## §7 Change analysis (PR-specific)

- Summary: files / lines changed, test coverage delta
- Architectural shift: does this change a public API? internal invariant?
- Performance impact claimed by author: verify with benchmark rerun if
  possible
- Compatibility: breaking change? deprecation? semver bump?
- Review surface: what are reviewers asking about in comments? Has the
  author addressed each comment?

## §8 Issue triage (issue-specific)

- Classification: bug / perf / feature / design / docs / UX
- Reproduction: does the issue include a minimal repro? Can you reproduce?
- Root cause: is it known? linked PR? open? (find with `gh issue view --json`)
- Related issues: is this a dup? a symptom of a larger problem?

## §9 Concurrency & memory

- Concurrency model: async? threads? processes? GIL?
- Lock hierarchy (if any): what locks in what order?
- Memory management: ref-counted? GC? manual? pool?
- Known concurrency bugs in history: check `git log --grep=race\|deadlock\|TOCTOU`

## §10 Performance characteristics

- Headline benchmark: what does the project publish? on what HW?
- Scaling: how does throughput / latency change with input size, concurrency,
  nodes?
- Profiling: where does the CPU / GPU time go? (run `py-spy` / `nsys` /
  `rocprof` if feasible)
- Regressions: any open issues about perf regression?

## §11 Tech debt & code quality

- Test coverage (`pytest --cov` / similar)
- Linter discipline (ruff / mypy / eslint / clippy config)
- CI matrix: what's tested, what's not
- Known warts: `TODO` / `FIXME` / `XXX` density, any section the
  maintainers openly say "this is a mess"
- Dependency health: major deps, any abandoned?

## §12 Community health

- Issue response time (median time to first maintainer reply)
- PR merge velocity
- Bus factor: how many people have committed in the last 3 months?
- Governance: BDFL, committee, foundation? Corporate-backed?

## §13 Comparison with alternatives

Table of this project vs its 2–3 closest alternatives, rows: features /
performance / ecosystem / license / maintainer credibility. Bold the
winner per row.

## §14 Verdict & recommendations

- When to adopt this (the "yes" regime)
- When NOT to adopt (the "no" regime)
- Suggested changes: what would you merge / propose if you contributed?
